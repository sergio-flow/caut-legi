# app_manual_embeddings.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os
import anthropic
from typing import List, Dict, Any

app = FastAPI(title="RAG API - Manual Embeddings")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

data_dir_to_serve = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../data"
)
app.mount(
    "/data",
    StaticFiles(directory=data_dir_to_serve),
    name="data",
)

# Initialize Anthropic client (make sure to set your API key)
anthropic_client = anthropic.Anthropic(
    api_key="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"
)

# Global variables
chroma_client = None
collections = {}
embedder = None


class QueryRequest(BaseModel):
    query: str
    collection_name: str = "moldova_constitution"
    n_results: int = 3


class QueryResponse(BaseModel):
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    distances: List[float]


def initialize_chroma():
    """Initialize ChromaDB client and load data with manual embeddings"""
    global chroma_client, collections, embedder

    try:
        # Initialize the sentence transformer
        print("Loading sentence transformer model...")
        embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        print("✓ Sentence transformer loaded successfully")

        # Create ChromaDB client
        chroma_client = chromadb.Client()
        print("✓ ChromaDB client initialized")

        # Load constitution data
        data_files = [
            # ("constitutia.json", "moldova_constitution"),
            ("data/constitutia.json", "moldova_constitution"),
        ]

        for file_path, collection_name in data_files:
            if os.path.exists(file_path):
                print(f"Loading data from {file_path}...")

                try:
                    # Create collection WITHOUT embedding function (we'll provide embeddings manually)
                    collection = chroma_client.create_collection(name=collection_name)

                    # Load data
                    with open(file_path, "r", encoding="utf-8") as f:
                        constitutional_documents = json.load(f)

                    # Prepare documents and metadata
                    documents = []
                    metadatas = []
                    ids = []

                    for item in constitutional_documents:
                        # Required fields
                        article_number = item.get("article_number")
                        laws = item.get("laws", [])

                        if not article_number or not laws:
                            print(
                                f"Skipping item due to missing required fields: {item}"
                            )
                            continue

                        # Combine the list of laws into a single string document
                        document_text = " ".join(laws)
                        documents.append(document_text)

                        # Build metadata dynamically from all optional fields
                        metadata = {"article_number": article_number}

                        # Optional fields
                        optional_keys = [
                            "category_number",
                            "category_description",
                            "chapter_number",
                            "chapter_description",
                            "section_number",
                            "section_description",
                            "article_description",
                        ]
                        for key in optional_keys:
                            if key in item:
                                metadata[key] = item[key]

                        metadatas.append(metadata)
                        ids.append(str(uuid.uuid4()))

                    # Generate embeddings manually
                    print(f"Generating embeddings for {len(documents)} documents...")
                    embeddings = embedder.encode(documents).tolist()
                    print("✓ Embeddings generated successfully")

                    # Add to Chroma collection
                    collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                        embeddings=embeddings,
                    )

                    collections[collection_name] = collection
                    print(
                        f"✓ Loaded {len(documents)} documents into {collection_name} collection"
                    )
                    break  # Success, don't try other paths

                except Exception as e:
                    print(f"✗ Error loading collection: {e}")
                    import traceback

                    traceback.print_exc()
                    return

        if not collections:
            print(
                "⚠ No collections loaded. Make sure constitutia.json exists in current directory or data/ folder"
            )

    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        import traceback

        traceback.print_exc()


def query_with_manual_embeddings(collection, query_text: str, n_results: int = 3):
    """Query collection using manual embeddings"""
    # Generate embedding for the query
    query_embedding = embedder.encode([query_text]).tolist()

    # Query the collection with the embedding
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    return results


# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_chroma()


@app.get("/")
async def root():
    return {
        "message": "RAG API is running with manual embeddings",
        "collections": list(collections.keys()),
        "status": "ready" if collections else "no collections loaded",
        "embedder_loaded": embedder is not None,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "collections": list(collections.keys()),
        "total_collections": len(collections),
        "embedder_ready": embedder is not None,
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not collections:
        raise HTTPException(
            status_code=503, detail="No collections available. Check server logs."
        )

    if not embedder:
        raise HTTPException(
            status_code=503, detail="Embedder not loaded. Check server logs."
        )

    if request.collection_name not in collections:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{request.collection_name}' not found. Available: {list(collections.keys())}",
        )

    collection = collections[request.collection_name]

    try:
        results = query_with_manual_embeddings(
            collection, request.query, request.n_results
        )

        return QueryResponse(
            documents=results["documents"][0] if results["documents"] else [],
            metadatas=results["metadatas"][0] if results["metadatas"] else [],
            distances=results["distances"][0] if results["distances"] else [],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


async def analyze_with_claude(document: str, query: str) -> tuple[str, str]:
    """
    Make two calls to Claude for document analysis
    Returns: (highlighted_sentences, explanation)
    """

    # First call: Get relevant sentences
#     first_prompt = f"""Document:
# {document}

# User's query: {query}

# Please analyze the document and identify the most relevant laws/sentences that relate to the user's query. Return only a comma-separated list of the exact sentences from the document that are most relevant. Do not add any explanations or additional text - just the sentences separated by commas."""

#     try:
#         first_response = anthropic_client.messages.create(
#             model="claude-3-5-haiku-20241022",
#             max_tokens=1000,
#             messages=[{"role": "user", "content": first_prompt}],
#         )
#         highlighted_sentences = first_response.content[0].text.strip()
#     except Exception as e:
#         highlighted_sentences = f"Error in first AI call: {str(e)}"

    # Second call: Get explanation in Romanian
    prompt = f"""Document original:
{document}

Întrebarea utilizatorului: {query}

Te rog să explici în română, în termeni simpli, de ce exact aceste legi sunt relevante pentru întrebarea utilizatorului. Oferă o explicație clară și ușor de înțeles.
Maxim: 5 propoziții. Maxim: 50 cuvinte. Maxim: 300 caractere. Ofera doar explicația, poate si un exemplu usor de inteles, fără alte detalii.
Daca nu gasești o explicatie relevanta pe baza ce cauta utilizatorul, atunci nu raspunde la întrebare. Returneaza doar un mesaj gol.
"""

    try:
        second_response = anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        explanation = second_response.content[0].text.strip()
    except Exception as e:
        explanation = f"Error in AI call: {str(e)}"

    return explanation


@app.get("/query/{collection_name}")
async def simple_query(collection_name: str, q: str, n_results: int = 3):
    """Enhanced GET endpoint with AI analysis for high-quality results"""
    if not collections:
        raise HTTPException(status_code=503, detail="No collections available")

    if not embedder:
        raise HTTPException(status_code=503, detail="Embedder not loaded")

    if collection_name not in collections:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found. Available: {list(collections.keys())}",
        )

    collection = collections[collection_name]

    try:
        results = query_with_manual_embeddings(collection, q, n_results)
        distances = results["distances"][0] if results["distances"] else []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        # Filter results to only include Good, Very good, and Excellent matches
        def filter_high_quality_results(distances, documents, metadatas):
            filtered_results = {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "relevance_labels": [],
                "relevance_scores": [],
            }

            for i, d in enumerate(distances):
                # Only include results with distance < 8.0 (Good or better)
                if d < 8.0:
                    if d < 5.0:
                        score = 100
                        label = "Excellent match"
                    elif d < 7.0:
                        score = 90
                        label = "Very good"
                    else:  # d < 8.0
                        score = 80
                        label = "Good"

                    filtered_results["documents"].append(documents[i])
                    filtered_results["metadatas"].append(metadatas[i])
                    filtered_results["distances"].append(d)
                    filtered_results["relevance_labels"].append(label)
                    filtered_results["relevance_scores"].append(score)

            return filtered_results

        filtered_results = filter_high_quality_results(distances, documents, metadatas)

        # If no high-quality results found, return empty response
        if not filtered_results["documents"]:
            return {
                "query": q,
                "collection": collection_name,
                "message": "No high-quality matches found (Good or better)",
                "documents": [],
                "metadatas": [],
                "distances": [],
                "relevance": [],
                "relevance_scores": [],
                "ai_analyses": [],
            }

        # Perform AI analysis on all high-quality documents (score >= 80)
        ai_analyses = []
        for i, document in enumerate(filtered_results["documents"]):
            if filtered_results["relevance_scores"][i] >= 80:
                explanation = await analyze_with_claude(
                    document, q
                )
                ai_analyses.append(
                    {
                        "document_index": i,
                        "relevance_score": filtered_results["relevance_scores"][i],
                        "relevance_label": filtered_results["relevance_labels"][i],
                        # "highlighted_sentences": highlighted_sentences,
                        "explanation": explanation,
                    }
                )

        return {
            "query": q,
            "collection": collection_name,
            "documents": filtered_results["documents"],
            "metadatas": filtered_results["metadatas"],
            "distances": filtered_results["distances"],
            "relevance": filtered_results["relevance_labels"],
            "relevance_scores": filtered_results["relevance_scores"],
            "ai_analyses": ai_analyses,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/test-embedder")
async def test_embedder():
    """Test endpoint to verify embedder is working"""
    if not embedder:
        raise HTTPException(status_code=503, detail="Embedder not loaded")

    try:
        test_text = "This is a test sentence"
        embedding = embedder.encode([test_text])
        return {
            "status": "success",
            "test_text": test_text,
            "embedding_shape": embedding.shape,
            "embedding_sample": embedding[0][:5].tolist(),  # First 5 dimensions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedder test failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
