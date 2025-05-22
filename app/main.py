# app_manual_embeddings.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os

app = FastAPI(title="RAG API - Manual Embeddings")

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
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("✓ Sentence transformer loaded successfully")
        
        # Create ChromaDB client
        chroma_client = chromadb.Client()
        print("✓ ChromaDB client initialized")
        
        # Load constitution data
        data_files = [
            ("constitutia.json", "moldova_constitution"),
            ("data/constitutia.json", "moldova_constitution")
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
                    
                    # Extract data
                    documents = [item["document"] for item in constitutional_documents]
                    metadatas = [item["metadatas"] for item in constitutional_documents]
                    ids = [str(uuid.uuid4()) for _ in range(len(documents))]
                    
                    # Generate embeddings manually
                    print(f"Generating embeddings for {len(documents)} documents...")
                    embeddings = embedder.encode(documents).tolist()
                    print("✓ Embeddings generated successfully")
                    
                    # Add to collection with manual embeddings
                    collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                        embeddings=embeddings
                    )
                    
                    collections[collection_name] = collection
                    print(f"✓ Loaded {len(documents)} documents into {collection_name} collection")
                    break  # Success, don't try other paths
                    
                except Exception as e:
                    print(f"✗ Error loading collection: {e}")
                    import traceback
                    traceback.print_exc()
                    return
        
        if not collections:
            print("⚠ No collections loaded. Make sure constitutia.json exists in current directory or data/ folder")
            
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()

def query_with_manual_embeddings(collection, query_text: str, n_results: int = 3):
    """Query collection using manual embeddings"""
    # Generate embedding for the query
    query_embedding = embedder.encode([query_text]).tolist()
    
    # Query the collection with the embedding
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    
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
        "embedder_loaded": embedder is not None
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "collections": list(collections.keys()),
        "total_collections": len(collections),
        "embedder_ready": embedder is not None
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not collections:
        raise HTTPException(status_code=503, detail="No collections available. Check server logs.")
    
    if not embedder:
        raise HTTPException(status_code=503, detail="Embedder not loaded. Check server logs.")
    
    if request.collection_name not in collections:
        raise HTTPException(
            status_code=404, 
            detail=f"Collection '{request.collection_name}' not found. Available: {list(collections.keys())}"
        )
    
    collection = collections[request.collection_name]
    
    try:
        results = query_with_manual_embeddings(collection, request.query, request.n_results)
        
        return QueryResponse(
            documents=results["documents"][0] if results["documents"] else [],
            metadatas=results["metadatas"][0] if results["metadatas"] else [],
            distances=results["distances"][0] if results["distances"] else []
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/query/{collection_name}")
async def simple_query(collection_name: str, q: str, n_results: int = 3):
    """Simple GET endpoint for testing"""
    if not collections:
        raise HTTPException(status_code=503, detail="No collections available")
    
    if not embedder:
        raise HTTPException(status_code=503, detail="Embedder not loaded")
    
    if collection_name not in collections:
        raise HTTPException(
            status_code=404, 
            detail=f"Collection '{collection_name}' not found. Available: {list(collections.keys())}"
        )
    
    collection = collections[collection_name]
    
    try:
        results = query_with_manual_embeddings(collection, q, n_results)

        distances = results["distances"][0] if results["distances"] else []

        # Convert distances to relevance percentages (rounded to nearest 5)
        relevance = [
            min(100, int(round((1 - d) * 100 / 5.0) * 5))
            for d in distances
        ]

        return {
            "query": q,
            "collection": collection_name,
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": distances,
            "relevance": relevance
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
            "embedding_sample": embedding[0][:5].tolist()  # First 5 dimensions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedder test failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)