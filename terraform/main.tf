# terraform/main.tf (Infrastructure as Code)
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Cloud Run Service
resource "google_cloud_run_service" "rag_api" {
  name     = "rag-api"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/rag-api:latest"
        
        resources {
          limits = {
            cpu    = "1000m"
            memory = "2Gi"
          }
        }

        env {
          name  = "ENVIRONMENT"
          value = "production"
        }

        ports {
          container_port = 8080
        }
      }

      container_concurrency = 10
      timeout_seconds      = 300
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = "10"
        "run.googleapis.com/cpu-throttling" = "false"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# IAM policy for public access
resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.rag_api.name
  location = google_cloud_run_service.rag_api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Output the service URL
output "service_url" {
  value = google_cloud_run_service.rag_api.status[0].url
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}