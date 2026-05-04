"""
Multimodal RAG Lakehouse PoC — Configuration
"""
import os

BASE_PATH   = "./lakehouse"
BRONZE_PATH = f"{BASE_PATH}/bronze"
SILVER_PATH = f"{BASE_PATH}/silver"
GOLD_PATH   = f"{BASE_PATH}/gold"

# Bronze tables
BRONZE_DOCS       = f"{BRONZE_PATH}/raw_documents"
BRONZE_RETRIEVAL  = f"{BRONZE_PATH}/raw_retrieval_logs"

# Silver tables
SILVER_CHUNKS     = f"{SILVER_PATH}/document_chunks"
SILVER_EMBEDDINGS = f"{SILVER_PATH}/embeddings"
SILVER_RETRIEVAL  = f"{SILVER_PATH}/retrieval_events"

# Gold tables
GOLD_FRESHNESS    = f"{GOLD_PATH}/embedding_freshness_kpi"
GOLD_QUALITY      = f"{GOLD_PATH}/retrieval_quality_kpi"
GOLD_COST         = f"{GOLD_PATH}/embedding_cost_kpi"

# Data generation parameters
N_DOCS              = 120           # 40 per modality
N_CHUNKS_PER_DOC    = 10            # 1200 chunks total
N_RETRIEVAL_EVENTS  = 3000
VECTOR_DIM          = 128           # prod = 1536; reduced for PoC speed
TENANTS             = ["acme", "beta", "gamma"]
MODALITIES          = ["text", "image_caption", "audio_transcript"]
EMBEDDING_MODEL     = "text-embed-3-large-mock"
COST_PER_1K_TOKENS  = 0.00013       # OpenAI text-embed-3-large pricing

# Threshold
STALE_THRESHOLD_DAYS  = 7
ZERO_HIT_THRESHOLD    = 0.30        # relevance score < này → zero hit

SEED = 42