# core/config/search/rag.py
from typing import Dict, Any

# Configuration RAG
RAG_CONFIG = {
    "model_params": {
        "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "embedding_device": "cuda",
        "batch_size": 32
    },
    "search_params": {
        "max_docs": 10,
        "min_score": 0.3,
        "vector_weight": 0.7,
        "semantic_weight": 0.3,
        "use_chunking": True,
        "chunk_size": 512,
        "chunk_overlap": 50
    },
    "performance": {
        "cache_embeddings": True,
        "parallel_processing": True,
        "max_concurrent_searches": 3
    }
}