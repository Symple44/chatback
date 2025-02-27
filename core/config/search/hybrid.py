# core/config/search/hybrid.py
from typing import Dict, Any

# Configuration Hybride
HYBRID_CONFIG = {
    "model_params": {
        "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "semantic_model": "Mistral-Small-24B-Instruct-2501",
        "device": "cuda"
    },
    "search_params": {
        "max_docs": 6,
        "min_score": 0.25,
        "rag_weight": 0.5,
        "semantic_weight": 0.5,
        "combine_method": "weighted_sum",
        "rerank_top_k": 10,
        "use_sliding_window": True,
        "window_size": 3
    },
    "performance": {
        "use_batching": True,
        "max_batch_size": 16,
        "cache_results": True
    }
}