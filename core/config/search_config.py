# core/config/search_config.py
from typing import Dict, Any

# Configuration des stratégies de recherche
SEARCH_STRATEGIES_CONFIG = {
    "rag": {
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
    },
    "hybrid": {
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
    },
    "semantic": {
        "model_params": {
            "model": "Mistral-Small-24B-Instruct-2501",
            "device": "cuda",
            "max_tokens": 100
        },
        "search_params": {
            "max_docs": 4,
            "min_score": 0.4,
            "use_concepts": True,
            "max_concepts": 5,
            "consider_synonyms": True,
            "boost_exact_matches": True,
            "include_metadata": True
        },
        "performance": {
            "use_query_expansion": True,
            "cache_duration": 3600,
            "max_query_length": 200
        }
    }
}

# Configuration des limites et seuils
SEARCH_LIMITS = {
    "max_total_docs": 20,
    "max_query_length": 500,
    "min_confidence": 0.1,
    "cache_ttl": 3600,
    "max_retries": 3,
    "timeout": 30.0
}

# Configuration de la gestion d'erreurs
ERROR_HANDLING = {
    "fallback_to_simple": True,
    "max_retries": 3,
    "retry_delay": 1.0,
    "log_errors": True,
    "error_threshold": 0.8
}

# Configuration des optimisations de performance
PERFORMANCE_CONFIG = {
    "use_caching": True,
    "cache_ttl": 3600,
    "batch_size": 32,
    "max_concurrent_requests": 10,
    "use_multithreading": True,
    "thread_pool_size": 4
}