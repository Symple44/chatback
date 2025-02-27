# core/config/search/semantic.py
from typing import Dict, Any

# Configuration Sémantique
SEMANTIC_CONFIG = {
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