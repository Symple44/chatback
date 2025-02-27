# core/config/search/base.py
from enum import Enum
from typing import Dict, Any

class SearchMethod(str, Enum):
    """Types de méthodes de recherche disponibles."""
    DISABLED = "disabled"  # Pas de recherche de contexte
    RAG = "rag"           # Retrieval Augmented Generation classique
    HYBRID = "hybrid"     # Combinaison de RAG et recherche sémantique
    SEMANTIC = "semantic" # Recherche purement sémantique

# Paramètres par défaut communs
search_defaults = {
    "max_docs": 20,
    "min_score": 0.3,
    "cache_ttl": 3600,
    "max_concurrent_searches": 10
}

# Limites et contraintes
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