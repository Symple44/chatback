# core/config/search/__init__.py
import os
from enum import Enum
from typing import Dict, Any

from .base import SearchMethod, search_defaults
from .rag import RAG_CONFIG
from .hybrid import HYBRID_CONFIG
from .semantic import SEMANTIC_CONFIG

class SearchConfig:
    """Configuration unifiée de recherche."""
    
    def __init__(self):
        # Configurations par défaut
        self.DEFAULT_SEARCH_METHOD = os.getenv("DEFAULT_SEARCH_METHOD", "rag")
        self.SEARCH_MAX_DOCS = int(os.getenv("SEARCH_MAX_DOCS", "20"))
        self.SEARCH_MIN_SCORE = float(os.getenv("SEARCH_MIN_SCORE", "0.3"))
        self.SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "3600"))
        self.SEARCH_MAX_CONCURRENT = int(os.getenv("SEARCH_MAX_CONCURRENT", "8"))
        
        # Configurations RAG
        self.RAG_VECTOR_WEIGHT = float(os.getenv("RAG_VECTOR_WEIGHT", "0.7"))
        self.RAG_SEMANTIC_WEIGHT = float(os.getenv("RAG_SEMANTIC_WEIGHT", "0.3"))
        
        # Configurations hybride
        self.HYBRID_RERANK_TOP_K = int(os.getenv("HYBRID_RERANK_TOP_K", "10"))
        self.HYBRID_WINDOW_SIZE = int(os.getenv("HYBRID_WINDOW_SIZE", "3"))
        
        # Configurations sémantique
        self.SEMANTIC_MAX_CONCEPTS = int(os.getenv("SEMANTIC_MAX_CONCEPTS", "5"))
        self.SEMANTIC_BOOST_EXACT = os.getenv("SEMANTIC_BOOST_EXACT", "true").lower() == "true"
        
        # Stratégies disponibles
        self.strategies = {
            "rag": RAG_CONFIG,
            "hybrid": HYBRID_CONFIG,
            "semantic": SEMANTIC_CONFIG
        }
    
    def get_strategy_config(self, method: str) -> Dict[str, Any]:
        """
        Récupère la configuration d'une stratégie de recherche.
        
        Args:
            method: Nom de la méthode
            
        Returns:
            Configuration de la stratégie
        """
        if method not in self.strategies:
            method = self.DEFAULT_SEARCH_METHOD
        
        base_config = self.strategies[method]
        
        # Mise à jour des paramètres avec les valeurs d'environnement
        if method == "rag":
            base_config["search_params"]["vector_weight"] = self.RAG_VECTOR_WEIGHT
            base_config["search_params"]["semantic_weight"] = self.RAG_SEMANTIC_WEIGHT
        elif method == "hybrid":
            base_config["search_params"]["rerank_top_k"] = self.HYBRID_RERANK_TOP_K
            base_config["search_params"]["window_size"] = self.HYBRID_WINDOW_SIZE
        elif method == "semantic":
            base_config["search_params"]["max_concepts"] = self.SEMANTIC_MAX_CONCEPTS
            base_config["search_params"]["boost_exact_matches"] = self.SEMANTIC_BOOST_EXACT
        
        # Paramètres communs
        base_config["search_params"]["max_docs"] = self.SEARCH_MAX_DOCS
        base_config["search_params"]["min_score"] = self.SEARCH_MIN_SCORE
        
        return base_config

__all__ = ['SearchConfig', 'SearchMethod']