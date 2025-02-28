import os
from enum import Enum
from typing import Dict, Any, Optional

from .base import SearchMethod, search_defaults, SEARCH_LIMITS, ERROR_HANDLING, PERFORMANCE_CONFIG
from .rag import RAG_CONFIG
from .hybrid import HYBRID_CONFIG
from .semantic import SEMANTIC_CONFIG

class SearchConfig:
    """
    Gestion centralisée de la configuration de recherche.
    Permet de configurer et personnaliser différentes stratégies de recherche.
    """
    
    def __init__(self):
        """
        Initialise la configuration de recherche avec des variables d'environnement.
        Charge et valide les configurations pour différentes stratégies de recherche.
        """
        # Configuration générale de recherche
        self.DEFAULT_SEARCH_METHOD = os.getenv("DEFAULT_SEARCH_METHOD", "rag")
        self.SEARCH_MAX_DOCS = int(os.getenv("SEARCH_MAX_DOCS", "20"))
        self.SEARCH_MIN_SCORE = float(os.getenv("SEARCH_MIN_SCORE", "0.3"))
        self.SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "3600"))
        self.SEARCH_MAX_CONCURRENT = int(os.getenv("SEARCH_MAX_CONCURRENT", "8"))
        
        # Configurations spécifiques aux stratégies
        self.RAG_VECTOR_WEIGHT = float(os.getenv("RAG_VECTOR_WEIGHT", "0.7"))
        self.RAG_SEMANTIC_WEIGHT = float(os.getenv("RAG_SEMANTIC_WEIGHT", "0.3"))
        
        self.HYBRID_RERANK_TOP_K = int(os.getenv("HYBRID_RERANK_TOP_K", "10"))
        self.HYBRID_WINDOW_SIZE = int(os.getenv("HYBRID_WINDOW_SIZE", "3"))
        
        self.SEMANTIC_MAX_CONCEPTS = int(os.getenv("SEMANTIC_MAX_CONCEPTS", "5"))
        self.SEMANTIC_BOOST_EXACT = os.getenv("SEMANTIC_BOOST_EXACT", "true").lower() == "true"
        
        # Stratégies de recherche disponibles
        self.strategies = {
            "rag": RAG_CONFIG.copy(),
            "hybrid": HYBRID_CONFIG.copy(),
            "semantic": SEMANTIC_CONFIG.copy()
        }
        
    def get_all_search_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Récupère les configurations de tous les modes de recherche disponibles.
        
        Retourne:
            Un dictionnaire des configurations de méthodes de recherche
        """
        return {method: self.get_strategy_config(method) for method in self.strategies.keys()}

    def get_strategy_config(self, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère la configuration pour une stratégie de recherche spécifique.
        
        Args:
            method: Nom de la méthode de recherche. Par défaut, utilise la méthode configurée.
            
        Retourne:
            Dictionnaire de configuration pour la stratégie de recherche spécifiée
        """
        # Utilise la méthode par défaut si non spécifiée ou invalide
        if not method or method not in self.strategies:
            method = self.DEFAULT_SEARCH_METHOD
        
        # Crée une copie profonde de la configuration de base pour éviter les modifications
        base_config = self.strategies[method].copy()
        
        # Met à jour les paramètres spécifiques à la stratégie
        search_params = base_config.get("search_params", {})
        
        if method == "rag":
            # Configuration spécifique à la stratégie RAG
            search_params["vector_weight"] = self.RAG_VECTOR_WEIGHT
            search_params["semantic_weight"] = self.RAG_SEMANTIC_WEIGHT
        elif method == "hybrid":
            # Configuration spécifique à la stratégie hybride
            search_params["rerank_top_k"] = self.HYBRID_RERANK_TOP_K
            search_params["window_size"] = self.HYBRID_WINDOW_SIZE
        elif method == "semantic":
            # Configuration spécifique à la stratégie sémantique
            search_params["max_concepts"] = self.SEMANTIC_MAX_CONCEPTS
            search_params["boost_exact_matches"] = self.SEMANTIC_BOOST_EXACT
        
        # Paramètres communs à toutes les stratégies
        search_params["max_docs"] = self.SEARCH_MAX_DOCS
        search_params["min_score"] = self.SEARCH_MIN_SCORE
        
        base_config["search_params"] = search_params
        
        return base_config

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Valide la configuration de recherche par rapport aux limites prédéfinies.
        
        Args:
            config: Dictionnaire de configuration à valider
            
        Retourne:
            Booléen indiquant si la configuration est valide
        """
        try:
            # Valide le nombre maximum de documents
            max_docs = config.get("search_params", {}).get("max_docs", 0)
            if max_docs < 1 or max_docs > SEARCH_LIMITS.get("max_docs", 100):
                return False
            
            # Valide le score minimum
            min_score = config.get("search_params", {}).get("min_score", 0)
            if min_score < 0 or min_score > 1:
                return False
            
            # Ajouter des validations spécifiques supplémentaires si nécessaire
            return True
        except Exception:
            return False

__all__ = ['SearchConfig']