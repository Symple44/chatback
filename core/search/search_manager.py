# core/search/search_manager.py
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
from api.models.responses import SearchResult
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings
from .strategies import SearchMethod, SearchStrategy
from .factory import SearchStrategyFactory
from core.config.search_config import SEARCH_STRATEGIES_CONFIG

logger = get_logger("search_manager")

class SearchManager:
    """Gestionnaire de recherche avec configuration dynamique."""
    
    def __init__(self, components):
        """Initialise le gestionnaire de recherche."""
        self.components = components
        self.config = SEARCH_STRATEGIES_CONFIG
        self.enabled = True
        self.current_method = SearchMethod.RAG
        self.current_params = self.config["rag"]["search_params"].copy()  # Paramètres par défaut
        self.metadata_filter = None
        self.cache = {}
        self.cache_ttl = settings.SEARCH_CACHE_TTL
        
    async def configure(
        self,
        method: SearchMethod,
        search_params: Dict[str, Any],
        metadata_filter: Optional[Dict] = None
    ):
        """Configure la stratégie de recherche."""
        self.current_method = method
        
        # Récupérer d'abord les paramètres par défaut de la config
        base_params = self.config.get(
            method.value, 
            self.config["rag"]
        )["search_params"].copy()
        
        # Les paramètres utilisateur écrasent les paramètres par défaut
        self.current_params = {
            **base_params,  # Paramètres de base depuis SEARCH_STRATEGIES_CONFIG
            **{k: v for k, v in search_params.items() if v is not None}  # Ne prendre que les valeurs non nulles
        }
        
        # Validation des paramètres
        self.current_params = self._validate_search_params(self.current_params, method)
        
        self.metadata_filter = metadata_filter
        self.enabled = method != SearchMethod.DISABLED
            
        logger.info(f"Recherche configurée: {method.value}, params: {self.current_params}")

    def _validate_search_params(
        self,
        params: Dict[str, Any],
        method: SearchMethod
    ) -> Dict[str, Any]:
        validated = {}
        default_config = self.config.get(method.value, self.config["rag"])["search_params"]
        
        # Ne permettre que les paramètres connus
        allowed_params = set(default_config.keys())
        
        for key, value in params.items():
            if key not in allowed_params:
                logger.warning(f"Paramètre ignoré car non autorisé: {key}")
                continue
                
            original_value = value
            if key == "max_docs":
                validated[key] = min(max(1, int(value)), 20)
                if validated[key] != value:
                    logger.info(f"max_docs ajusté de {value} à {validated[key]} pour respecter les limites [1-20]")
            elif key == "min_score":
                validated[key] = min(max(0.0, float(value)), 1.0)
                if validated[key] != value:
                    logger.info(f"min_score ajusté de {value} à {validated[key]} pour respecter les limites [0-1]")
            elif key in ["vector_weight", "semantic_weight"]:
                validated[key] = min(max(0.0, float(value)), 1.0)
                if validated[key] != value:
                    logger.info(f"{key} ajusté de {value} à {validated[key]} pour respecter les limites [0-1]")
            else:
                validated[key] = value
                
        return validated

    async def search_context(
        self,
        query: str,
        metadata_filter: Optional[Dict] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Effectue la recherche selon la configuration actuelle."""
        if not self.enabled:
            logger.debug("Recherche désactivée")
            return []

        try:
            # Vérification du cache
            cache_key = self._get_cache_key(query, metadata_filter)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                # Pour les résultats en cache, respecter aussi max_docs
                max_docs = kwargs.get("max_docs", self.current_params.get("max_docs", 10))
                logger.debug(f"Utilisation de max_docs={max_docs} pour les résultats en cache")
                return cached_result[:max_docs]

            # Fusion des paramètres de recherche - priorité aux paramètres spécifiques
            search_params = {
                **self.current_params,
                **kwargs  # kwargs écrase les paramètres de la configuration
            }
            
            # Fusion des filtres de métadonnées
            combined_filter = {}
            if self.metadata_filter:
                combined_filter.update(self.metadata_filter)
            if metadata_filter:
                combined_filter.update(metadata_filter)

            # Création et configuration de la stratégie
            strategy = SearchStrategyFactory.create_strategy(
                method=self.current_method,
                components=self.components
            )

            strategy.configure(search_params)  # Utilise les paramètres fusionnés
            results = await strategy.search(
                query=query,
                metadata_filter=combined_filter or None
            )

            # Mise en cache des résultats COMPLETS
            self._cache_results(cache_key, results)

            # Retourne les résultats limités par max_docs
            max_docs = search_params.get("max_docs", 10)
            return results[:max_docs]

        except Exception as e:
            logger.error(f"Erreur recherche contexte: {e}", exc_info=True)
            metrics.increment_counter("search_errors")
            return []

    def _get_cache_key(self, query: str, metadata_filter: Optional[Dict]) -> str:
        """Génère une clé de cache unique."""
        filter_str = str(sorted(metadata_filter.items())) if metadata_filter else ""
        return f"{self.current_method.value}:{query}:{filter_str}"

    def _get_from_cache(self, key: str) -> Optional[List[Dict]]:
        """Récupère un résultat du cache."""
        cached = self.cache.get(key)
        if not cached:
            metrics.track_cache_operation(hit=False)
            return None
            
        timestamp, results = cached
        if (datetime.utcnow() - timestamp).seconds > self.cache_ttl:
            del self.cache[key]
            metrics.track_cache_operation(hit=False)
            return None
        
        metrics.track_cache_operation(hit=True)
        return results

    def _cache_results(self, key: str, results: List[Dict]):
        """Met en cache les résultats de recherche."""
        self.cache[key] = (datetime.utcnow(), results)
        
        # Nettoyage du cache si nécessaire
        if len(self.cache) > 1000:  # Limite arbitraire
            oldest = sorted(self.cache.items(), key=lambda x: x[1][0])[0][0]
            del self.cache[oldest]

    def _update_metrics(self, method: SearchMethod, results_count: int, processing_time: float):
        """Met à jour les métriques de recherche."""
        metrics.increment_counter(f"search_{method.value}_total")
        metrics.increment_counter(f"search_{method.value}_results", results_count)
        metrics.record_time(f"search_{method.value}_time", processing_time)

    async def cleanup(self):
        """Nettoie les ressources."""
        self.cache.clear()
        logger.info("Ressources SearchManager nettoyées")