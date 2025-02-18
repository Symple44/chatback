# core/search/search_manager.py
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
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
        
        # Fusion avec les paramètres par défaut
        default_params = self.config.get(
            method.value, 
            self.config["rag"]
        )["search_params"].copy()
        
        # Mise à jour avec les nouveaux paramètres validés
        validated_params = self._validate_search_params(search_params, method)
        default_params.update(validated_params)
        
        self.current_params = default_params
        self.metadata_filter = metadata_filter
        self.enabled = method != SearchMethod.DISABLED
        
        logger.info(f"Recherche configurée: {method.value}, params: {self.current_params}")

    def _validate_search_params(
        self,
        params: Dict[str, Any],
        method: SearchMethod
    ) -> Dict[str, Any]:
        """Valide et normalise les paramètres de recherche."""
        validated = {}
        default_config = self.config.get(method.value, self.config["rag"])["search_params"]
        
        # Validation des paramètres spécifiques à la méthode
        for key, value in params.items():
            if key in default_config:
                if key == "max_docs":
                    validated[key] = min(max(1, value), settings.SEARCH_MAX_DOCS)
                elif key == "min_score":
                    validated[key] = min(max(0.0, value), 1.0)
                elif key in ["vector_weight", "semantic_weight"]:
                    validated[key] = min(max(0.0, value), 1.0)
                else:
                    validated[key] = value
                    
        return validated
    
    async def search_context(
        self,
        query: str,
        metadata_filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Dict]:
        """Effectue la recherche selon la configuration actuelle."""
        if not self.enabled:
            logger.debug("Recherche désactivée")
            return []

        try:
            # Vérification du cache
            cache_key = self._get_cache_key(query, metadata_filter)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.debug("Résultat trouvé en cache")
                metrics.increment_counter("search_cache_hits")
                return cached_result

            start_time = datetime.utcnow()

            # Fusion des paramètres de recherche
            search_params = self.current_params.copy()
            search_params.update(kwargs)
            
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

            strategy.configure(search_params)
            results = await strategy.search(
                query=query,
                metadata_filter=combined_filter or None
            )

            # Calcul du temps de traitement
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Mise à jour des métriques
            self._update_metrics(
                method=self.current_method,
                results_count=len(results),
                processing_time=processing_time
            )

            # Mise en cache des résultats
            self._cache_results(cache_key, results)

            return results

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