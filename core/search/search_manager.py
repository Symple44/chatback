# core/search/search_manager.py
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings
from .strategies import SearchMethod, SearchStrategy
from .factory import SearchStrategyFactory

logger = get_logger("search_manager")

class SearchManager:
    """Gestionnaire de recherche avec configuration dynamique."""
    
    def __init__(self, components):
        """Initialise le gestionnaire de recherche."""
        self.components = components
        self.enabled = True
        self.current_method = SearchMethod.RAG
        self.current_params = {}
        self.metadata_filter = None
        self.cache = {}
        self.cache_ttl = 3600  # 1 heure
        
    def configure(
        self,
        method: SearchMethod,
        search_params: Dict[str, Any],
        metadata_filter: Optional[Dict] = None
    ):
        """Configure la stratégie de recherche."""
        self.current_method = method
        self.current_params = search_params
        self.metadata_filter = metadata_filter
        self.enabled = method != SearchMethod.DISABLED
        logger.info(f"Recherche configurée: {method.value}, params: {search_params}")

    async def search_context(
        self,
        query: str,
        metadata_filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Effectue la recherche selon la configuration actuelle.
        
        Args:
            query: Requête utilisateur
            metadata_filter: Filtres optionnels
            **kwargs: Paramètres additionnels
            
        Returns:
            Liste des documents trouvés
        """
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

            # Création de la stratégie via factory
            strategy = SearchStrategyFactory.create_strategy(
                method=self.current_method,
                components=self.components
            )

            # Configuration de la stratégie
            strategy.configure({
                **self.current_params,
                **kwargs
            })

            # Exécution de la recherche
            start_time = datetime.utcnow()
            results = await strategy.search(
                query=query,
                metadata_filter=metadata_filter or self.metadata_filter
            )
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Mise à jour des métriques
            self._update_metrics(
                method=self.current_method,
                results_count=len(results),
                processing_time=processing_time
            )

            # Mise en cache
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
            return None
            
        timestamp, results = cached
        if (datetime.utcnow() - timestamp).seconds > self.cache_ttl:
            del self.cache[key]
            return None
            
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