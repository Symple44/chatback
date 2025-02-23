# core/search/strategies/rag_strategy.py
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from ..interfaces import (
    SearchContext,
    SearchResultBase,
    SearchSourceCapabilities,
    SourceType
)
from .base import SearchStrategy
from ..exceptions import SearchProcessingError
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger(__name__)

class RAGStrategy(SearchStrategy):
    """
    Stratégie de Recherche RAG (Retrieval Augmented Generation).
    Combine recherche vectorielle et textuelle.
    """

    def __init__(
        self, 
        sources: List['BaseDataSource'],
        vector_weight: float = 0.7,
        semantic_weight: float = 0.3
    ):
        """
        Initialise la stratégie RAG.
        
        Args:
            sources: Liste des sources de données
            vector_weight: Poids pour les scores vectoriels
            semantic_weight: Poids pour les scores sémantiques
        """
        super().__init__(sources)
        self.vector_weight = vector_weight
        self.semantic_weight = semantic_weight
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Valide les poids de la stratégie."""
        total = self.vector_weight + self.semantic_weight
        if not (0.99 <= total <= 1.01):  # Permettre une petite marge d'erreur
            raise SearchProcessingError(
                f"La somme des poids doit être égale à 1, reçu: {total}",
                details={
                    "vector_weight": self.vector_weight,
                    "semantic_weight": self.semantic_weight
                }
            )

    def get_required_capabilities(self) -> List[SearchSourceCapabilities]:
        """Retourne les capacités requises pour cette stratégie."""
        return [
            SearchSourceCapabilities.FULL_TEXT_SEARCH,
            SearchSourceCapabilities.VECTOR_SEARCH
        ]

    async def search(self, context: SearchContext) -> List[SearchResultBase]:
        """
        Effectue une recherche avec la stratégie RAG.
        
        Args:
            context: Contexte de la recherche
            
        Returns:
            Liste des résultats de recherche
        """
        try:
            with metrics.timer("rag_search"):
                # Configuration des poids par source si non définis
                if not self.source_weights:
                    default_weights = self._get_default_weights()
                    self.set_source_weights(default_weights)

                # Recherche parallèle dans toutes les sources
                search_tasks = []
                for source in self.sources:
                    # Cloner le contexte pour chaque source
                    source_context = self._create_source_context(context, source.get_source_type())
                    search_tasks.append(source.search(source_context))

                # Exécution parallèle des recherches
                results_by_source = {}
                all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

                # Traitement des résultats par source
                for source, results in zip(self.sources, all_results):
                    if isinstance(results, Exception):
                        logger.error(f"Erreur dans la source {source.get_source_type()}: {results}")
                        continue
                    results_by_source[source.get_source_type()] = results

                # Fusion et scoring des résultats
                merged_results = await self._merge_results(results_by_source, context)

                # Mise à jour des métriques
                self._update_metrics(merged_results)

                return merged_results

        except Exception as e:
            logger.error(f"Erreur dans la stratégie RAG: {e}")
            metrics.increment_counter("rag_search_errors")
            raise SearchProcessingError(str(e))

    def _get_default_weights(self) -> Dict[SourceType, float]:
        """Définit les poids par défaut pour chaque type de source."""
        source_types = [source.get_source_type() for source in self.sources]
        weight_per_source = 1.0 / len(source_types)
        return {source_type: weight_per_source for source_type in source_types}

    def _create_source_context(
        self,
        context: SearchContext,
        source_type: SourceType
    ) -> SearchContext:
        """
        Crée un contexte spécifique pour une source.
        
        Args:
            context: Contexte de base
            source_type: Type de la source
            
        Returns:
            Contexte adapté à la source
        """
        # Clone le contexte
        source_context = SearchContext(
            query=context.query,
            language=context.language,
            max_results=context.max_results,
            min_score=context.min_score,
            filters=context.filters.copy() if context.filters else None,
            metadata=context.metadata.copy() if context.metadata else None
        )

        # Adaptations spécifiques selon le type de source
        if source_type == SourceType.ELASTICSEARCH:
            source_context.max_results *= 2  # Plus de résultats pour le ranking
        elif source_type == SourceType.FILE:
            # Filtres spécifiques pour les fichiers
            if source_context.filters is None:
                source_context.filters = {}
            source_context.filters["supported_only"] = True

        return source_context

    async def _merge_results(
        self,
        results_by_source: Dict[SourceType, List[SearchResultBase]],
        context: SearchContext
    ) -> List[SearchResultBase]:
        """
        Fusionne et score les résultats de toutes les sources.
        
        Args:
            results_by_source: Résultats groupés par source
            context: Contexte de la recherche
            
        Returns:
            Liste fusionnée des résultats
        """
        try:
            all_results = []
            
            for source_type, results in results_by_source.items():
                source_weight = self.source_weights.get(source_type, 0.0)
                
                for result in results:
                    # Calcul du score combiné
                    vector_score = result.score * self.vector_weight
                    semantic_score = self._calculate_semantic_score(
                        result,
                        context.query
                    ) * self.semantic_weight
                    
                    # Score final avec poids de la source
                    final_score = (vector_score + semantic_score) * source_weight
                    
                    # Mise à jour du score et des métadonnées
                    result.score = final_score
                    if result.metadata.extra is None:
                        result.metadata.extra = {}
                    result.metadata.extra.update({
                        "vector_score": vector_score,
                        "semantic_score": semantic_score,
                        "source_weight": source_weight,
                        "ranking_info": {
                            "method": "rag",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    })
                    
                    all_results.append(result)
            
            # Tri final et limitation
            sorted_results = sorted(
                all_results,
                key=lambda x: x.score,
                reverse=True
            )
            
            return sorted_results[:context.max_results]

        except Exception as e:
            logger.error(f"Erreur fusion résultats: {e}")
            raise SearchProcessingError(f"Erreur lors de la fusion des résultats: {str(e)}")

    def _calculate_semantic_score(self, result: SearchResultBase, query: str) -> float:
        """
        Calcule un score sémantique entre le résultat et la requête.
        
        Args:
            result: Résultat à scorer
            query: Requête originale
            
        Returns:
            Score sémantique entre 0 et 1
        """
        try:
            # Pour l'instant, utilisation d'une méthode simple
            # TODO: Implémenter un vrai scoring sémantique
            query_words = set(query.lower().split())
            content_words = set(result.content.lower().split())
            
            if not query_words or not content_words:
                return 0.0
            
            # Coefficient de Jaccard
            intersection = len(query_words & content_words)
            union = len(query_words | content_words)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Erreur calcul score sémantique: {e}")
            return 0.0

    def _update_metrics(self, results: List[SearchResultBase]) -> None:
        """Met à jour les métriques de recherche."""
        metrics.increment_counter("rag_searches")
        metrics.increment_counter("rag_results", len(results))
        
        scores = [result.score for result in results]
        if scores:
            metrics.record_value("rag_average_score", sum(scores) / len(scores))
            metrics.record_value("rag_max_score", max(scores))