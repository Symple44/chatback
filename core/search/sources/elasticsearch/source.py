from typing import List, Dict, Any, Optional
from datetime import datetime

from ...interfaces import (
    SearchContext,
    SearchResultBase,
    SearchResultMetadata,
    SearchSourceCapabilities,
    SourceType,
    ContentType
)
from ..base import BaseDataSource
from .client import ElasticsearchClient
from ...exceptions import (
    SourceConnectionError,
    SearchConfigurationError
)

class ElasticsearchSource(BaseDataSource):
    """Source de données Elasticsearch."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client: Optional[ElasticsearchClient] = None
        self._capabilities = [
            SearchSourceCapabilities.FULL_TEXT_SEARCH,
            SearchSourceCapabilities.VECTOR_SEARCH,
            SearchSourceCapabilities.METADATA_SEARCH,
            SearchSourceCapabilities.FILTERS,
            SearchSourceCapabilities.AGGREGATIONS
        ]

    def get_source_type(self) -> SourceType:
        return SourceType.ELASTICSEARCH

    def validate_config(self) -> None:
        """Valide la configuration Elasticsearch."""
        required_fields = ["host", "index"]
        missing = [f for f in required_fields if f not in self.config]
        if missing:
            raise SearchConfigurationError(
                f"Configuration Elasticsearch invalide. Champs manquants: {', '.join(missing)}"
            )

    async def initialize(self) -> None:
        """Initialise le client Elasticsearch."""
        try:
            if not self._initialized:
                self.validate_config()
                self.client = ElasticsearchClient(
                    host=self.config["host"],
                    index=self.config["index"],
                    **{k: v for k, v in self.config.items() if k not in ["host", "index"]}
                )
                await self.client.initialize()
                self._initialized = True
        except Exception as e:
            raise SourceConnectionError(
                f"Erreur d'initialisation Elasticsearch: {str(e)}",
                source="elasticsearch"
            )

    async def search(self, context: SearchContext) -> List[SearchResultBase]:
        """Effectue une recherche dans Elasticsearch."""
        if not self._initialized:
            await self.initialize()

        try:
            # Adaptation des paramètres de recherche
            search_params = {
                "query": context.query,
                "size": context.max_results,
                "min_score": context.min_score
            }

            # Ajout des filtres si présents
            if context.filters:
                search_params["metadata_filter"] = context.filters

            # Recherche dans Elasticsearch
            results = await self.client.search_documents(**search_params)

            # Conversion des résultats au format standard
            return [
                SearchResultBase(
                    content=hit["_source"].get("content", ""),
                    score=float(hit["_score"]),
                    metadata=SearchResultMetadata(
                        title=hit["_source"].get("title"),
                        source="elasticsearch",
                        source_type=SourceType.ELASTICSEARCH,
                        content_type=ContentType.TEXT,
                        language=context.language,
                        timestamp=datetime.utcnow(),
                        extra=hit["_source"].get("metadata", {})
                    ),
                    source_id=hit["_id"],
                    vector=hit["_source"].get("embedding")
                )
                for hit in results["hits"]["hits"]
            ]

        except Exception as e:
            raise SourceConnectionError(
                f"Erreur de recherche Elasticsearch: {str(e)}",
                source="elasticsearch"
            )

    async def get_by_id(self, id: str) -> Optional[SearchResultBase]:
        """Récupère un document par son ID."""
        if not self._initialized:
            await self.initialize()

        try:
            doc = await self.client.get_document(id)
            if not doc:
                return None

            return SearchResultBase(
                content=doc.get("content", ""),
                score=1.0,  # Score par défaut pour get_by_id
                metadata=SearchResultMetadata(
                    title=doc.get("title"),
                    source="elasticsearch",
                    source_type=SourceType.ELASTICSEARCH,
                    content_type=ContentType.TEXT,
                    timestamp=datetime.utcnow(),
                    extra=doc.get("metadata", {})
                ),
                source_id=id,
                vector=doc.get("embedding")
            )

        except Exception as e:
            raise SourceConnectionError(
                f"Erreur de récupération du document {id}: {str(e)}",
                source="elasticsearch"
            )

    async def health_check(self) -> Dict[str, Any]:
        """Vérifie l'état de santé d'Elasticsearch."""
        if not self._initialized:
            return {"status": "not_initialized"}

        try:
            cluster_health = await self.client.health_check()
            return {
                "status": "healthy" if cluster_health["status"] in ["green", "yellow"] else "unhealthy",
                "details": cluster_health
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """Nettoie les ressources Elasticsearch."""
        if self.client:
            await self.client.cleanup()
        await super().cleanup()