# core/vectorstore/client.py
from typing import Dict, List, Optional, Any
from elasticsearch import AsyncElasticsearch
from .search import SearchManager
from .index import IndexManager
from .query_builder import QueryBuilder
from .response_formatter import ResponseFormatter
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger(__name__)

class ElasticsearchClient:
    def __init__(self):
        self.es = self._initialize_client()
        self.search_manager = SearchManager(self.es)
        self.index_manager = IndexManager(self.es)
        self.query_builder = QueryBuilder()
        self.response_formatter = ResponseFormatter()
        self.initialized = False

    def _initialize_client(self) -> AsyncElasticsearch:
        """Initialise le client Elasticsearch avec SSL."""
        ssl_context = {
            'verify_certs': bool(settings.ELASTICSEARCH_CA_CERT),
            'ca_certs': settings.ELASTICSEARCH_CA_CERT,
            'client_cert': settings.ELASTICSEARCH_CLIENT_CERT,
            'client_key': settings.ELASTICSEARCH_CLIENT_KEY
        }

        return AsyncElasticsearch(
            hosts=[settings.ELASTICSEARCH_HOST],
            basic_auth=(settings.ELASTICSEARCH_USER, settings.ELASTICSEARCH_PASSWORD),
            **ssl_context,
            retry_on_timeout=True,
            max_retries=3,
            timeout=30,
            sniff_on_start=False,
            sniff_on_node_failure=False
        )

    async def initialize(self):
        """Initialise le client et configure les indices."""
        if not self.initialized:
            try:
                if not await self.check_connection():
                    raise ConnectionError("Impossible de se connecter à Elasticsearch")

                # Configuration des indices
                await self.index_manager.setup_indices()
                self.initialized = True
                logger.info("Client Elasticsearch initialisé")

            except Exception as e:
                logger.error(f"Erreur initialisation ES: {e}")
                raise

    async def check_connection(self) -> bool:
        """Vérifie la connexion à Elasticsearch."""
        try:
            return await self.es.ping()
        except Exception as e:
            logger.error(f"Erreur connexion ES: {e}")
            return False

    async def search_documents(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        metadata_filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Dict]:
        """Interface principale de recherche."""
        try:
            results = await self.search_manager.search_documents(
                query, vector, metadata_filter, **kwargs
            )
            metrics.increment_counter("successful_searches")
            return results

        except Exception as e:
            logger.error(f"Erreur recherche: {e}")
            metrics.increment_counter("failed_searches")
            return []

    async def index_document(self, **kwargs) -> bool:
        """Interface d'indexation."""
        return await self.index_manager.index_document(**kwargs)

    async def bulk_index(self, documents: List[Dict], **kwargs) -> Dict[str, int]:
        """Interface d'indexation par lots."""
        return await self.index_manager.bulk_index(documents, **kwargs)

    async def get_cluster_health(self) -> Dict[str, Any]:
        """Récupère l'état de santé du cluster."""
        try:
            health = await self.es.cluster.health()
            metrics = await self.es.cluster.stats()
            
            return {
                "status": health["status"],
                "nodes": health["number_of_nodes"],
                "shards": {
                    "total": health["active_shards"],
                    "relocating": health["relocating_shards"],
                    "initializing": health["initializing_shards"],
                    "unassigned": health["unassigned_shards"]
                },
                "memory": {
                    "total": metrics["nodes"]["os"]["mem"]["total_in_bytes"],
                    "free": metrics["nodes"]["os"]["mem"]["free_in_bytes"]
                },
                "indices": {
                    "count": metrics["indices"]["count"],
                    "docs": metrics["indices"]["docs"]["count"],
                    "size": metrics["indices"]["store"]["size_in_bytes"]
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur récupération santé cluster: {e}")
            return {"status": "error", "error": str(e)}

    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            await self.es.close()
            logger.info("Ressources Elasticsearch nettoyées")
        except Exception as e:
            logger.error(f"Erreur nettoyage ES: {e}")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
