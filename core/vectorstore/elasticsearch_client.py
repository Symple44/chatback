# core/vectorstore/elasticsearch_client.py
from typing import Dict, List, Optional, Any, Union, Tuple
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk, BulkIndexError
from datetime import datetime
import json
import logging
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .indexation import ElasticsearchIndexManager

logger = get_logger(__name__)

class ElasticsearchClient:
    def __init__(self):
        """Initialise le client Elasticsearch."""
        self.es = self._initialize_client()
        self.index_prefix = settings.ELASTICSEARCH_INDEX_PREFIX
        self.embedding_dim = settings.ELASTICSEARCH_EMBEDDING_DIM
        self.initialized = False
        self.index_manager = None

    def _initialize_client(self) -> AsyncElasticsearch:
        """Initialise le client avec SSL."""
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
        """Initialize le client et configure les indices."""
        if not self.initialized:
            try:
                if not await self.check_connection():
                    raise ConnectionError("Impossible de se connecter à Elasticsearch")
                
                # Initialiser le gestionnaire d'index
                self.index_manager = ElasticsearchIndexManager(
                    es_client=self.es,
                    index_prefix=self.index_prefix
                )
                
                # Configuration des indices
                await self.index_manager.setup_indices()
                
                self.initialized = True
                logger.info("Elasticsearch initialisé avec succès")

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

    async def index_document(
        self,
        title: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None,
        refresh: bool = True
    ) -> bool:
        """
        Indexe un document dans Elasticsearch.
        
        Args:
            title: Titre du document
            content: Contenu du document
            embedding: Vecteur d'embedding (optionnel)
            metadata: Métadonnées additionnelles
            doc_id: ID du document (optionnel)
            refresh: Rafraîchir l'index immédiatement
            
        Returns:
            bool: Succès de l'indexation
        """
        try:
            if not self.initialized:
                await self.initialize()

            # Préparation du document
            doc = {
                "title": title,
                "content": content,
                "metadata": metadata or {},
                "processed_date": datetime.utcnow().isoformat()
            }

            # Ajout de l'embedding s'il est fourni
            if embedding is not None:
                if len(embedding) != self.embedding_dim:
                    raise ValueError(f"Dimension d'embedding incorrecte: {len(embedding)}, attendu: {self.embedding_dim}")
                doc["embedding"] = embedding

            # Indexation avec retry
            index_name = f"{self.index_prefix}_documents"
            for attempt in range(3):
                try:
                    response = await self.es.index(
                        index=index_name,
                        document=doc,
                        id=doc_id,
                        refresh=refresh
                    )
                    
                    success = response.get("result") in ["created", "updated"]
                    if success:
                        metrics.increment_counter("documents_indexed")
                        logger.info(f"Document indexé avec succès: {title}")
                    return success

                except Exception as e:
                    if attempt == 2:  # Dernière tentative
                        raise
                    logger.warning(f"Tentative {attempt + 1} échouée: {e}")
                    await asyncio.sleep(1 * (2 ** attempt))  # Backoff exponentiel

        except Exception as e:
            logger.error(f"Erreur indexation document {title}: {e}")
            metrics.increment_counter("indexation_errors")
            return False

    async def search_documents(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        metadata_filter: Optional[Dict] = None,
        size: int = 5,
        min_score: float = 0.1
    ) -> List[Dict]:
        """
        Recherche des documents.
        
        Args:
            query: Requête textuelle
            vector: Vecteur de recherche optionnel
            metadata_filter: Filtres sur les métadonnées
            size: Nombre maximum de résultats
            min_score: Score minimum pour les résultats
        """
        try:
            if not self.initialized:
                await self.initialize()

            # Construction de la requête plus robuste
            query_body = {
                "size": size,
                "min_score": min_score,
                "query": {
                    "bool": {
                        "must": [],
                        "should": [],
                        "filter": [],
                        "minimum_should_match": 1
                    }
                },
                "_source": {
                    "excludes": ["embedding"]  # Optimisation: ne pas retourner les vecteurs
                }
            }

            # Recherche textuelle avec meilleure analyse
            if query:
                query_body["query"]["bool"]["must"].append({
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "content"],
                        "type": "most_fields",
                        "operator": "and",
                        "fuzziness": "AUTO",
                        "prefix_length": 2
                    }
                })

            # Filtres de métadonnées plus robustes
            if metadata_filter:
                for key, value in metadata_filter.items():
                    if value is not None:  # Ignore les valeurs None
                        filter_clause = {
                            "term": {
                                f"metadata.{key}.keyword": value
                            }
                        }
                        query_body["query"]["bool"]["filter"].append(filter_clause)

            # Recherche vectorielle améliorée
            if vector is not None:
                if len(vector) != self.embedding_dim:
                    logger.warning(f"Dimension du vecteur incorrecte: {len(vector)}")
                else:
                    query_body["query"]["bool"]["should"].append({
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": vector}
                            }
                        }
                    })

            # Exécution de la recherche avec retry
            for attempt in range(3):
                try:
                    response = await self.es.search(
                        index=f"{self.index_prefix}_documents",
                        body=query_body,
                        request_timeout=30
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(1 * (2 ** attempt))

            hits = response["hits"]["hits"]
            results = []
            
            for hit in hits:
                doc = {
                    "title": hit["_source"]["title"],
                    "content": hit["_source"]["content"],
                    "score": hit["_score"],
                    "metadata": hit["_source"].get("metadata", {}),
                    "application": hit["_source"].get("metadata", {}).get("application", "unknown")
                }
                results.append(doc)

            return results


        except Exception as e:
            logger.error(f"Erreur recherche documents: {e}", exc_info=True)
            if "details" in vars(e):
                logger.error(f"Détails de l'erreur ES: {e.details}")
            metrics.increment_counter("search_errors")
            return []

    async def delete_document(self, doc_id: str) -> bool:
        """Supprime un document par son ID."""
        try:
            response = await self.es.delete(
                index=f"{self.index_prefix}_documents",
                id=doc_id,
                refresh=True
            )
            return response["result"] == "deleted"
        except Exception as e:
            logger.error(f"Erreur suppression document: {e}")
            return False

    async def update_document(self, doc_id: str, update_fields: Dict) -> bool:
        """Met à jour un document existant."""
        try:
            response = await self.es.update(
                index=f"{self.index_prefix}_documents",
                id=doc_id,
                body={"doc": update_fields},
                refresh=True
            )
            return response["result"] == "updated"
        except Exception as e:
            logger.error(f"Erreur mise à jour document: {e}")
            return False

    async def bulk_index_documents(self, documents: List[Dict]) -> Tuple[int, List[Dict]]:
        """Indexe plusieurs documents en masse."""
        return await self.index_manager.bulk_index_documents(documents)

    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            if self.index_manager:
                await self.index_manager.cleanup()
            if self.es:
                await self.es.close()
            self.initialized = False
            logger.info("Ressources Elasticsearch nettoyées")
        except Exception as e:
            logger.error(f"Erreur nettoyage ressources ES: {e}")
