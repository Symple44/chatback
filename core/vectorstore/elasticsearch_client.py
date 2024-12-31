from typing import Dict, List, Optional, Any, Union
from elasticsearch import AsyncElasticsearch
from datetime import datetime
import logging
import json
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger(__name__)

class ElasticsearchClient:
    def __init__(self):
        """Initialise le client Elasticsearch."""
        self.es = self._initialize_client()
        self.index_prefix = settings.ELASTICSEARCH_INDEX_PREFIX
        self.embedding_dim = settings.ELASTICSEARCH_EMBEDDING_DIM
        self.initialized = False

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

                # Configuration des indices
                await self.setup_indices()
                
                # Vérification du mapping
                await self.verify_mapping()
                
                self.initialized = True
                logger.info("Elasticsearch initialisé avec succès")

            except Exception as e:
                logger.error(f"Erreur initialisation ES: {e}")
                raise

    async def setup_indices(self):
        """Configure les indices nécessaires."""
        try:
            # Configuration du template pour les documents
            template = {
                "index_patterns": [f"{self.index_prefix}_*"],
                "priority": 100,
                "template": {
                    "settings": {
                        "analysis": {
                            "analyzer": {
                                "content_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "stop", "snowball"]
                                }
                            }
                        },
                        "number_of_shards": settings.ELASTICSEARCH_NUMBER_OF_SHARDS,
                        "number_of_replicas": settings.ELASTICSEARCH_NUMBER_OF_REPLICAS,
                        "refresh_interval": settings.ELASTICSEARCH_REFRESH_INTERVAL
                    },
                    "mappings": {
                        "properties": {
                            "title": {
                                "type": "text",
                                "analyzer": "content_analyzer",
                                "fields": {
                                    "keyword": {"type": "keyword"},
                                    "suggest": {
                                        "type": "completion",
                                        "analyzer": "content_analyzer"
                                    }
                                }
                            },
                            "content": {
                                "type": "text",
                                "analyzer": "content_analyzer"
                            },
                            "embedding": {
                                "type": "dense_vector",
                                "dims": settings.ELASTICSEARCH_EMBEDDING_DIM,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "metadata": {
                                "type": "object",
                                "dynamic": True
                            }
                        }
                    }
                }
            }

            # Création du template
            template_name = f"{self.index_prefix}_template"
            try:
                await self.es.indices.delete_index_template(name=template_name)
            except Exception:
                pass
            await self.es.indices.put_index_template(
                name=template_name,
                body=template
            )

            # Création des indices si nécessaire
            indices = [
                f"{self.index_prefix}_documents",
                f"{self.index_prefix}_vectors"
            ]

            for index in indices:
                if not await self.es.indices.exists(index=index):
                    await self.es.indices.create(index=index)
                    logger.info(f"Index {index} créé")

        except Exception as e:
            logger.error(f"Erreur configuration indices: {e}")
            raise

    async def check_connection(self) -> bool:
        """Vérifie la connexion à Elasticsearch."""
        try:
            return await self.es.ping()
        except Exception as e:
            logger.error(f"Erreur connexion ES: {e}")
            return False

    async def verify_mapping(self):
        """Vérifie et corrige le mapping si nécessaire."""
        try:
            index = f"{self.index_prefix}_documents"
            mapping = await self.es.indices.get_mapping(index=index)
            
            # Vérifie si le champ embedding existe avec les bonnes propriétés
            if not mapping.get(index, {}).get("mappings", {}).get("properties", {}).get("embedding"):
                logger.warning("Mapping incorrect - recréation de l'index")
                
                # Sauvegarde des données
                docs = await self._backup_documents(index)
                
                # Recréation de l'index
                await self.es.indices.delete(index=index)
                await self.setup_indices()
                
                # Restauration des données
                if docs:
                    await self._restore_documents(index, docs)

        except Exception as e:
            logger.error(f"Erreur vérification mapping: {e}")
            raise

    async def _backup_documents(self, index: str) -> List[Dict]:
        """Sauvegarde les documents d'un index."""
        try:
            docs = []
            async for hit in self.es.scan(
                index=index,
                query={"match_all": {}}
            ):
                docs.append(hit["_source"])
            return docs
        except Exception as e:
            logger.error(f"Erreur sauvegarde documents: {e}")
            return []

    async def _restore_documents(self, index: str, documents: List[Dict]):
        """Restaure les documents dans un index."""
        try:
            for doc in documents:
                await self.es.index(
                    index=index,
                    document=doc,
                    refresh=True
                )
            logger.info(f"{len(documents)} documents restaurés")
        except Exception as e:
            logger.error(f"Erreur restauration documents: {e}")
            raise

    async def index_document(
        self,
        title: str,
        content: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
        refresh: bool = True
    ) -> bool:
        """Indexe un document."""
        try:
            doc = {
                "title": title,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }

            if vector is not None and len(vector) == self.embedding_dim:
                doc["embedding"] = vector

            await self.es.index(
                index=f"{self.index_prefix}_documents",
                document=doc,
                refresh=refresh
            )
            
            metrics.increment_counter("documents_indexed")
            return True
            
        except Exception as e:
            logger.error(f"Erreur indexation document: {e}")
            metrics.increment_counter("indexing_errors")
            return False

    async def search_documents(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        metadata_filter: Optional[Dict] = None,
        size: int = 5
    ) -> List[Dict]:
        """Recherche des documents avec gestion robuste des erreurs."""
        try:
            if not self.initialized:
                await self.initialize()
    
            # Construction de la requête de base
            query_body = {
                "size": size,
                "query": {"match_all": {}}
            }
    
            # Ajout de la recherche textuelle si présente
            if query:
                query_body["query"] = {
                    "match": {
                        "content": {
                            "query": query,
                            "operator": "or"
                        }
                    }
                }
    
            # Ajout du filtre de métadonnées
            if metadata_filter:
                query_body["query"] = {
                    "bool": {
                        "must": [
                            query_body["query"],
                            *[
                                {"term": {f"metadata.{key}.keyword": value}} 
                                for key, value in metadata_filter.items()
                            ]
                        ]
                    }
                }
    
            # Utilisation de l'index correct
            index = f"{self.index_prefix}_documents"
    
            try:
                response = await self.es.search(
                    index=index,
                    body=query_body
                )
            except Exception as search_error:
                logger.error(f"Erreur recherche Elasticsearch: {search_error}")
                return []
    
            # Transformation des résultats
            return [
                {
                    "title": hit.get("_source", {}).get("title", ""),
                    "content": hit.get("_source", {}).get("content", ""),
                    "score": hit.get("_score", 0.0),
                    "metadata": hit.get("_source", {}).get("metadata", {})
                }
                for hit in response.get("hits", {}).get("hits", [])
            ]
    
        except Exception as e:
            logger.error(f"Erreur fatale recherche documents: {e}", exc_info=True)
            return []

    async def delete_document(self, doc_id: str) -> bool:
        """Supprime un document."""
        try:
            await self.es.delete(
                index=f"{self.index_prefix}_documents",
                id=doc_id,
                refresh=True
            )
            return True
        except Exception as e:
            logger.error(f"Erreur suppression document: {e}")
            return False

    async def update_document(self, doc_id: str, update_fields: Dict) -> bool:
        """Met à jour un document."""
        try:
            await self.es.update(
                index=f"{self.index_prefix}_documents",
                id=doc_id,
                body={"doc": update_fields},
                refresh=True
            )
            return True
        except Exception as e:
            logger.error(f"Erreur mise à jour document: {e}")
            return False

    async def get_document(self, doc_id: str) -> Optional[Dict]:
        """Récupère un document par son ID."""
        try:
            response = await self.es.get(
                index=f"{self.index_prefix}_documents",
                id=doc_id
            )
            return response["_source"]
        except Exception as e:
            logger.error(f"Erreur récupération document: {e}")
            return None

    async def cleanup(self):
        """Nettoie les ressources."""
        if self.es:
            await self.es.close()
            self.initialized = False
            logger.info("Ressources Elasticsearch nettoyées")
