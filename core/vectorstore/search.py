# core/vectorstore/search.py
from elasticsearch import AsyncElasticsearch
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
from core.config import settings

logger = logging.getLogger(__name__)

class ElasticsearchClient:
    def __init__(self):
        """Initialise le client Elasticsearch."""
        try:
            self.es = AsyncElasticsearch(
                hosts=[settings.ELASTICSEARCH_HOST],
                verify_certs=False,
                basic_auth=(settings.ELASTICSEARCH_USER, settings.ELASTICSEARCH_PASSWORD) if settings.ELASTICSEARCH_USER else None
            )
            
            self.index_name = "chat_vectors"
            self.embedding_dim = 384  # Dimension des embeddings
            logger.info("Client Elasticsearch initialisé")
            
        except Exception as e:
            logger.error(f"Erreur initialisation Elasticsearch: {e}")
            raise

    async def initialize(self):
        """Configure l'index et les mappings."""
        try:
            if not await self.es.indices.exists(index=self.index_name):
                await self._create_index()
            logger.info("Index Elasticsearch initialisé")
        except Exception as e:
            logger.error(f"Erreur initialisation index: {e}")
            raise

    async def _create_index(self):
        """Crée l'index avec le mapping approprié."""
        mapping = {
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 1
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": True
                    },
                    "vector": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "timestamp": {"type": "date"},
                    "source": {"type": "keyword"},
                    "doc_type": {"type": "keyword"},
                    "status": {"type": "keyword"}
                }
            }
        }

        await self.es.indices.create(index=self.index_name, body=mapping)
        logger.info(f"Index {self.index_name} créé avec succès")

    async def index_document(self, title: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Indexe un document."""
        try:
            doc = {
                "title": title,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow(),
                "status": "active"
            }
            
            await self.es.index(index=self.index_name, document=doc)
            return True
            
        except Exception as e:
            logger.error(f"Erreur indexation document: {e}")
            return False

    async def search_documents(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        size: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """Recherche des documents."""
        try:
            # Construction de la requête
            search_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"content": query}},
                            {"term": {"status": "active"}}
                        ]
                    }
                },
                "size": size
            }

            # Ajout de la recherche vectorielle si un vecteur est fourni
            if vector:
                search_query["query"]["bool"]["should"] = [{
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {"query_vector": vector}
                        }
                    }
                }]

            # Exécution de la recherche
            response = await self.es.search(
                index=self.index_name,
                body=search_query
            )

            return [hit["_source"] for hit in response["hits"]["hits"]]

        except Exception as e:
            logger.error(f"Erreur recherche documents: {e}")
            return []

    async def cleanup(self):
        """Nettoie les ressources."""
        if hasattr(self, 'es'):
            await self.es.close()
            logger.info("Client Elasticsearch fermé")
