# core/vectorstore/search.py
from elasticsearch import Elasticsearch, AsyncElasticsearch
from typing import List, Dict, Optional, Any, Tuple
import logging
import ssl
import os
from datetime import datetime
from elasticsearch.helpers import bulk, BulkIndexError
from elasticsearch.exceptions import ConnectionError, AuthenticationException
import certifi

from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger(__name__)

class ElasticsearchClient:
    def __init__(self):
        """Initialise le client Elasticsearch."""
        try:
            # Configuration SSL
            ssl_context = {
                'verify_certs': True if settings.ELASTICSEARCH_CA_CERT else False,
                'ca_certs': settings.ELASTICSEARCH_CA_CERT or certifi.where(),
                'client_cert': settings.ELASTICSEARCH_CLIENT_CERT,
                'client_key': settings.ELASTICSEARCH_CLIENT_KEY
            }

            # Configuration client
            self.es = AsyncElasticsearch(
                hosts=[settings.ELASTICSEARCH_HOST],
                basic_auth=(settings.ELASTICSEARCH_USER, settings.ELASTICSEARCH_PASSWORD),
                **ssl_context,
                retry_on_timeout=True,
                max_retries=3,
                timeout=30,
                sniff_on_start=False,
                sniff_on_node_failure=False
            )

            self.index_name = f"{settings.ELASTICSEARCH_INDEX_PREFIX}_documents"
            self.vector_index = f"{settings.ELASTICSEARCH_INDEX_PREFIX}_vectors"
            self.embedding_dim = settings.ELASTICSEARCH_EMBEDDING_DIM

            logger.info("Client Elasticsearch initialisé")

        except Exception as e:
            logger.error(f"Erreur initialisation Elasticsearch: {e}")
            raise

    async def initialize(self):
        """Configure les indices et templates."""
        try:
            if not await self.es.ping():
                raise ConnectionError("Elasticsearch connection failed")
            await self._verify_ssl_certs()
            await self._setup_templates()
            await self._setup_indices()
            logger.info("Indices et templates configurés")
        except Exception as e:
            logger.error(f"Erreur initialisation: {e}")
            raise

    async def _verify_ssl_certs(self):
        """Vérifie les certificats SSL."""
        if settings.ELASTICSEARCH_CA_CERT:
            if not os.path.exists(settings.ELASTICSEARCH_CA_CERT):
                raise FileNotFoundError(f"CA cert non trouvé: {settings.ELASTICSEARCH_CA_CERT}")
            if settings.ELASTICSEARCH_CLIENT_CERT and not os.path.exists(settings.ELASTICSEARCH_CLIENT_CERT):
                raise FileNotFoundError(f"Client cert non trouvé: {settings.ELASTICSEARCH_CLIENT_CERT}")

    async def _setup_templates(self):
        document_template = {
            "template": {
                "settings": {
                    "number_of_shards": 2,
                    "number_of_replicas": 1,
                    "refresh_interval": "1s",
                    "analysis": {
                        "analyzer": {
                            "content_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop", "snowball"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "title": {
                            "type": "text",
                            "analyzer": "content_analyzer",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        "content": {
                            "type": "text",
                            "analyzer": "content_analyzer"
                        },
                        "application": {
                            "type": "keyword"
                        },
                        "file_path": {
                            "type": "keyword"
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "page_count": {"type": "integer"},
                                "indexed_at": {"type": "date"},
                                "sync_date": {"type": "date"}
                            }
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.embedding_dim,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "timestamp": {"type": "date"}
                    }
                }
            },
            "index_patterns": [f"{settings.ELASTICSEARCH_INDEX_PREFIX}_documents*"]
        }
        await self.es.indices.put_index_template(
            name=f"{settings.ELASTICSEARCH_INDEX_PREFIX}_documents_template",
            body=document_template
        )

    async def _setup_indices(self):
        """Crée les indices s'ils n'existent pas."""
        for index in [self.index_name, self.vector_index]:
            if not await self.es.indices.exists(index=index):
                await self.es.indices.create(index=index)

    async def index_document(
        self,
        title: str,
        content: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Indexe un document."""
        try:
            doc = {
                "title": title,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }

            if vector:
                doc["embedding"] = vector

            response = await self.es.index(
                index=self.index_name,
                document=doc,
                refresh=True
            )

            metrics.increment_counter("documents_indexed")
            return True

        except Exception as e:
            logger.error(f"Erreur indexation document: {e}")
            metrics.increment_counter("indexing_errors")
            return False

    async def bulk_index_documents(
        self,
        documents: List[Dict],
        chunk_size: int = 500
    ) -> Tuple[int, int]:
        """Indexation en masse avec support des vecteurs."""
        success = errors = 0
        try:
            actions = [
                {
                    "_index": self.index_name,
                    "_source": {
                        "title": doc["title"],
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {}),
                        "embedding": doc.get("vector"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                for doc in documents
            ]

            for i in range(0, len(actions), chunk_size):
                chunk = actions[i:i + chunk_size]
                try:
                    response = await self.es.bulk(operations=chunk)
                    if response["errors"]:
                        errors += sum(1 for item in response["items"] if item["index"].get("error"))
                        success += len(chunk) - errors
                    else:
                        success += len(chunk)
                except BulkIndexError as e:
                    errors += len(chunk)
                    logger.error(f"Erreur bulk indexation chunk {i}: {e}")

            metrics.increment_counter("documents_indexed", success)
            if errors:
                metrics.increment_counter("indexing_errors", errors)

            return success, errors

        except Exception as e:
            logger.error(f"Erreur indexation bulk: {e}")
            return success, len(documents) - success

    async def search_documents(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        application: Optional[str] = None,
        size: int = 5,
        min_score: float = 0.1
    ) -> List[Dict]:
        try:
            search_query = {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^2", "content"],
                                "type": "best_fields",
                                "operator": "or",
                                "minimum_should_match": "75%"
                            }
                        }
                    ],
                    "filter": []
                }
            }

            if application:
                search_query["bool"]["filter"].append({
                    "term": {"application": application}
                })

            if vector:
                search_query["bool"]["should"] = [{
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": vector}
                        }
                    }
                }]
                search_query["bool"]["minimum_should_match"] = 1

            response = await self.es.search(
                index=self.index_name,
                body={
                    "query": search_query,
                    "size": size,
                    "min_score": min_score,
                    "_source": ["title", "content", "application", "metadata"],
                    "highlight": {
                        "fields": {
                            "content": {
                                "fragment_size": 150,
                                "number_of_fragments": 3
                            }
                        },
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    }
                }
            )

            return [self._format_search_result(hit) for hit in response["hits"]["hits"]]

        except Exception as e:
            logger.error(f"Erreur recherche documents: {e}")
            return []

    def _format_search_result(self, hit: Dict) -> Dict:
        """Formate un résultat de recherche."""
        source = hit["_source"]
        return {
            "title": source["title"],
            "content": source["content"],
            "application": source["application"],
            "score": hit["_score"],
            "highlights": hit.get("highlight", {}).get("content", []),
            "metadata": source.get("metadata", {})
        }

    async def get_document(self, doc_id: str) -> Optional[Dict]:
        """Récupère un document par ID."""
        try:
            doc = await self.es.get(
                index=self.index_name,
                id=doc_id
            )
            return doc["_source"]
        except Exception as e:
            logger.error(f"Erreur récupération document {doc_id}: {e}")
            return None

    async def delete_document(self, doc_id: str) -> bool:
        """Supprime un document."""
        try:
            await self.es.delete(
                index=self.index_name,
                id=doc_id,
                refresh=True
            )
            return True
        except Exception as e:
            logger.error(f"Erreur suppression document {doc_id}: {e}")
            return False

    async def cleanup(self):
        """Nettoie les ressources."""
        if self.es:
            await self.es.close()
            logger.info("Elasticsearch fermé")

    async def health_check(self) -> Dict[str, Any]:
        """Vérifie l'état du cluster."""
        try:
            health = await self.es.cluster.health()
            return {
                "status": health["status"],
                "nodes": health["number_of_nodes"],
                "docs": (await self.es.count(index=self.index_name))["count"],
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Erreur health check: {e}")
            return {"status": "error", "error": str(e)}
