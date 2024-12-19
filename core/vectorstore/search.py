# core/vectorstore/search.py
from elasticsearch import AsyncElasticsearch, helpers
from elasticsearch.exceptions import NotFoundError, RequestError
from typing import List, Dict, Optional, Any, Union
import numpy as np
from datetime import datetime
import asyncio
from functools import lru_cache

from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger("elasticsearch")

class ElasticsearchClient:
    def __init__(self):
        """Initialise le client Elasticsearch."""
        try:
            self.es = AsyncElasticsearch(
                hosts=[settings.ELASTICSEARCH_HOST],
                basic_auth=(settings.ELASTICSEARCH_USER, settings.ELASTICSEARCH_PASSWORD),
                verify_certs=settings.ELASTICSEARCH_VERIFY_CERTS,
                ca_certs=settings.ELASTICSEARCH_CA_CERTS,
                retry_on_timeout=True,
                max_retries=3,
                timeout=30
            )
            
            self.index_name = f"{settings.ELASTICSEARCH_INDEX_PREFIX}_vectors"
            self.embedding_dim = settings.ELASTICSEARCH_EMBEDDING_DIM
            
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
                "number_of_replicas": 1,
                "index": {
                    "similarity": {
                        "default": {
                            "type": "scripted_similarity",
                            "script": {
                                "source": "1.0 / (1.0 + l2norm(params.queryVector, doc['vector']))"
                            }
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "suggest": {"type": "completion"}
                        }
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

    async def index_vectors(
        self,
        vectors: List[List[float]],
        contents: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Indexe des vecteurs avec leur contenu et métadonnées.
        
        Args:
            vectors: Liste des vecteurs
            contents: Liste des contenus associés
            metadata: Liste optionnelle des métadonnées
            
        Returns:
            Liste des IDs des documents indexés
        """
        try:
            if len(vectors) != len(contents):
                raise ValueError("Nombre de vecteurs et contenus différent")

            metadata = metadata or [{}] * len(vectors)
            actions = []
            doc_ids = []

            for vector, content, meta in zip(vectors, contents, metadata):
                doc_id = meta.get("id") or str(uuid.uuid4())
                doc_ids.append(doc_id)
                
                action = {
                    "_index": self.index_name,
                    "_id": doc_id,
                    "_source": {
                        "vector": vector,
                        "content": content,
                        "metadata": meta,
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "active"
                    }
                }
                actions.append(action)

            with metrics.timer("es_bulk_index"):
                success, failed = await helpers.async_bulk(
                    self.es,
                    actions,
                    chunk_size=settings.ES_BULK_SIZE,
                    raise_on_error=False
                )
                
                logger.info(f"Indexation: {success} succès, {failed} échecs")
                metrics.increment_counter("vectors_indexed", success)
                if failed:
                    metrics.increment_counter("vectors_index_failed", failed)

            return doc_ids

        except Exception as e:
            logger.error(f"Erreur indexation vecteurs: {e}")
            metrics.increment_counter("vector_indexing_errors")
            raise

    async def search_similar(
        self,
        query_vector: List[float],
        k: int = 5,
        min_score: float = 0.7,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Recherche les vecteurs similaires.
        
        Args:
            query_vector: Vecteur de requête
            k: Nombre de résultats
            min_score: Score minimum de similarité
            filters: Filtres additionnels
            
        Returns:
            Liste des documents similaires trouvés
        """
        try:
            with metrics.timer("es_vector_search"):
                query = {
                    "script_score": {
                        "query": {"bool": {"must": [{"term": {"status": "active"}}]}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {"query_vector": query_vector}
                        }
                    }
                }

                # Ajout des filtres
                if filters:
                    for field, value in filters.items():
                        query["script_score"]["query"]["bool"]["must"].append(
                            {"term": {field: value}}
                        )

                response = await self.es.search(
                    index=self.index_name,
                    body={
                        "query": query,
                        "min_score": min_score,
                        "_source": {"excludes": ["vector"]},
                        "size": k
                    }
                )

                results = []
                for hit in response["hits"]["hits"]:
                    result = {
                        "id": hit["_id"],
                        "content": hit["_source"]["content"],
                        "score": hit["_score"] - 1.0,  # Normalisation du score
                        "metadata": hit["_source"]["metadata"]
                    }
                    results.append(result)

                metrics.increment_counter("vector_searches")
                return results

        except Exception as e:
            logger.error(f"Erreur recherche similaires: {e}")
            metrics.increment_counter("vector_search_errors")
            return []

    @lru_cache(maxsize=1000)
    async def get_document_vector(self, doc_id: str) -> Optional[List[float]]:
        """Récupère le vecteur d'un document."""
        try:
            doc = await self.es.get(
                index=self.index_name,
                id=doc_id,
                _source=["vector"]
            )
            return doc["_source"]["vector"]
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"Erreur récupération vecteur: {e}")
            return None

    async def delete_vectors(self, ids: List[str]) -> Dict[str, int]:
        """Supprime des vecteurs par leurs IDs."""
        try:
            results = await self.es.delete_by_query(
                index=self.index_name,
                body={"query": {"terms": {"_id": ids}}}
            )
            
            return {
                "deleted": results["deleted"],
                "failed": results["failed"]
            }
        except Exception as e:
            logger.error(f"Erreur suppression vecteurs: {e}")
            return {"deleted": 0, "failed": len(ids)}

    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            await self.es.close()
            logger.info("Client Elasticsearch fermé")
        except:
            pass

    async def health_check(self) -> bool:
        """Vérifie l'état de santé."""
        try:
            return await self.es.ping()
        except Exception as e:
            logger.error(f"Erreur health check: {e}")
            return False
