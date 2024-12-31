# core/vectorstore/index.py
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from elasticsearch.helpers import BulkIndexError
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger(__name__)

class IndexManager:
    def __init__(self, es_client):
        self.es = es_client
        self.index_prefix = settings.ELASTICSEARCH_INDEX_PREFIX
        self.embedding_dim = settings.ELASTICSEARCH_EMBEDDING_DIM

    async def setup_indices(self):
        """Configure les indices nécessaires."""
        try:
            # Supprime les anciens templates si nécessaire
            template_name = f"{self.index_prefix}_template"
            if await self.es.indices.exists_index_template(name=template_name):
                await self.es.indices.delete_index_template(name=template_name)

            await self._setup_templates()
            await self._ensure_indices_exist()
            logger.info("Configuration des indices terminée avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la configuration des indices: {e}")
            raise

    async def _setup_templates(self):
        """Configure les templates des indices."""
        template = {
            "index_patterns": [f"{self.index_prefix}_*"],
            "priority": 1,  # Priorité plus élevée pour éviter les conflits
            "template": {
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "content_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase",
                                    "stop",
                                    "snowball"
                                ]
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
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        "content": {
                            "type": "text",
                            "analyzer": "content_analyzer"
                        },
                        "metadata": {
                            "type": "object",
                            "dynamic": True
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
            }
        }

        try:
            # Configuration du template
            template_name = f"{self.index_prefix}_template"
            await self.es.indices.put_index_template(
                name=template_name,
                body=template
            )
            logger.info(f"Template {template_name} créé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la création du template: {e}")
            raise

    async def _ensure_indices_exist(self):
        """Crée les indices s'ils n'existent pas."""
        indices = [
            f"{self.index_prefix}_documents",
            f"{self.index_prefix}_vectors"
        ]
        
        for index in indices:
            if not await self.es.indices.exists(index=index):
                await self.es.indices.create(index=index)
                logger.info(f"Index {index} créé")

    async def index_document(
        self,
        title: str,
        content: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None,
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

            if vector and len(vector) == self.embedding_dim:
                doc["embedding"] = vector

            index_params = {
                "index": f"{self.index_prefix}_documents",
                "body": doc,
                "refresh": refresh
            }
            
            if doc_id:
                index_params["id"] = doc_id

            await self.es.index(**index_params)
            metrics.increment_counter("documents_indexed")
            return True

        except Exception as e:
            logger.error(f"Erreur indexation document: {e}")
            metrics.increment_counter("indexing_errors")
            return False

    async def bulk_index(
        self,
        documents: List[Dict],
        chunk_size: int = 500
    ) -> Dict[str, int]:
        """Indexation par lots avec rapport d'erreurs."""
        stats = {"success": 0, "errors": 0}
        
        try:
            actions = []
            for doc in documents:
                action = {
                    "_index": f"{self.index_prefix}_documents",
                    "_source": {
                        "title": doc["title"],
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {}),
                        "embedding": doc.get("vector"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                if "id" in doc:
                    action["_id"] = doc["id"]
                actions.append(action)

            for i in range(0, len(actions), chunk_size):
                chunk = actions[i:i + chunk_size]
                try:
                    response = await self.es.bulk(operations=chunk)
                    
                    if response.get("errors", False):
                        for item in response["items"]:
                            if "error" in item.get("index", {}):
                                stats["errors"] += 1
                            else:
                                stats["success"] += 1
                    else:
                        stats["success"] += len(chunk)
                        
                except BulkIndexError as e:
                    stats["errors"] += len(chunk)
                    logger.error(f"Erreur bulk chunk {i}: {str(e)}")

            return stats

        except Exception as e:
            logger.error(f"Erreur indexation bulk: {e}")
            return stats

    async def get_index_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques des indices."""
        try:
            stats = await self.es.indices.stats(index=f"{self.index_prefix}_*")
            return {
                "doc_count": stats["_all"]["primaries"]["docs"]["count"],
                "store_size": stats["_all"]["primaries"]["store"]["size_in_bytes"],
                "indices": stats["indices"]
            }
        except Exception as e:
            logger.error(f"Erreur récupération stats indices: {e}")
            return {}

    async def cleanup_old_documents(self, days: int = 30) -> int:
        """Nettoie les anciens documents."""
        try:
            response = await self.es.delete_by_query(
                index=f"{self.index_prefix}_*",
                body={
                    "query": {
                        "range": {
                            "timestamp": {
                                "lte": f"now-{days}d"
                            }
                        }
                    }
                }
            )
            return response.get("deleted", 0)
        except Exception as e:
            logger.error(f"Erreur nettoyage documents: {e}")
            return 0
