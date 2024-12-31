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
        await self._setup_templates()
        await self._ensure_indices_exist()

    async def _setup_templates(self):
        """Configure les templates des indices."""
        template = {
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
                    "number_of_shards": 2,
                    "number_of_replicas": 1,
                    "refresh_interval": "1s"
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
            },
            "index_patterns": [f"{self.index_prefix}_*"]
        }

        await self.es.indices.put_index_template(
            name=f"{self.index_prefix}_template",
            body=template
        )

    async def _ensure_indices_exist(self):
        """Crée les indices s'ils n'existent pas."""
        indices = [
            f"{self.index_prefix}_documents",
            f"{self.index_prefix}_vectors"
        ]
        
        for index in indices:
            if not await self.es.indices.exists(index=index):
                await self.es.indices.create(index=index)

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

    async def update_document(
        self,
        doc_id: str,
        update_fields: Dict,
        upsert: bool = False
    ) -> bool:
        """Met à jour un document existant."""
        try:
            body = {"doc": update_fields}
            if upsert:
                body["doc_as_upsert"] = True

            await self.es.update(
                index=f"{self.index_prefix}_documents",
                id=doc_id,
                body=body,
                refresh=True
            )
            return True
            
        except Exception as e:
            logger.error(f"Erreur mise à jour document {doc_id}: {e}")
            return False

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
            logger.error(f"Erreur suppression document {doc_id}: {e}")
            return False

    async def refresh_indices(self):
        """Force le rafraîchissement des indices."""
        try:
            await self.es.indices.refresh(index=f"{self.index_prefix}_*")
        except Exception as e:
            logger.error(f"Erreur rafraîchissement indices: {e}")
            
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
