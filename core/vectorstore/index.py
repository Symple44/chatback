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
        self.template_name = f"{self.index_prefix}_template"

    async def setup_indices(self):
        """Configure les indices nécessaires."""
        try:
            # Suppression des anciens templates si nécessaires
            if await self.es.indices.exists_index_template(name=self.template_name):
                await self.es.indices.delete_index_template(name=self.template_name)

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
            "priority": 1,
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
                    "refresh_interval": settings.ELASTICSEARCH_REFRESH_INTERVAL,
                    "index": {
                        "mapping": {
                            "total_fields": {
                                "limit": 2000
                            }
                        }
                    }
                },
                "mappings": {
                    "dynamic": False,
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
                            "analyzer": "content_analyzer",
                            "term_vector": "with_positions_offsets"
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.embedding_dim,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {
                            "type": "object",
                            "dynamic": True,
                            "properties": {
                                "application": {"type": "keyword"},
                                "source": {"type": "keyword"},
                                "page": {"type": "integer"},
                                "indexed_at": {"type": "date"}
                            }
                        },
                        "timestamp": {"type": "date"}
                    }
                }
            }
        }

        try:
            # Configuration du template
            await self.es.indices.put_index_template(
                name=self.template_name,
                body=template
            )
            logger.info(f"Template {self.template_name} créé avec succès")
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
            try:
                if not await self.es.indices.exists(index=index):
                    await self.es.indices.create(index=index)
                    logger.info(f"Index {index} créé")
            except Exception as e:
                logger.error(f"Erreur création index {index}: {e}")
                raise

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

            if vector is not None and len(vector) == self.embedding_dim:
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
            logger.debug(f"Document indexé: {title}")
            return True

        except Exception as e:
            logger.error(f"Erreur indexation document: {e}")
            metrics.increment_counter("indexing_errors")
            return False

    async def verify_index_mapping(self) -> bool:
        """Vérifie et corrige le mapping de l'index si nécessaire."""
        try:
            index_name = f"{self.index_prefix}_documents"
            mapping = await self.es.indices.get_mapping(index=index_name)
            
            if not mapping.get(index_name, {}).get("mappings", {}).get("properties", {}).get("embedding"):
                logger.warning("Champ 'embedding' manquant dans le mapping - recréation de l'index")
                
                # Sauvegarde des données existantes
                docs = await self._backup_documents(index_name)
                
                # Suppression et recréation de l'index
                await self.es.indices.delete(index=index_name)
                await self._ensure_indices_exist()
                
                # Restauration des données
                if docs:
                    await self.bulk_index(docs)
                    
                return True
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur vérification mapping: {e}")
            return False

    async def _backup_documents(self, index_name: str) -> List[Dict]:
        """Sauvegarde les documents d'un index."""
        try:
            result = await self.es.search(
                index=index_name,
                body={"query": {"match_all": {}}},
                size=10000  # Ajustez selon vos besoins
            )
            return [doc["_source"] for doc in result["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Erreur sauvegarde documents: {e}")
            return []
