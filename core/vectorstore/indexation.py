# core/vectorstore/indexation.py
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import time
import json
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk, BulkIndexError
import logging
from pathlib import Path

from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("es_indexation")

class ElasticsearchIndexManager:
    def __init__(self, es_client: AsyncElasticsearch, index_prefix: str = None):
        """Initialise le gestionnaire d'index."""
        self.es = es_client
        self.index_prefix = index_prefix or settings.ELASTICSEARCH_INDEX_PREFIX
        self.embedding_dim = settings.ELASTICSEARCH_EMBEDDING_DIM

    async def setup_indices(self) -> None:
        """Configure tous les indices nécessaires."""
        indices_config = {
            "documents": self._get_documents_mapping(),
            "vectors": self._get_vectors_mapping(),
            "chunks": self._get_chunks_mapping()
        }
        
        for name, config in indices_config.items():
            index_name = f"{self.index_prefix}_{name}"
            try:
                if not await self.es.indices.exists(index=index_name):
                    await self.create_index(index_name, config)
                else:
                    # Vérifier et mettre à jour le mapping si nécessaire
                    await self.update_mapping_if_needed(index_name, config)
            except Exception as e:
                logger.error(f"Erreur configuration index {index_name}: {e}")
                raise

    def _get_documents_mapping(self) -> Dict:
        """Retourne le mapping pour l'index documents."""
        return {
            "settings": {
                "number_of_shards": settings.ELASTICSEARCH_NUMBER_OF_SHARDS,
                "number_of_replicas": settings.ELASTICSEARCH_NUMBER_OF_REPLICAS,
                "refresh_interval": settings.ELASTICSEARCH_REFRESH_INTERVAL,
                "analysis": {
                    "analyzer": {
                        "french_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "french_elision",
                                "french_stop",
                                "french_stemmer"
                            ]
                        }
                    },
                    "filter": {
                        "french_elision": {
                            "type": "elision",
                            "articles_case": True,
                            "articles": ["l", "m", "t", "qu", "n", "s", "j", "d", "c"]
                        },
                        "french_stop": {
                            "type": "stop",
                            "stopwords": "_french_"
                        },
                        "french_stemmer": {
                            "type": "stemmer",
                            "language": "light_french"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "french_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "suggest": {
                                "type": "completion",
                                "analyzer": "french_analyzer"
                            }
                        }
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "french_analyzer"
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "similarity": "cosine",
                        "index": True
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "keyword"},
                            "language": {"type": "keyword"},
                            "application": {"type": "keyword"},
                            "version": {"type": "keyword"},
                            "last_updated": {"type": "date"},
                            "tags": {"type": "keyword"},
                            "category": {"type": "keyword"}
                        },
                        "dynamic": True
                    },
                    "importance_score": {"type": "float"},
                    "processed_date": {"type": "date"}
                }
            }
        }

    def _get_vectors_mapping(self) -> Dict:
        """Retourne le mapping pour l'index vectors."""
        return {
            "settings": {
                "number_of_shards": settings.ELASTICSEARCH_NUMBER_OF_SHARDS,
                "number_of_replicas": settings.ELASTICSEARCH_NUMBER_OF_REPLICAS,
                "analysis": {
                    "analyzer": {
                        "french_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "french_elision",
                                "french_stop",
                                "french_stemmer"
                            ]
                        }
                    },
                    "filter": {
                        "french_elision": {
                            "type": "elision",
                            "articles_case": True,
                            "articles": ["l", "m", "t", "qu", "n", "s", "j", "d", "c"]
                        },
                        "french_stop": {
                            "type": "stop",
                            "stopwords": "_french_"
                        },
                        "french_stemmer": {
                            "type": "stemmer",
                            "language": "light_french"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "vector_id": {"type": "keyword"},
                    "source_id": {"type": "keyword"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "similarity": "cosine",
                        "index": True
                    },
                    "metadata": {
                        "type": "object",
                        "dynamic": True
                    },
                    "created_at": {"type": "date"}
                }
            }
        }

    def _get_chunks_mapping(self) -> Dict:
        """Retourne le mapping pour l'index chunks."""
        return {
            "settings": {
                "number_of_shards": settings.ELASTICSEARCH_NUMBER_OF_SHARDS,
                "number_of_replicas": settings.ELASTICSEARCH_NUMBER_OF_REPLICAS,
                "analysis": {  # Ajout de la configuration de l'analyseur
                    "analyzer": {
                        "french_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "french_elision",
                                "french_stop",
                                "french_stemmer"
                            ]
                        }
                    },
                    "filter": {
                        "french_elision": {
                            "type": "elision",
                            "articles_case": True,
                            "articles": ["l", "m", "t", "qu", "n", "s", "j", "d", "c"]
                        },
                        "french_stop": {
                            "type": "stop",
                            "stopwords": "_french_"
                        },
                        "french_stemmer": {
                            "type": "stemmer",
                            "language": "light_french"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "content": {
                        "type": "text",
                        "analyzer": "french_analyzer"
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "similarity": "cosine",
                        "index": True
                    },
                    "metadata": {
                        "type": "object",
                        "dynamic": True
                    }
                }
            }
        }

    async def create_index(self, index_name: str, config: Dict) -> None:
        """Crée un nouvel index avec la configuration spécifiée."""
        try:
            await self.es.indices.create(
                index=index_name,
                body=config
            )
            logger.info(f"Index {index_name} créé avec succès")
        except Exception as e:
            logger.error(f"Erreur création index {index_name}: {e}")
            raise

    async def update_mapping_if_needed(self, index_name: str, config: Dict) -> None:
        """Met à jour le mapping d'un index existant si nécessaire."""
        try:
            current_mapping = await self.es.indices.get_mapping(index=index_name)
            current_properties = current_mapping[index_name]["mappings"].get("properties", {})
            new_properties = config["mappings"]["properties"]

            if self._mapping_needs_update(current_properties, new_properties):
                logger.info(f"Mise à jour du mapping pour {index_name}")
                await self._reindex_with_new_mapping(index_name, config)
        except Exception as e:
            logger.error(f"Erreur vérification mapping {index_name}: {e}")
            raise

    def _mapping_needs_update(self, current: Dict, new: Dict) -> bool:
        """Vérifie si le mapping a besoin d'être mis à jour."""
        def normalize_mapping(mapping: Dict) -> Dict:
            """Normalise un mapping pour la comparaison."""
            return {k: {
                "type": v.get("type"),
                "dims": v.get("dims") if v.get("type") == "dense_vector" else None
            } for k, v in mapping.items()}

        current_normalized = normalize_mapping(current)
        new_normalized = normalize_mapping(new)
        return current_normalized != new_normalized

    async def _reindex_with_new_mapping(self, index_name: str, config: Dict) -> None:
        """Réindexe les données avec un nouveau mapping."""
        temp_index = f"{index_name}_temp_{int(time.time())}"
        try:
            # Créer le nouvel index temporaire
            await self.create_index(temp_index, config)

            # Réindexer les données
            await self.es.reindex(
                body={
                    "source": {"index": index_name},
                    "dest": {"index": temp_index}
                }
            )

            # Supprimer l'ancien index
            await self.es.indices.delete(index=index_name)

            # Renommer le nouvel index
            await self.es.indices.put_alias(index=temp_index, name=index_name)

            logger.info(f"Réindexation terminée pour {index_name}")

        except Exception as e:
            logger.error(f"Erreur réindexation {index_name}: {e}")
            raise

    async def bulk_index_documents(
        self,
        documents: List[Dict],
        index_name: Optional[str] = None
    ) -> Tuple[int, List[Dict]]:
        """Indexe plusieurs documents en masse."""
        if not documents:
            return 0, []

        index = index_name or f"{self.index_prefix}_documents"
        actions = []
        
        for doc in documents:
            action = {
                "_index": index,
                "_source": {
                    **doc,
                    "processed_date": datetime.utcnow().isoformat()
                }
            }
            if "id" in doc:
                action["_id"] = doc["id"]
            actions.append(action)

        try:
            success, errors = await async_bulk(
                client=self.es,
                actions=actions,
                chunk_size=500,
                max_retries=3,
                raise_on_error=False
            )
            
            logger.info(f"Indexation en masse: {success} succès, {len(errors)} erreurs")
            return success, errors
            
        except BulkIndexError as e:
            logger.error(f"Erreur indexation en masse: {e.errors}")
            return 0, e.errors
        except Exception as e:
            logger.error(f"Erreur inattendue lors de l'indexation en masse: {e}")
            return 0, [{"error": str(e)}]

    async def cleanup(self):
        """Nettoie les ressources."""
        # Ce gestionnaire n'a pas besoin de nettoyage spécial car il utilise
        # le client ES qui sera nettoyé par ailleurs
        pass
