# core/vectorstore/index.py
from typing import Dict, List, Optional, Union, Any, Generator, Tuple
from datetime import datetime
from elasticsearch.helpers import BulkIndexError, async_bulk
import asyncio
import json
import hashlib
from pathlib import Path

from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger(__name__)

class IndexManager:
    def __init__(self, es_client):
        """Initialise le gestionnaire d'index."""
        self.es = es_client
        self.index_prefix = settings.ELASTICSEARCH_INDEX_PREFIX
        self.embedding_dim = settings.ELASTICSEARCH_EMBEDDING_DIM
        self.template_name = f"{self.index_prefix}_template"
        self.bulk_size = 1000
        self.refresh_interval = "30s"

    async def setup_indices(self):
        """Configure les indices nécessaires."""
        try:
            # Suppression des anciens templates si nécessaire
            if await self.es.indices.exists_index_template(name=self.template_name):
                await self.es.indices.delete_index_template(name=self.template_name)

            await self._setup_templates()
            await self._ensure_indices_exist()
            logger.info("Configuration des indices terminée")
            
        except Exception as e:
            logger.error(f"Erreur configuration indices: {e}")
            raise

    async def _setup_templates(self):
        """Configure les templates des indices avec paramètres optimisés."""
        template = {
            "index_patterns": [f"{self.index_prefix}_*"],
            "priority": 1,
            "template": {
                "settings": {
                    "index": {
                        "mapping": {
                            "total_fields": {"limit": 2000}
                        },
                        "refresh_interval": self.refresh_interval,
                        "number_of_shards": settings.ELASTICSEARCH_NUMBER_OF_SHARDS,
                        "number_of_replicas": settings.ELASTICSEARCH_NUMBER_OF_REPLICAS,
                        "max_result_window": 20000,
                        "analysis": self._get_analysis_settings()
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
                                },
                                "vector": {
                                    "type": "dense_vector",
                                    "dims": self.embedding_dim,
                                    "index": True,
                                    "similarity": "cosine"
                                }
                            }
                        },
                        "content": {
                            "type": "text",
                            "analyzer": "content_analyzer",
                            "term_vector": "with_positions_offsets"
                        },
                        "content_vector": {
                            "type": "dense_vector",
                            "dims": self.embedding_dim,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {
                            "type": "object",
                            "dynamic": True,
                            "properties": {
                                "source": {"type": "keyword"},
                                "application": {"type": "keyword"},
                                "tags": {"type": "keyword"},
                                "language": {"type": "keyword"},
                                "version": {"type": "keyword"},
                                "last_updated": {"type": "date"},
                                "processed": {"type": "boolean"},
                                "importance_score": {"type": "float"},
                                "confidence": {"type": "float"},
                                "custom_data": {"type": "object", "enabled": False}
                            }
                        },
                        "indexed_at": {"type": "date"},
                        "checksum": {"type": "keyword"},
                        "status": {"type": "keyword"}
                    }
                }
            }
        }

        await self.es.indices.put_index_template(
            name=self.template_name,
            body=template
        )
        logger.info(f"Template {self.template_name} créé")

    def _get_analysis_settings(self) -> Dict:
        """Retourne la configuration d'analyse pour ES."""
        return {
            "analyzer": {
                "content_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        "word_delimiter_graph",
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

    async def bulk_index_documents(
        self,
        documents: List[Dict],
        index_name: Optional[str] = None,
        chunk_size: int = 500
    ) -> Tuple[int, List[Dict]]:
        """
        Indexe un lot de documents avec gestion optimisée.
        
        Args:
            documents: Liste des documents à indexer
            index_name: Nom de l'index (optionnel)
            chunk_size: Taille des lots
            
        Returns:
            Tuple (nombre succès, liste erreurs)
        """
        try:
            index = index_name or f"{self.index_prefix}_documents"
            successes = 0
            errors = []

            async def document_generator() -> Generator[Dict, None, None]:
                for doc in documents:
                    # Préparation du document
                    processed_doc = self._prepare_document(doc)
                    
                    yield {
                        "_index": index,
                        "_id": processed_doc.pop("id", None) or self._generate_doc_id(processed_doc),
                        "_source": processed_doc
                    }

            # Indexation par lots avec async_bulk
            async for ok, result in async_bulk(
                client=self.es,
                actions=document_generator(),
                chunk_size=chunk_size,
                raise_on_error=False,
                raise_on_exception=False,
                refresh="wait_for"
            ):
                if ok:
                    successes += 1
                else:
                    errors.append(result)

            metrics.increment_counter("documents_indexed", successes)
            if errors:
                metrics.increment_counter("indexing_errors", len(errors))

            return successes, errors

        except Exception as e:
            logger.error(f"Erreur indexation en masse: {e}")
            metrics.increment_counter("bulk_indexing_errors")
            raise

    def _prepare_document(self, doc: Dict) -> Dict:
        """Prépare un document pour l'indexation."""
        prepared_doc = doc.copy()
        
        # Ajout des champs système
        prepared_doc.update({
            "indexed_at": datetime.utcnow().isoformat(),
            "checksum": self._calculate_document_hash(doc),
            "status": "active"
        })

        # Validation et nettoyage des vecteurs
        if "content_vector" in prepared_doc:
            vector = prepared_doc["content_vector"]
            if len(vector) != self.embedding_dim:
                logger.warning(f"Dimension du vecteur incorrecte: {len(vector)} != {self.embedding_dim}")
                del prepared_doc["content_vector"]

        # Enrichissement des métadonnées
        metadata = prepared_doc.get("metadata", {})
        metadata.update({
            "last_updated": datetime.utcnow().isoformat(),
            "processed": True
        })
        prepared_doc["metadata"] = metadata

        return prepared_doc

    def _calculate_document_hash(self, doc: Dict) -> str:
        """Calcule un hash unique pour le document."""
        content = f"{doc.get('title', '')}{doc.get('content', '')}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _generate_doc_id(self, doc: Dict) -> str:
        """Génère un ID unique pour le document."""
        base = f"{doc.get('title', '')}-{doc.get('metadata', {}).get('source', '')}"
        return hashlib.md5(base.encode()).hexdigest()

    async def update_document(
        self,
        doc_id: str,
        update_fields: Dict,
        index_name: Optional[str] = None
    ) -> bool:
        """
        Met à jour un document existant.
        
        Args:
            doc_id: ID du document
            update_fields: Champs à mettre à jour
            index_name: Nom de l'index (optionnel)
        """
        try:
            index = index_name or f"{self.index_prefix}_documents"
            
            # Préparation de la mise à jour
            update_body = {
                "doc": {
                    **update_fields,
                    "metadata": {
                        "last_updated": datetime.utcnow().isoformat(),
                        **(update_fields.get("metadata", {}))
                    }
                }
            }

            response = await self.es.update(
                index=index,
                id=doc_id,
                body=update_body,
                refresh=True
            )

            success = response.get("result") == "updated"
            if success:
                metrics.increment_counter("documents_updated")
            return success

        except Exception as e:
            logger.error(f"Erreur mise à jour document {doc_id}: {e}")
            metrics.increment_counter("update_errors")
            return False

    async def verify_index_mapping(self) -> bool:
        """Vérifie et corrige le mapping de l'index si nécessaire."""
        try:
            index_name = f"{self.index_prefix}_documents"
            mapping = await self.es.indices.get_mapping(index=index_name)
            
            if not self._verify_mapping_structure(mapping[index_name]["mappings"]):
                logger.warning(f"Mapping incorrect pour {index_name}")
                
                # Sauvegarde et recréation
                docs = await self._backup_documents(index_name)
                await self.es.indices.delete(index=index_name)
                await self._ensure_indices_exist()
                
                if docs:
                    await self.bulk_index_documents(docs)
                    
            return True
            
        except Exception as e:
            logger.error(f"Erreur vérification mapping: {e}")
            return False

    def _verify_mapping_structure(self, mapping: Dict) -> bool:
        """Vérifie la structure du mapping."""
        required_fields = {
            "title", "content", "content_vector", "metadata", 
            "indexed_at", "checksum", "status"
        }
        
        properties = mapping.get("properties", {})
        return all(field in properties for field in required_fields)

    async def get_index_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques de l'index."""
        try:
            index_name = f"{self.index_prefix}_documents"
            stats = await self.es.indices.stats(index=index_name)
            
            return {
                "doc_count": stats["_all"]["total"]["docs"]["count"],
                "store_size": stats["_all"]["total"]["store"]["size_in_bytes"],
                "indexing": {
                    "total": stats["_all"]["total"]["indexing"]["index_total"],
                    "time_ms": stats["_all"]["total"]["indexing"]["index_time_in_millis"]
                },
                "search": {
                    "total": stats["_all"]["total"]["search"]["query_total"],
                    "time_ms": stats["_all"]["total"]["search"]["query_time_in_millis"]
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur récupération stats: {e}")
            return {}
