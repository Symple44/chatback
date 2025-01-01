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
        """Configure les indices avec mapping optimisé."""
        try:
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
                                    "filter": [
                                        "lowercase",
                                        "asciifolding",
                                        "word_delimiter_graph",
                                        "fr_stop",
                                        "fr_stemmer"
                                    ]
                                }
                            },
                            "filter": {
                                "fr_stop": {
                                    "type": "stop",
                                    "stopwords": "_french_"
                                },
                                "fr_stemmer": {
                                    "type": "stemmer",
                                    "language": "light_french"
                                }
                            }
                        },
                        "number_of_shards": settings.ELASTICSEARCH_NUMBER_OF_SHARDS,
                        "number_of_replicas": settings.ELASTICSEARCH_NUMBER_OF_REPLICAS,
                        "refresh_interval": settings.ELASTICSEARCH_REFRESH_INTERVAL,
                        "index": {
                            "similarity": {
                                "scripted_tfidf": {
                                    "type": "scripted",
                                    "script": {
                                        "source": "double tf = Math.sqrt(doc.freq); double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0; return query.boost * tf * idf;"
                                    }
                                }
                            }
                        }
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
                                "analyzer": "content_analyzer",
                                "term_vector": "with_positions_offsets",
                                "similarity": "scripted_tfidf"
                            },
                            "embedding": {
                                "type": "dense_vector",
                                "dims": self.embedding_dim,
                                "similarity": "cosine",
                                "index": True
                            },
                            "metadata": {
                                "type": "object",
                                "dynamic": True,
                                "properties": {
                                    "source": {"type": "keyword"},
                                    "last_updated": {"type": "date"},
                                    "confidence": {"type": "float"},
                                    "tags": {"type": "keyword"},
                                    "application": {"type": "keyword"},
                                    "language": {"type": "keyword"},
                                    "version": {"type": "keyword"}
                                }
                            },
                            "processed_date": {"type": "date"},
                            "importance_score": {"type": "float"}
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

            # Création des indices nécessaires
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
        importance_score: float = 1.0,
        refresh: bool = True
    ) -> bool:
        """
        Indexe un document avec prétraitement et gestion améliorée des métadonnées.
        
        Args:
            title: Titre du document
            content: Contenu du document
            vector: Vecteur d'embedding optionnel
            metadata: Métadonnées additionnelles
            importance_score: Score d'importance du document
            refresh: Rafraîchir l'index immédiatement
            
        Returns:
            bool: Succès de l'indexation
        """
        try:
            # Prétraitement du document
            processed_doc = self.preprocessor.preprocess_document({
                "content": content,
                "metadata": metadata or {},
                "title": title,
                "doc_id": hashlib.sha256(f"{title}-{content}".encode()).hexdigest()
            })

            # Construction du document enrichi
            doc = {
                "title": title,
                "content": content,
                "processed_content": {
                    "sections": processed_doc["sections"],
                    "importance_scores": [s.importance_score for s in processed_doc["sections"]]
                },
                "metadata": {
                    **(metadata or {}),
                    "word_count": len(content.split()),
                    "processed_sections": len(processed_doc["sections"]),
                    "language": processed_doc.get("metadata", {}).get("language", "fr"),
                    "revision": processed_doc.get("metadata", {}).get("revision", "1"),
                    "source": processed_doc.get("metadata", {}).get("source", "unknown"),
                    "processing_info": {
                        "processor_version": "1.0",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                },
                "processed_date": datetime.utcnow().isoformat(),
                "importance_score": importance_score,
                "indexing_status": {
                    "version": 1,
                    "last_update": datetime.utcnow().isoformat(),
                    "has_embedding": vector is not None
                }
            }

            # Ajout du vecteur si fourni et valide
            if vector is not None:
                if len(vector) != self.embedding_dim:
                    logger.warning(
                        f"Dimension du vecteur incorrecte: {len(vector)} != {self.embedding_dim}"
                    )
                else:
                    doc["embedding"] = vector

            # Indexation avec retry en cas d'erreur
            for attempt in range(3):
                try:
                    response = await self.es.index(
                        index=f"{self.index_prefix}_documents",
                        document=doc,
                        refresh=refresh,
                        pipeline="document_processing",  # Pipeline de traitement optionnel
                        timeout="30s"
                    )
                    
                    successful = bool(response.get("_shards", {}).get("successful", 0))
                    if successful:
                        metrics.increment_counter("documents_indexed")
                        logger.info(
                            f"Document indexé avec succès: {title} "
                            f"({len(processed_doc['sections'])} sections)"
                        )
                    return successful

                except Exception as e:
                    if attempt == 2:  # Dernière tentative
                        raise
                    logger.warning(f"Tentative {attempt + 1} échouée: {e}")
                    await asyncio.sleep(1 * (2 ** attempt))  # Backoff exponentiel

        except Exception as e:
            logger.error(
                f"Erreur indexation document {title}: {e}",
                extra={
                    "document_title": title,
                    "error": str(e),
                    "has_vector": vector is not None
                }
            )
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
