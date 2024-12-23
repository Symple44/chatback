# core/vectorstore/search.py
from elasticsearch import Elasticsearch, helpers
from typing import List, Dict, Optional
import logging
from core.config import settings
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
import elastic_transport
import asyncio

logger = logging.getLogger(__name__)

class ElasticsearchClient:
    def __init__(self):
        """Initialize Elasticsearch client."""
        try:
            self.es = Elasticsearch(
                [settings.ELASTICSEARCH_HOST],
                basic_auth=(settings.ELASTICSEARCH_USER, settings.ELASTICSEARCH_PASSWORD),
                verify_certs=False,
                ssl_show_warn=False
            )
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"}
            )
            
            self.vector_store = ElasticsearchStore(
                es_connection=self.es,
                index_name="documents",
                embedding=self.embeddings
            )
            
            # Vérification synchrone initiale
            if not self.es.ping():
                raise ConnectionError("Impossible de se connecter à Elasticsearch.")
                
            logger.info("Connexion Elasticsearch initialisée avec succès")
            self._setup_index()
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation d'Elasticsearch : {e}")
            raise

    def _setup_index(self):
        """Configure l'index Elasticsearch."""
        try:
            if not self.es.indices.exists(index="documents"):
                index_body = {
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
                        "number_of_shards": 2,
                        "number_of_replicas": 1
                    },
                    "mappings": {
                        "properties": {
                            "title": {
                                "type": "text",
                                "analyzer": "content_analyzer",
                                "fields": {"keyword": {"type": "keyword"}}
                            },
                            "content": {"type": "text", "analyzer": "content_analyzer"},
                            "metadata": {"type": "object"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": settings.ELASTICSEARCH_EMBEDDING_DIM,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "timestamp": {"type": "date"}
                        }
                    }
                }
                self.es.indices.create(index="documents", body=index_body)
                logger.info("Index 'documents' créé avec succès")

        except Exception as e:
            logger.error(f"Erreur setup_index : {e}")

    async def check_connection(self) -> bool:
        """Vérifie la connexion à Elasticsearch de manière asynchrone."""
        try:
            return self.es.ping()
        except Exception as e:
            logger.error(f"Erreur de connexion Elasticsearch: {e}")
            return False

    async def index_document(self, title: str, content: str, metadata: Optional[Dict] = None):
        """Indexe un seul document dans Elasticsearch."""
        try:
            body = {
                "title": title,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            response = self.es.index(index="documents", body=body)
            logger.info(f"Document indexé : {title}")
            return response['_id']
        except Exception as e:
            logger.error(f"Erreur lors de l'indexation de {title} : {e}")
            return None

    async def bulk_index_documents(self, documents: List[Dict[str, str]]):
        """Indexation en masse avec support des embeddings."""
        try:
            # Préparation des documents pour l'indexation standard
            actions = []
            texts = []
            metadatas = []
            
            for doc in documents:
                embedding = self.embeddings.embed_query(doc["content"])
                
                actions.append({
                    "_index": "documents",
                    "_source": {
                        "title": doc["title"],
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {}),
                        "timestamp": datetime.utcnow().isoformat(),
                        "embedding": embedding
                    }
                })
                
                texts.append(doc["content"])
                metadatas.append({"title": doc["title"], **doc.get("metadata", {})})
            
            # Indexation dans l'index principal
            helpers.bulk(self.es, actions)
            
            # Indexation dans le vectorstore Langchain
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)
            
            logger.info(f"{len(documents)} documents indexés avec embeddings.")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'indexation en bulk : {e}")

    async def search_documents(
        self,
        query: str,
        metadata: Optional[Dict] = None,
        size: int = 10,
        page: int = 0,
        use_semantic: bool = True
    ) -> List[Dict]:
        """Recherche hybride combinant recherche sémantique et textuelle."""
        try:
            results = []
            
            if use_semantic:
                # Recherche sémantique via Langchain
                semantic_docs = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=size
                )
                
                # Conversion des résultats Langchain
                for doc, score in semantic_docs:
                    results.append({
                        "title": doc.metadata.get("title", "Unknown"),
                        "content": doc.page_content,
                        "score": float(score),
                        "metadata": doc.metadata,
                        "search_type": "semantic"
                    })
            
            # Recherche textuelle classique
            search_query = {
                "query": {
                    "bool": {
                        "must": [{
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "content^2"],
                                "type": "most_fields",
                                "operator": "and",
                                "minimum_should_match": "80%"
                            }
                        }],
                        "should": [{
                            "match_phrase": {
                                "content": {
                                    "query": query,
                                    "boost": 2
                                }
                            }
                        }]
                    }
                },
                "highlight": {
                    "fields": {
                        "content": {
                            "fragment_size": 300,
                            "number_of_fragments": 3,
                            "fragmenter": "span"
                        }
                    },
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                },
                "_source": ["title", "content", "metadata"],
                "size": size,
                "from": page * size
            }

            if metadata:
                for key, value in metadata.items():
                    search_query["query"]["bool"].setdefault("filter", []).append(
                        {"term": {f"metadata.{key}": value}}
                    )

            # Exécution de la recherche textuelle
            response = self.es.search(index="documents", body=search_query)
            
            # Fusion et déduplication des résultats
            seen_contents = set()
            final_results = []
            
            for result in results + [{
                "title": hit["_source"]["title"],
                "content": hit["_source"]["content"],
                "score": hit["_score"],
                "highlights": hit.get("highlight", {}).get("content", []),
                "metadata": hit["_source"].get("metadata", {}),
                "search_type": "textual"
            } for hit in response["hits"]["hits"]]:
                content_hash = hash(result["content"])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    final_results.append(result)
            
            # Tri des résultats par score
            final_results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"{len(final_results)} résultats trouvés pour la requête : {query}")
            return final_results[:size]

        except Exception as e:
            logger.error(f"Erreur lors de la recherche : {e}")
            return []