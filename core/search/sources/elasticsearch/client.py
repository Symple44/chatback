# core/search/sources/elasticsearch/client.py
from typing import Dict, Any, Optional, List
from elasticsearch import AsyncElasticsearch
from datetime import datetime
import logging
from core.utils.logger import get_logger
from core.config.config import settings
from ...exceptions import SourceConnectionError, SearchConfigurationError

logger = get_logger("elasticsearch_client")

class ElasticsearchClient:
    """Client pour interagir avec Elasticsearch."""

    def __init__(
        self,
        host: str,
        index: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        ca_cert: Optional[str] = None,
        verify_certs: bool = True
    ):
        """
        Initialise le client Elasticsearch.
        
        Args:
            host: URL du serveur Elasticsearch
            index: Nom de l'index à utiliser
            user: Nom d'utilisateur (optionnel)
            password: Mot de passe (optionnel)
            ca_cert: Chemin du certificat CA (optionnel)
            verify_certs: Vérifier les certificats SSL
        """
        self.host = host
        self.index = index
        self.user = user
        self.password = password
        self.ca_cert = ca_cert
        self.verify_certs = verify_certs
        self.client: Optional[AsyncElasticsearch] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialise la connexion Elasticsearch."""
        try:
            if not self._initialized:
                # Configuration du client
                es_config = {
                    "hosts": [self.host],
                    "verify_certs": self.verify_certs
                }

                # Ajout de l'authentification si nécessaire
                if self.user and self.password:
                    es_config["basic_auth"] = (self.user, self.password)

                # Ajout du certificat CA si fourni
                if self.ca_cert:
                    es_config["ca_certs"] = self.ca_cert

                # Création du client
                self.client = AsyncElasticsearch(**es_config)

                # Vérification de la connexion
                if not await self.client.ping():
                    raise SourceConnectionError("Impossible de se connecter à Elasticsearch")

                # Vérification de l'index
                if not await self.client.indices.exists(index=self.index):
                    await self._create_index()

                self._initialized = True
                logger.info(f"Client Elasticsearch initialisé pour l'index {self.index}")

        except Exception as e:
            raise SourceConnectionError(f"Erreur d'initialisation Elasticsearch: {str(e)}")

    async def _create_index(self) -> None:
        """Crée l'index avec la configuration appropriée."""
        try:
            index_config = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1
                },
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "title": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": settings.ELASTICSEARCH_EMBEDDING_DIM,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "keyword"},
                                "page": {"type": "integer"},
                                "language": {"type": "keyword"},
                                "timestamp": {"type": "date"}
                            }
                        }
                    }
                }
            }
            await self.client.indices.create(index=self.index, body=index_config)
            logger.info(f"Index {self.index} créé avec succès")

        except Exception as e:
            raise SearchConfigurationError(f"Erreur création index: {str(e)}")

    async def search_documents(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        metadata_filter: Optional[Dict] = None,
        size: int = 10,
        min_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        Effectue une recherche dans Elasticsearch.
        
        Args:
            query: Texte de la requête
            vector: Vecteur de recherche (optionnel)
            metadata_filter: Filtres sur les métadonnées
            size: Nombre maximum de résultats
            min_score: Score minimum des résultats
            
        Returns:
            Résultats de la recherche
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Construction de la requête de base
            search_query = {
                "bool": {
                    "must": [
                        {"match": {"content": query}}
                    ],
                    "should": []
                }
            }

            # Ajout de la recherche vectorielle si un vecteur est fourni
            if vector:
                search_query["bool"]["should"].append({
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": vector}
                        }
                    }
                })

            # Ajout des filtres de métadonnées
            if metadata_filter:
                for key, value in metadata_filter.items():
                    search_query["bool"]["must"].append({
                        "match": {f"metadata.{key}": value}
                    })

            # Exécution de la recherche
            response = await self.client.search(
                index=self.index,
                query=search_query,
                size=size,
                min_score=min_score
            )

            logger.info(f"Recherche réussie: {len(response['hits']['hits'])} résultats trouvés")
            return response

        except Exception as e:
            raise SourceConnectionError(f"Erreur recherche: {str(e)}")

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un document par son ID."""
        if not self._initialized:
            await self.initialize()

        try:
            response = await self.client.get(index=self.index, id=doc_id)
            return response["_source"]
        except Exception as e:
            logger.error(f"Erreur récupération document {doc_id}: {e}")
            return None

    async def index_document(self, document: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """
        Indexe un document dans Elasticsearch.
        
        Args:
            document: Document à indexer
            doc_id: ID optionnel du document
            
        Returns:
            ID du document indexé
        """
        if not self._initialized:
            await self.initialize()

        try:
            response = await self.client.index(
                index=self.index,
                body=document,
                id=doc_id,
                refresh=True
            )
            return response["_id"]
        except Exception as e:
            raise SourceConnectionError(f"Erreur indexation: {str(e)}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état de santé d'Elasticsearch.
        
        Returns:
            État de santé du cluster
        """
        try:
            if not self._initialized:
                return {"status": "not_initialized"}

            health = await self.client.cluster.health()
            index_stats = await self.client.indices.stats(index=self.index)

            return {
                "status": health["status"],
                "cluster_name": health["cluster_name"],
                "number_of_nodes": health["number_of_nodes"],
                "active_shards": health["active_shards"],
                "index_stats": {
                    "doc_count": index_stats["_all"]["total"]["docs"]["count"],
                    "store_size": index_stats["_all"]["total"]["store"]["size_in_bytes"]
                },
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def cleanup(self) -> None:
        """Nettoie les ressources du client."""
        if self.client:
            await self.client.close()
            self._initialized = False
            logger.info("Client Elasticsearch fermé")