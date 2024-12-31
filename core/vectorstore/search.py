# core/vectorstore/search.py
from elasticsearch import RequestError
from elasticsearch.exceptions import ConnectionError as ESConnectionError
from typing import List, Dict, Optional, Any, Tuple
import logging
import asyncio
import json
from datetime import datetime

from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .query_builder import QueryBuilder
from .response_formatter import ResponseFormatter

logger = get_logger(__name__)

class SearchManager:
    def __init__(self, es_client):
        """Initialise le gestionnaire de recherche."""
        self.es = es_client
        self.index_prefix = settings.ELASTICSEARCH_INDEX_PREFIX
        self.query_builder = QueryBuilder()
        self.response_formatter = ResponseFormatter()
        self.retry_delay = 1.0
        self.max_retries = 3

    async def search_documents(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        metadata_filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Effectue une recherche dans les documents.
        
        Args:
            query: Requête textuelle
            vector: Vecteur d'embedding optionnel
            metadata_filter: Filtres de métadonnées
            **kwargs: Options supplémentaires de recherche
            
        Returns:
            Liste des résultats formatés
        """
        try:
            # Construction de la requête
            search_body = self.query_builder.build_search_query(
                query=query,
                vector=vector,
                metadata_filter=metadata_filter,
                **kwargs
            )

            # Exécution avec retry
            response = await self._execute_with_retry(
                lambda: self.es.search(
                    index=f"{self.index_prefix}_documents",
                    body=search_body
                )
            )

            # Formatage des résultats
            results = self.response_formatter.format_search_results(response)
            
            # Métriques
            metrics.increment_counter("successful_searches")
            metrics.set_gauge("last_search_hits", len(results))
            
            return results

        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            metrics.increment_counter("failed_searches")
            return []

    async def _execute_with_retry(self, operation, max_retries: Optional[int] = None) -> Any:
        """
        Exécute une opération avec retry.
        
        Args:
            operation: Fonction à exécuter
            max_retries: Nombre maximum de tentatives
            
        Returns:
            Résultat de l'opération
        """
        retries = max_retries or self.max_retries
        last_error = None

        for attempt in range(retries):
            try:
                return await operation()
                
            except ESConnectionError as e:
                last_error = e
                logger.warning(f"Erreur de connexion (tentative {attempt + 1}/{retries}): {e}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                
            except RequestError as e:
                # Les erreurs de requête ne nécessitent pas de retry
                logger.error(f"Erreur de requête: {str(e)}")
                raise
                
            except Exception as e:
                last_error = e
                logger.error(f"Erreur inattendue (tentative {attempt + 1}/{retries}): {e}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

        raise last_error

    async def vector_search(
        self,
        vector: List[float],
        size: int = 5,
        min_score: float = 0.7,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Effectue une recherche par similarité vectorielle.
        
        Args:
            vector: Vecteur de recherche
            size: Nombre de résultats
            min_score: Score minimum de similarité
            metadata_filter: Filtres additionnels
            
        Returns:
            Liste des documents similaires
        """
        try:
            script_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": vector}
                    },
                    "min_score": min_score
                }
            }

            # Ajout des filtres si présents
            if metadata_filter:
                script_query = {
                    "bool": {
                        "must": [script_query],
                        "filter": self.query_builder._build_filters(metadata_filter)
                    }
                }

            response = await self._execute_with_retry(
                lambda: self.es.search(
                    index=f"{self.index_prefix}_documents",
                    body={
                        "size": size,
                        "query": script_query,
                        "_source": ["title", "content", "metadata"],
                        "timeout": "30s"
                    }
                )
            )

            return self.response_formatter.format_search_results(response)

        except Exception as e:
            logger.error(f"Erreur recherche vectorielle: {e}")
            return []

    async def hybrid_search(
        self,
        text_query: str,
        vector: List[float],
        weights: Tuple[float, float] = (0.3, 0.7),
        **kwargs
    ) -> List[Dict]:
        """
        Effectue une recherche hybride (texte + vecteur).
        
        Args:
            text_query: Requête textuelle
            vector: Vecteur d'embedding
            weights: Poids relatifs (texte, vecteur)
            **kwargs: Options additionnelles
            
        Returns:
            Liste des résultats combinés
        """
        try:
            text_weight, vector_weight = weights
            
            # Construction de la requête hybride
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": text_query,
                                    "fields": ["title^2", "content"],
                                    "type": "best_fields",
                                    "tie_breaker": 0.3,
                                    "boost": text_weight
                                }
                            },
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": f"cosineSimilarity(params.query_vector, 'embedding') * {vector_weight}",
                                        "params": {"query_vector": vector}
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                **kwargs
            }

            response = await self._execute_with_retry(
                lambda: self.es.search(
                    index=f"{self.index_prefix}_documents",
                    body=search_body
                )
            )

            return self.response_formatter.format_search_results(response)

        except Exception as e:
            logger.error(f"Erreur recherche hybride: {e}")
            return []

    async def get_similar_documents(
        self,
        doc_id: str,
        size: int = 5,
        min_score: float = 0.7
    ) -> List[Dict]:
        """
        Trouve des documents similaires à un document donné.
        
        Args:
            doc_id: ID du document de référence
            size: Nombre de résultats
            min_score: Score minimum de similarité
            
        Returns:
            Liste des documents similaires
        """
        try:
            # Récupération du document source
            source_doc = await self._execute_with_retry(
                lambda: self.es.get(
                    index=f"{self.index_prefix}_documents",
                    id=doc_id,
                    _source=["embedding"]
                )
            )

            if "embedding" not in source_doc["_source"]:
                logger.warning(f"Document {doc_id} n'a pas d'embedding")
                return []

            # Recherche de documents similaires
            vector = source_doc["_source"]["embedding"]
            return await self.vector_search(
                vector=vector,
                size=size + 1,  # +1 car le document source sera dans les résultats
                min_score=min_score
            )

        except Exception as e:
            logger.error(f"Erreur recherche documents similaires: {e}")
            return []