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

    def _convert_es_response(self, response: Any) -> Dict:
        """Convertit une réponse Elasticsearch en dictionnaire."""
        if hasattr(response, 'body'):
            return response.body
        elif hasattr(response, 'raw'):
            return response.raw
        return dict(response)

    async def search_documents(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        metadata_filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Effectue une recherche dans les documents.
        """
        try:
            # Construction de la requête
            search_body = self.query_builder.build_search_query(
                query=query,
                vector=vector,
                metadata_filter=metadata_filter,
                **kwargs
            )

            # Log de la requête pour le debug
            logger.debug(f"Requête ES: {json.dumps(search_body, indent=2)}")

            # Vérification de l'existence de l'index avant la recherche
            index_name = f"{self.index_prefix}_documents"
            exists_response = await self.es.indices.exists(index=index_name)
            if not exists_response:
                logger.error(f"Index {index_name} n'existe pas")
                return []

            # Récupération du mapping pour debug
            try:
                mapping_response = await self.es.indices.get_mapping(index=index_name)
                mapping_dict = self._convert_es_response(mapping_response)
                logger.debug(f"Mapping actuel: {json.dumps(mapping_dict, indent=2)}")
            except Exception as e:
                logger.error(f"Erreur lors de la récupération du mapping: {e}")

            try:
                # Exécution de la recherche
                search_response = await self._execute_with_retry(
                    lambda: self.es.search(
                        index=index_name,
                        body=search_body
                    )
                )

                # Conversion et log de la réponse
                response_dict = self._convert_es_response(search_response)
                logger.debug(f"Réponse ES: {json.dumps(response_dict, indent=2)}")

                # Formatage des résultats
                results = self.response_formatter.format_search_results(response_dict)
                
                # Métriques
                metrics.increment_counter("successful_searches")
                metrics.set_gauge("last_search_hits", len(results))
                
                return results

            except RequestError as e:
                # Log détaillé de l'erreur
                error_body = self._convert_es_response(e.body) if hasattr(e, 'body') else {}
                if isinstance(error_body, dict):
                    reason = error_body.get('error', {}).get('reason', str(e))
                    root_cause = error_body.get('error', {}).get('root_cause', [])
                    logger.error(f"Erreur ES détaillée - Raison: {reason}")
                    if root_cause:
                        logger.error(f"Cause racine: {json.dumps(root_cause, indent=2)}")
                raise

        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}", exc_info=True)
            metrics.increment_counter("failed_searches")
            return []

    async def _execute_with_retry(self, operation, max_retries: Optional[int] = None) -> Any:
        """
        Exécute une opération avec retry.
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
                # Log détaillé des erreurs de requête
                error_body = self._convert_es_response(e.body) if hasattr(e, 'body') else {}
                if isinstance(error_body, dict):
                    logger.error(f"Erreur de requête ES: {json.dumps(error_body, indent=2)}")
                raise
                
            except Exception as e:
                last_error = e
                logger.error(f"Erreur inattendue (tentative {attempt + 1}/{retries}): {e}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

        raise last_error

    async def get_mapping(self) -> Dict:
        """Récupère le mapping actuel de l'index."""
        try:
            index_name = f"{self.index_prefix}_documents"
            mapping_response = await self.es.indices.get_mapping(index=index_name)
            return self._convert_es_response(mapping_response)
        except Exception as e:
            logger.error(f"Erreur récupération mapping: {e}")
            return {}
