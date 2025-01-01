from typing import List, Dict, Optional, Any, Union
from elasticsearch import RequestError
from elasticsearch.exceptions import ConnectionError as ESConnectionError
import asyncio
import json
from datetime import datetime

from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .query_builder import QueryBuilder
from .response_formatter import ResponseFormatter

logger = get_logger("search")

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
        size: int = 5
    ) -> List[Dict]:
        """
        Effectue une recherche hybride (vectorielle + textuelle) dans les documents.
        
        Args:
            query: Texte de la requête
            vector: Vecteur d'embedding de la requête
            metadata_filter: Filtres sur les métadonnées
            size: Nombre de résultats souhaités
            
        Returns:
            Liste des documents trouvés avec leurs scores
        """
        try:
            with metrics.timer("search.document_search"):
                # Construction de la requête
                search_body = self.query_builder.build_search_query(
                    query=query,
                    vector=vector,
                    metadata_filter=metadata_filter,
                    size=size,
                    min_score=0.1
                )

                # Log de la requête en mode debug
                logger.debug(f"Requête ES: {json.dumps(search_body, indent=2)}")

                # Vérification de l'existence de l'index
                index_name = f"{self.index_prefix}_documents"
                if not await self.es.indices.exists(index=index_name):
                    logger.error(f"Index {index_name} non trouvé")
                    return []

                # Exécution de la recherche avec retry
                search_response = await self._execute_with_retry(
                    lambda: self.es.search(
                        index=index_name,
                        body=search_body
                    )
                )

                # Formatage des résultats
                results = self.response_formatter.format_search_results(search_response)
                
                # Log des métriques
                num_results = len(results)
                avg_score = sum(r.get("score", 0) for r in results) / max(num_results, 1)
                metrics.set_gauge("search.num_results", num_results)
                metrics.set_gauge("search.avg_score", avg_score)

                return results

        except RequestError as e:
            # Log détaillé des erreurs de requête
            error_body = e.body if hasattr(e, 'body') else {}
            if isinstance(error_body, dict):
                reason = error_body.get('error', {}).get('reason', str(e))
                root_cause = error_body.get('error', {}).get('root_cause', [])
                logger.error(f"Erreur ES - Raison: {reason}")
                if root_cause:
                    logger.error(f"Cause racine: {json.dumps(root_cause, indent=2)}")
            raise

        except Exception as e:
            logger.error(f"Erreur recherche: {e}", exc_info=True)
            metrics.increment_counter("search.errors")
            return []

    async def _execute_with_retry(
        self,
        operation,
        max_retries: Optional[int] = None
    ) -> Any:
        """
        Exécute une opération ES avec retry en cas d'erreur.
        
        Args:
            operation: Opération à exécuter
            max_retries: Nombre max de tentatives
            
        Returns:
            Résultat de l'opération
        """
        retries = max_retries or self.max_retries
        last_error = None

        for attempt in range(retries):
            try:
                return await operation()
                
            except ESConnectionError as e:
                # Erreurs de connexion - retry possible
                last_error = e
                logger.warning(f"Erreur connexion ES (tentative {attempt + 1}/{retries}): {e}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                
            except RequestError as e:
                # Erreurs de requête - pas de retry
                error_body = e.body if hasattr(e, 'body') else {}
                if isinstance(error_body, dict):
                    logger.error(f"Erreur requête ES: {json.dumps(error_body, indent=2)}")
                raise
                
            except Exception as e:
                # Autres erreurs - retry possible
                last_error = e
                logger.error(f"Erreur inattendue (tentative {attempt + 1}/{retries}): {e}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

        raise last_error

    async def find_similar_documents(
        self,
        vector: List[float],
        metadata_filter: Optional[Dict] = None,
        min_score: float = 0.7,
        size: int = 5
    ) -> List[Dict]:
        """
        Trouve les documents similaires basés sur un vecteur.
        
        Args:
            vector: Vecteur de référence
            metadata_filter: Filtres sur les métadonnées
            min_score: Score minimum de similarité
            size: Nombre de résultats
            
        Returns:
            Liste des documents similaires
        """
        try:
            # Construction de la requête vectorielle
            search_body = self.query_builder.build_vector_query(
                vector=vector,
                metadata_filter=metadata_filter,
                min_score=min_score,
                size=size
            )

            # Exécution de la recherche
            results = await self.search_documents(
                query="",  # Pas de recherche textuelle
                vector=vector,
                metadata_filter=metadata_filter,
                size=size
            )

            return results

        except Exception as e:
            logger.error(f"Erreur recherche similarité: {e}")
            return []

    async def get_document_by_id(
        self,
        doc_id: str,
        include_vector: bool = False
    ) -> Optional[Dict]:
        """
        Récupère un document par son ID.
        
        Args:
            doc_id: ID du document
            include_vector: Inclure le vecteur d'embedding
            
        Returns:
            Document trouvé ou None
        """
        try:
            index_name = f"{self.index_prefix}_documents"
            response = await self.es.get(
                index=index_name,
                id=doc_id,
                _source_excludes=['embedding'] if not include_vector else None
            )
            return response['_source']

        except Exception as e:
            logger.error(f"Erreur récupération document {doc_id}: {e}")
            return None

    async def get_index_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques des index."""
        try:
            index_name = f"{self.index_prefix}_documents"
            stats = await self.es.indices.stats(index=index_name)
            return {
                "doc_count": stats['_all']['total']['docs']['count'],
                "store_size": stats['_all']['total']['store']['size_in_bytes'],
                "indexing_stats": stats['_all']['total']['indexing']
            }

        except Exception as e:
            logger.error(f"Erreur récupération stats: {e}")
            return {}
