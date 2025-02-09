# core/search/search_manager.py
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings

logger = get_logger("search_manager")

class SearchManager:
    """Gestionnaire de recherche documentaire."""
    
    def __init__(self, components):
        """Initialise le gestionnaire de recherche."""
        self.es_client = components.es_client if hasattr(components, 'es_client') else None
        self.model = components.model if hasattr(components, 'model') else None
        self.enabled = True  # État par défaut

    def set_enabled(self, enabled: bool):
        """Active ou désactive la recherche documentaire."""
        self.enabled = enabled
        logger.info(f"Recherche documentaire {'activée' if enabled else 'désactivée'}")

    async def search_context(
        self, 
        query: str,
        metadata_filter: Optional[Dict] = None,
        min_score: float = 0.3,
        max_docs: int = 5
    ) -> List[Dict]:
        """
        Recherche le contexte documentaire si activé.
        
        Args:
            query: Requête de l'utilisateur
            metadata_filter: Filtres de métadonnées
            min_score: Score minimum de pertinence
            max_docs: Nombre maximum de documents
            
        Returns:
            Liste des documents pertinents ou liste vide si désactivé
        """
        if not self.enabled:
            logger.debug("Recherche documentaire désactivée, retour liste vide")
            return []

        try:
            # Génération du vecteur de recherche
            if not self.model:
                logger.warning("Modèle non disponible pour la génération de vecteurs")
                return []
                
            query_vector = await self.model.create_embedding(query)
            
            if not query_vector:
                logger.warning("Échec génération vecteur de recherche")
                return []

            # Recherche des documents
            if not self.es_client:
                logger.warning("Client Elasticsearch non disponible")
                return []
                
            docs = await self.es_client.search_documents(
                query=query,
                vector=query_vector,
                metadata_filter=metadata_filter,
                size=max_docs,
                min_score=min_score
            )
            
            # Filtrage et tri des résultats
            filtered_docs = [
                doc for doc in docs 
                if doc.get("score", 0) >= min_score
            ]
            
            # Tri par score décroissant
            filtered_docs.sort(
                key=lambda x: x.get("score", 0),
                reverse=True
            )

            metrics.increment_counter("document_searches")
            logger.info(f"Recherche documentaire : {len(filtered_docs)} documents trouvés")
            
            return filtered_docs[:max_docs]

        except Exception as e:
            logger.error(f"Erreur recherche documentaire: {e}")
            metrics.increment_counter("search_errors")
            return []

    async def get_search_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de recherche."""
        try:
            return {
                "enabled": self.enabled,
                "es_available": self.es_client is not None,
                "model_available": self.model is not None,
                "searches_count": metrics.counters.get("document_searches", 0),
                "errors_count": metrics.counters.get("search_errors", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Erreur récupération stats recherche: {e}")
            return {}