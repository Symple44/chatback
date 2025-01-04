# core/chat/generic_processor.py
from typing import Dict, Optional
from datetime import datetime
from .base_processor import BaseProcessor
from core.utils.logger import get_logger

logger = get_logger("generic_processor")

class GenericProcessor(BaseProcessor):
    """Processeur générique pour les requêtes sans contexte métier."""
    
    def __init__(self, components):
        super().__init__(components)
        self.model = components.model
        
    async def process_message(self, request: Dict, context: Optional[Dict] = None) -> Dict:
        """Traite un message de manière générique."""
        try:
            # Génération de l'embedding pour la recherche
            query_vector = await self.model.create_embedding(request["query"])
            
            # Recherche de documents pertinents
            relevant_docs = await self.components.es_client.search_documents(
                query=request["query"],
                vector=query_vector,
                metadata_filter=request.get("metadata"),
                size=5
            )
            
            # Génération de la réponse
            response = await self.model.generate_response(
                query=request["query"],
                context_docs=relevant_docs,
                language=request.get("language", "fr")
            )
            
            return {
                "response": response.get("response", ""),
                "confidence_score": self._calculate_confidence(relevant_docs),
                "documents": relevant_docs,
                "tokens_used": response.get("tokens_used", {}),
                "metadata": {
                    "processor_type": "generic",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur traitement générique: {e}")
            raise
