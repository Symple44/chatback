# core/chat/generic_processor.py
from typing import Dict, Optional
from datetime import datetime
from core.chat.base_processor import BaseProcessor
from core.utils.logger import get_logger

logger = get_logger("generic_processor")

class GenericProcessor(BaseProcessor):
    """Processeur générique pour les requêtes sans contexte métier."""
    
    def __init__(self, components):
        super().__init__(components)
        self.model = components.model
        self.es_client = components.es_client
        
    async def process_message(
        self,
        request: Dict,
        context: Optional[Dict] = None
    ) -> Dict:
        """Traite un message de manière générique."""
        try:
            query = request["query"]
            
            # Génération de l'embedding pour la recherche
            query_vector = await self.model.create_embedding(query)
            
            # Recherche de documents pertinents
            relevant_docs = await self.es_client.search_documents(
                query=query,
                vector=query_vector,
                metadata_filter=request.get("metadata"),
                size=5
            )
            
            # Génération de la réponse
            response = await self.model.generate_response(
                query=query,
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
            error_response = {
                "response": "Je suis désolé, une erreur est survenue lors du traitement de votre demande.",
                "confidence_score": 0.0,
                "documents": [],
                "tokens_used": {},
                "metadata": {
                    "processor_type": "generic",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            return error_response
