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
    ) -> ChatResponse:  # Important: Retourne ChatResponse, pas Dict
        """Traite un message de manière générique."""
        try:
            start_time = datetime.utcnow()
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
            model_response = await self.model.generate_response(
                query=query,
                context_docs=relevant_docs,
                language=request.get("language", "fr")
            )
            
            # Construction de la réponse en tant qu'instance de ChatResponse
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            return ChatResponse(
                response=model_response.get("response", ""),
                session_id=str(request.get("session_id", uuid.uuid4())),
                conversation_id=str(uuid.uuid4()),
                documents=[
                    DocumentReference(
                        title=doc.get("title", ""),
                        page=doc.get("page", 1),
                        score=float(doc.get("score", 0.0)),
                        content=doc.get("content", ""),
                        metadata=doc.get("metadata", {})
                    ) for doc in relevant_docs
                ],
                confidence_score=self._calculate_confidence(relevant_docs),
                processing_time=processing_time,
                tokens_used=model_response.get("tokens_used", {}).get("total", 0),
                tokens_details=model_response.get("tokens_used", {}),
                metadata={
                    "processor_type": "generic",
                    "timestamp": datetime.utcnow().isoformat(),
                    "context_used": len(relevant_docs)
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur traitement générique: {e}")
            return ChatResponse(
                response="Je suis désolé, une erreur est survenue lors du traitement de votre demande.",
                session_id=str(request.get("session_id", uuid.uuid4())),
                conversation_id=str(uuid.uuid4()),
                documents=[],
                confidence_score=0.0,
                processing_time=0.0,
                tokens_used=0,
                metadata={
                    "processor_type": "generic",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
