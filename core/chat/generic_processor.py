# core/chat/generic_processor.py
from typing import Dict, Optional, List
from datetime import datetime
import uuid
import torch

from core.chat.base_processor import BaseProcessor
from core.utils.logger import get_logger
from core.config import settings
from core.utils.metrics import metrics
from core.database.models import ChatHistory, ReferencedDocument
from core.database.base import DatabaseSession
from core.llm.prompt_system import PromptSystem

from api.models.responses import (
    ChatResponse,
    DocumentReference,
    ErrorResponse,
    VectorStats
)

logger = get_logger("generic_processor")

class GenericProcessor(BaseProcessor):
    """Processeur générique pour les requêtes sans contexte métier."""
    
    def __init__(self, components):
        super().__init__(components)
        self.model = components.model
        self.es_client = components.es_client
        self.prompt_system = PromptSystem()
        
    async def process_message(
        self,
        request: Dict,
        context: Optional[Dict] = None
    ) -> ChatResponse:
        """Traite un message de manière générique."""
        try:
            start_time = datetime.utcnow()
            query = request.get("query")
            if not query:
                return self._create_error_response("Query cannot be empty")
            
            # Génération de l'embedding pour la recherche
            query_vector = await self.model.create_embedding(query)
            
            # Recherche de documents pertinents
            relevant_docs = await self.es_client.search_documents(
                query=query,
                vector=query_vector,
                metadata_filter=request.get("metadata"),
                size=settings.MAX_RELEVANT_DOCS
            )
            
            # Construction des messages via PromptSystem
            messages = await self.prompt_system.build_chat_prompt(
                query=query,
                context_docs=relevant_docs,
                conversation_history=context.get("history", []) if context else None,
                language=request.get("language", "fr")
            )
            logger.debug(f"Messages formatés: {messages}")
            
            # Génération de la réponse
            model_response = await self.model.generate_response(
                messages=messages,  # Passage direct des messages formatés
                language=request.get("language", "fr"),
                response_type=request.get("response_type", "comprehensive")  # Type de réponse pour contrôler la génération
            )
            
            # Construction de la réponse
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            response = ChatResponse(
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
                    "context_used": len(relevant_docs),
                    "model_version": settings.MODEL_NAME,
                    "prompt_messages": messages
                }
            )
            
            # Sauvegarde en base de données
            await self._save_interaction(
                request=request,
                response=response,
                query_vector=query_vector,
                response_vector=await self.model.create_embedding(response.response),
                relevant_docs=relevant_docs,
                messages=messages
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur traitement générique: {e}")
            return self._create_error_response(str(e))

    async def _save_interaction(
        self,
        request: Dict,
        response: ChatResponse,
        query_vector: List[float],
        response_vector: List[float],
        relevant_docs: List[Dict],
        messages: List[Dict]
    ):
        """Sauvegarde l'interaction en base de données."""
        try:
            async with DatabaseSession() as session:
                # Création de l'historique
                chat_history = ChatHistory(
                    id=uuid.uuid4(),
                    session_id=response.session_id,
                    user_id=request.get("user_id"),
                    query=request.get("query"),
                    response=response.response,
                    query_vector=query_vector,
                    response_vector=response_vector,
                    confidence_score=response.confidence_score,
                    tokens_used=response.tokens_used,
                    processing_time=response.processing_time,
                    chat_metadata={
                        "prompt_messages": messages,
                        "application": request.get("application"),
                        "language": request.get("language", "fr"),
                        "model": settings.MODEL_NAME
                    }
                )
                session.add(chat_history)

                # Ajout des documents référencés
                for doc in relevant_docs:
                    ref_doc = ReferencedDocument(
                        id=uuid.uuid4(),
                        chat_history_id=chat_history.id,
                        document_name=doc.get("title", "Unknown"),
                        page_number=doc.get("page", 1),
                        relevance_score=float(doc.get("score", 0.0)),
                        content_snippet=doc.get("content", ""),
                        document_metadata=doc.get("metadata", {})
                    )
                    session.add(ref_doc)

                await session.commit()
                logger.info(f"Interaction sauvegardée - ID: {chat_history.id}")
                metrics.increment_counter("chat_saves")

        except Exception as e:
            logger.error(f"Erreur sauvegarde interaction: {e}")
            metrics.increment_counter("chat_save_errors")

    def _create_error_response(self, error_message: str) -> ChatResponse:
        """Crée une réponse d'erreur formatée."""
        return ChatResponse(
            response="Je suis désolé, une erreur est survenue lors du traitement de votre demande.",
            session_id=str(uuid.uuid4()),
            conversation_id=str(uuid.uuid4()),
            documents=[],
            confidence_score=0.0,
            processing_time=0.0,
            tokens_used=0,
            metadata={
                "processor_type": "generic",
                "error": error_message,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    def _calculate_confidence(self, docs: List[Dict]) -> float:
        """Calcule le score de confiance moyen des documents."""
        if not docs:
            return 0.0
        scores = [doc.get("score", 0.0) for doc in docs]
        return sum(scores) / len(scores) if scores else 0.0