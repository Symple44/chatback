# core/chat/generic_processor.py
from typing import Dict, Optional, List
from datetime import datetime
import uuid
import torch

from core.chat.base_processor import BaseProcessor
from core.utils.logger import get_logger
from core.config.config import settings
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
            session_id = request.get("session_id")
            
            if not query:
                return self._create_error_response("Query cannot be empty")
            if not session_id:
                return self._create_error_response("Session ID is required")
                
            # Récupération de l'historique de conversation si un context existe
            conversation_history = []
            if context and "history" in context:
                conversation_history = context["history"]
            elif session_id:
                # Récupération de l'historique depuis la base de données
                async with self.components.db.session_factory() as session:
                    history = await session.execute(
                        select(ChatHistory)
                        .where(ChatHistory.session_id == session_id)
                        .order_by(ChatHistory.created_at.desc())
                        .limit(settings.MAX_HISTORY_MESSAGES)
                    )
                    history_records = history.scalars().all()
                    
                    conversation_history = []
                    for record in reversed(history_records):
                        conversation_history.extend([
                            {
                                "role": "user",
                                "content": record.query,
                                "timestamp": record.created_at.isoformat()
                            },
                            {
                                "role": "assistant",
                                "content": record.response,
                                "timestamp": record.created_at.isoformat()
                            }
                        ])

            # Génération de l'embedding pour la recherche
            query_vector = await self.model.create_embedding(query)
            
            # Recherche de documents pertinents
            relevant_docs = await self.es_client.search_documents(
                query=query,
                vector=query_vector,
                metadata_filter=request.get("metadata"),
                size=settings.MAX_RELEVANT_DOCS,
                min_score=0.3
            )
            
            # Filtrage des documents par score
            filtered_docs = [
                doc for doc in relevant_docs 
                if doc["score"] >= settings.CONFIDENCE_THRESHOLD
            ]

            # Construction du prompt avec l'historique
            messages = await self.components.prompt_system.build_chat_prompt(
                query=query,
                context_docs=filtered_docs,
                conversation_history=conversation_history,  # Ajout de l'historique
                language=request.get("language", "fr")
            )

            # Génération de la réponse
            model_response = await self.model.generate_response(
                messages=messages,
                language=request.get("language", "fr"),
                response_type=request.get("response_type", "comprehensive")
            )
            
            response_text = model_response.get("response", "")
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Sauvegarde de l'interaction
            chat_history_id = await self.components.db_manager.save_chat_interaction(
                session_id=session_id,
                user_id=request.get("user_id"),
                query=query,
                response=response_text,
                query_vector=query_vector,
                response_vector=await self.model.create_embedding(response_text),
                confidence_score=float(self._calculate_confidence(filtered_docs)),
                tokens_used=int(model_response.get("tokens_used", {}).get("total", 0)),
                processing_time=float(processing_time),
                referenced_docs=[{
                    "name": doc.get("title", doc.get("name", "Unknown Document")),
                    "page": doc.get("metadata", {}).get("page", 1),
                    "score": float(doc.get("score", 0.0)),
                    "snippet": doc.get("content", ""),
                    "metadata": doc.get("metadata", {})
                } for doc in filtered_docs] if filtered_docs else None,
                metadata={
                    "processor_type": "generic",
                    "documents_found": len(filtered_docs),
                    "language": request.get("language", "fr"),
                    "response_type": request.get("response_type", "comprehensive"),
                    "model_name": self.model.model_name if hasattr(self.model, 'model_name') else None,
                    "history_used": bool(conversation_history),  # Ajout d'un indicateur d'utilisation de l'historique
                    "history_length": len(conversation_history) if conversation_history else 0
                },
                raw_response=model_response.get("metadata", {}).get("raw_response"),
                raw_prompt=model_response.get("metadata", {}).get("raw_prompt")
            )

            return ChatResponse(
                response=response_text,
                session_id=session_id,
                conversation_id=str(uuid.uuid4()),
                documents=[
                    DocumentReference(
                        title=doc.get("title", ""),
                        page=int(doc.get("metadata", {}).get("page", 1)),
                        score=float(doc.get("score", 0.0)),
                        content=doc.get("content", ""),
                        metadata=doc.get("metadata", {})
                    ) for doc in filtered_docs
                ],
                confidence_score=float(self._calculate_confidence(filtered_docs)),
                processing_time=float(processing_time),
                tokens_used=int(model_response.get("tokens_used", {}).get("total", 0)),
                tokens_details=model_response.get("tokens_used", {}),
                metadata={
                    "processor_type": "generic",
                    "documents_found": len(filtered_docs),
                    "context_quality": float(sum(doc.get("score", 0) for doc in filtered_docs) / len(filtered_docs)) if filtered_docs else 0.0,
                    "processing_info": {
                        "embedding_generated": query_vector is not None,
                        "history_used": bool(conversation_history),
                        "history_messages": len(conversation_history) if conversation_history else 0,
                    }
                }
            )

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
        messages: List[Dict],
        session_id: str
    ):
        """Sauvegarde l'interaction en base de données."""
        try:
            async with DatabaseSession() as session:
                # Création de l'historique
                chat_history = ChatHistory(
                    id=uuid.uuid4(),
                    session_id=str(session_id),
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
                if referenced_docs:
                    for doc in referenced_docs:
                        ref_doc = ReferencedDocument(
                            id=uuid.uuid4(),
                            chat_history_id=chat_history.id,
                            document_name=doc.get("title", "Unknown Document"), 
                            page_number=doc.get("metadata", {}).get("page", 1), 
                            relevance_score=float(doc.get("score", 0.0)),
                            content_snippet=doc.get("content", ""),
                            metadata=doc.get("metadata", {})
                        )
                        session.add(ref_doc)

                await session.commit()
                logger.info(f"Interaction sauvegardée - ID: {chat_history.id}")
                metrics.increment_counter("chat_saves")

        except Exception as e:
            logger.error(f"Erreur sauvegarde interaction: {e}")
            metrics.increment_counter("chat_save_errors")
            
    def _update_conversation_history(
        self,
        current_history: List[Dict],
        query: str,
        response: str
    ) -> List[Dict]:
        """
        Met à jour l'historique de conversation en ajoutant la nouvelle interaction.
        
        Args:
            current_history: Historique actuel
            query: Question de l'utilisateur
            response: Réponse du système
            
        Returns:
            Liste mise à jour avec les nouveaux messages
        """
        # Ajout des nouveaux messages
        updated_history = current_history + [
            {
                "role": "user",
                "content": query,
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
        
        # Garder uniquement les 5 dernières interactions (10 messages)
        max_history = settings.MAX_HISTORY_MESSAGES * 2  # 2 messages par interaction
        return updated_history[-max_history:] if max_history > 0 else updated_history
    
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
        """
        Calcule le score de confiance basé sur les documents.
        Args:
            docs: Liste des documents avec leurs scores
        Returns:
            Score de confiance entre 0 et 1
        """
        if not docs:
            return 0.0
        try:
            # Convertir explicitement tous les scores en float
            scores = [float(doc.get("score", 0.0)) for doc in docs]
            return sum(scores) / len(scores) if scores else 0.0
        except (ValueError, TypeError) as e:
            logger.error(f"Erreur conversion scores: {e}")
            return 0.0