# core/chat/processor.py
from fastapi import HTTPException
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import asyncio
import uuid
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics

from api.models.requests import ChatRequest, ChatContext
from api.models.responses import ChatResponse, DocumentReference, SimilarQuestion
from core.database.models import User, ChatSession, ChatHistory, ReferencedDocument

logger = get_logger("chat_processor")

class ChatProcessor:
    def __init__(self, components):
        self.components = components
        self.confidence_threshold = 0.7
        self.min_relevant_docs = 2

    async def process_message(
        self,
        request: ChatRequest,
        chat_session: Optional[ChatSession] = None
    ) -> ChatResponse:
        """
        Traite un message chat avec une logique améliorée d'interaction.
        """
        start_time = datetime.utcnow()
        try:
            # 1. Génération du vecteur de requête
            try:
                query_vector = await self.components.model.create_embedding(request.query)
                if not isinstance(query_vector, list) or len(query_vector) == 0:
                    logger.error("Échec de la création du vecteur de requête")
                    raise ValueError("Vecteur de requête invalide")
            except Exception as e:
                logger.error(f"Erreur création vecteur requête: {e}")
                raise

            # 2. Recherche des documents pertinents
            try:
                relevant_docs = await self.components.es_client.search_documents(
                    query=request.query,  # Ajout de la requête textuelle
                    vector=query_vector,
                    metadata_filter={"application": request.application} if request.application else None,
                    size=settings.MAX_RELEVANT_DOCS
                )
                
                # Calcul du score de confiance
                context_confidence = self._calculate_context_confidence(relevant_docs)
            except Exception as e:
                logger.error(f"Erreur recherche documents: {e}")
                relevant_docs = []
                context_confidence = 0.0

            # 3. Vérification du contexte de session pour les clarifications
            if chat_session and chat_session.session_context.get("pending_clarification"):
                return await self._handle_clarification_response(
                    request, chat_session, relevant_docs
                )

            # 4. Analyse de contexte
            context_analysis = await self._analyze_context(
                request.query,
                relevant_docs,
                context_confidence
            )

            # 5. Génération de la réponse appropriée
            if context_analysis.get("needs_clarification", False):
                response = await self._create_clarification_response(
                    request, chat_session, context_analysis
                )
            else:
                response = await self._generate_final_response(
                    request,
                    chat_session,
                    relevant_docs,
                    context_analysis,
                    start_time
                )

            # 6. Enrichissement de la réponse avec les vecteurs
            response.query_vector = query_vector
            if response.response:
                try:
                    response.response_vector = await self.components.model.create_embedding(response.response)
                except Exception as e:
                    logger.error(f"Erreur création vecteur réponse: {e}")
                    response.response_vector = [0.0] * len(query_vector)  # Vecteur par défaut

            return response

        except Exception as e:
            logger.error(f"Erreur traitement message: {e}", exc_info=True)
            metrics.increment_counter("chat_errors")
            raise

    async def _find_relevant_documents(
        self,
        query_vector: List[float],
        application: Optional[str]
    ) -> Tuple[List[Dict], float]:
        """
        Recherche les documents pertinents et évalue la confiance du contexte.
        """
        relevant_docs = await self.components.es_client.search_documents(
            query="",
            vector=query_vector,
            metadata_filter={"application": application} if application else None,
            size=settings.MAX_RELEVANT_DOCS
        )

        # Calcul du score de confiance du contexte
        if not relevant_docs:
            return [], 0.0

        scores = [doc.get("score", 0.0) for doc in relevant_docs]
        avg_confidence = sum(scores) / len(scores)

        return relevant_docs, avg_confidence

    async def _analyze_context(
        self,
        query: str,
        relevant_docs: List[Dict],
        context_confidence: float
    ) -> Dict:
        """Analyse le contexte pour déterminer si des clarifications sont nécessaires."""
        try:
            # Analyse du contexte par le summarizer si disponible
            if hasattr(self.components.model, 'summarizer'):
                context_analysis = await self.components.model.summarizer.summarize_documents(relevant_docs)
            else:
                context_analysis = {
                    "needs_clarification": context_confidence < self.confidence_threshold,
                    "themes": [],
                    "questions": []
                }

            # Ajout des critères de clarification
            needs_clarification = (
                context_confidence < self.confidence_threshold or
                len(relevant_docs) < self.min_relevant_docs or
                context_analysis.get("needs_clarification", False)
            )

            return {
                **context_analysis,
                "needs_clarification": needs_clarification,
                "context_confidence": context_confidence
            }

        except Exception as e:
            logger.error(f"Erreur analyse contexte: {e}")
            return {
                "needs_clarification": True,
                "context_confidence": 0.0,
                "error": str(e)
            }

    async def _create_clarification_response(
        self,
        request: ChatRequest,
        chat_session: ChatSession,
        context_analysis: Dict
    ) -> ChatResponse:
        """
        Crée une réponse demandant des clarifications à l'utilisateur.
        """
        # Construction du message de clarification
        if context_analysis["clarification_reason"] == "confidence_low":
            clarification_text = (
                "Je ne suis pas certain de bien comprendre votre demande. "
                "Pourriez-vous préciser :"
            )
        elif context_analysis["clarification_reason"] == "insufficient_context":
            clarification_text = (
                "Je n'ai pas trouvé suffisamment d'informations dans mes documents. "
                "Pourriez-vous reformuler votre question ? "
                "Voici ce que j'ai compris :"
            )
        else:
            clarification_text = (
                "J'ai trouvé plusieurs thèmes possibles dans votre demande. "
                "Pour mieux vous aider, pourriez-vous préciser :"
            )

        # Ajout des questions spécifiques
        questions = context_analysis.get("questions", [])
        if questions:
            clarification_text += f"\n\n{questions[0]}"
            if len(questions) > 1:
                clarification_text += "\n\nAutres points à clarifier :\n" + \
                                    "\n".join(f"- {q}" for q in questions[1:])

        # Mise à jour du contexte de session
        await self.components.session_manager.update_session_context(
            chat_session.session_id,
            {
                "pending_clarification": True,
                "original_query": request.query,
                "context_analysis": context_analysis
            }
        )

        return ChatResponse(
            response=clarification_text,
            session_id=chat_session.session_id,
            conversation_id=uuid.uuid4(),
            metadata={
                "needs_clarification": True,
                "themes": context_analysis.get("themes", []),
                "source": "model",
                "timestamp": datetime.utcnow().isoformat(),
                "session_context": chat_session.session_context
            },
            confidence_score=0.0,
            processing_time=0.0,
            tokens_used=0
        )
        
    def _calculate_context_confidence(self, docs: List[Dict]) -> float:
        """Calcule le score de confiance moyen des documents."""
        if not docs:
            return 0.0
        scores = [doc.get("score", 0.0) for doc in docs]
        return sum(scores) / len(scores)

    async def _handle_clarification_response(
        self,
        request: ChatRequest,
        chat_session: ChatSession,
        relevant_docs: List[Dict]
    ) -> ChatResponse:
        """
        Traite la réponse de l'utilisateur à une demande de clarification.
        """
        original_query = chat_session.session_context.get("original_query", "")
        context_analysis = chat_session.session_context.get("context_analysis", {})

        # Combiner la requête originale avec la clarification
        combined_query = f"{original_query} {request.query}"

        # Génération de la réponse avec le contexte enrichi
        response = await self.components.model.generate_response(
            query=combined_query,
            context_docs=relevant_docs,
            conversation_history=[
                {"role": "user", "content": original_query},
                {"role": "assistant", "content": "Demande de clarification..."},
                {"role": "user", "content": request.query}
            ]
        )

        # Réinitialisation du contexte de session
        await self.components.session_manager.update_session_context(
            chat_session.session_id,
            {"pending_clarification": False}
        )

        return ChatResponse(
            response=response["response"],
            session_id=str(chat_session.session_id),
            confidence_score=response.get("confidence_score", 0.0),
            metadata={"clarification_handled": True}
        )

    async def _generate_final_response(
        self,
        request: ChatRequest,
        chat_session: ChatSession,
        relevant_docs: List[Dict],
        context_analysis: Dict,
        start_time: datetime
    ) -> ChatResponse:
        """
        Génère la réponse finale en utilisant le contexte approprié.
        """
        if not relevant_docs:
            # Aucun document pertinent trouvé
            response = await self.components.model.generate_response(
                query=request.query,
                context_docs=[],
                conversation_history=chat_session.session_context.get("history", [])
            )
            
            response_text = (
                "Je n'ai pas trouvé de documentation spécifique pour votre question, "
                "mais voici ce que je peux vous dire :\n\n" + 
                response["response"]
            )
        else:
            # Génération avec documents pertinents
            response = await self.components.model.generate_response(
                query=request.query,
                context_docs=relevant_docs,
                context_summary=context_analysis.get("structured_summary"),
                conversation_history=chat_session.session_context.get("history", [])
            )
            response_text = response["response"]

        # Construction de la réponse finale
        return ChatResponse(
            response=response_text,
            session_id=str(chat_session.session_id),
            documents=[DocumentReference(**doc) for doc in relevant_docs],
            confidence_score=response.get("confidence_score", 0.0),
            processing_time=(datetime.utcnow() - start_time).total_seconds(),
            tokens_used=response.get("tokens_used", {}).get("total", 0),
            metadata={
                "has_context": bool(relevant_docs),
                "context_analysis": context_analysis
            }
        )
