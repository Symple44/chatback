# core/chat/processor.py
from fastapi import HTTPException
from datetime import datetime
from typing import Dict, List, Optional, Set
import asyncio
import uuid
import re
import traceback
from pathlib import Path
from core.utils.system_optimizer import SystemOptimizer
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.chat.context_analyzer import context_analyzer

from api.models.requests import ChatRequest
from api.models.responses import ChatResponse, DocumentReference, SimilarQuestion
from core.database.models import User, ChatSession, ChatHistory, ReferencedDocument, ErrorLog

logger = get_logger("chat_processor")

class ChatProcessor:
    def __init__(self, components):
        """Initialise le processeur de chat."""
        self.components = components
        self.confidence_threshold = 0.7
        self.min_relevant_docs = 2
        self.max_themes = 2
        self.max_clarification_attempts = 3
        self.min_query_length = 3

    async def process_message(
        self,
        request: ChatRequest,
        chat_session: Optional[ChatSession] = None
    ) -> ChatResponse:
        """
        Traite un message chat avec une logique améliorée d'interaction.
        """
        start_time = datetime.utcnow()
        metrics.increment_counter("chat_requests")

        try:
            # 1. Génération du vecteur de requête
            query_vector = await self.components.model.create_embedding(request.query)
            if not isinstance(query_vector, list) or len(query_vector) == 0:
                raise ValueError("Vecteur de requête invalide")

            # 2. Recherche des documents pertinents
            relevant_docs = await self.components.es_client.search_documents(
                query=request.query,
                vector=query_vector,
                metadata_filter={"application": request.application} if request.application else None,
                size=settings.MAX_RELEVANT_DOCS
            )

            # 3. Analyse du contexte actuel
            context_analysis = await self._analyze_context(
                query=request.query,
                relevant_docs=relevant_docs,
                context_confidence=self._calculate_context_confidence(relevant_docs)
            )

            # 4. Vérification du contexte de session pour les clarifications en cours
            if chat_session and chat_session.session_context.get("pending_clarification"):
                if await self._should_handle_clarification(request, chat_session):
                    return await self._handle_clarification_response(
                        request, chat_session, context_analysis
                    )

            # 5. Décision: clarification ou réponse directe
            if context_analysis.get("needs_clarification"):
                if await self._should_ask_clarification(chat_session):
                    return await self._create_clarification_response(
                        request, chat_session, context_analysis
                    )

            # 6. Génération de la réponse finale
            response = await self._generate_final_response(
                request=request,
                chat_session=chat_session,
                relevant_docs=relevant_docs,
                context_analysis=context_analysis,
                start_time=start_time
            )

            # 7. Mise à jour du contexte de session
            if chat_session:
                await self._update_session_context(
                    chat_session=chat_session,
                    request=request,
                    response=response,
                    context_analysis=context_analysis
                )
            
            await self._cleanup_session_context(chat_session)

            return response

        except Exception as e:
            logger.error(f"Erreur traitement message: {e}", exc_info=True)
            metrics.increment_counter("chat_errors")
            await self._log_error(request, e)
            raise HTTPException(
                status_code=500,
                detail="Une erreur est survenue lors du traitement de votre message."
            )

    async def _analyze_context(
        self,
        query: str,
        relevant_docs: List[Dict],
        context_confidence: float
    ) -> Dict:
        """Analyse le contexte de la conversation."""
        from core.utils.context_analyzer import context_analyzer
        try:
            context_analysis = await context_analyzer.analyze_context(
                query=query,
                context_docs=relevant_docs,
                context_confidence=context_confidence,
                summarizer=self.components.model.summarizer if hasattr(self.components.model, 'summarizer') else None
            )
            
            return context_analysis

        except Exception as e:
            logger.error(f"Erreur analyse contexte: {e}")
            return {
                "needs_clarification": True,
                "context_confidence": 0.0,
                "error": str(e)
            }

    async def _should_handle_clarification(
        self,
        request: ChatRequest,
        chat_session: ChatSession
    ) -> bool:
        """Détermine si une réponse de clarification doit être traitée."""
        context = chat_session.session_context
        if not context.get("pending_clarification"):
            return False

        original_query = context.get("original_query")
        if not original_query:
            return False

        # Analyse de la réponse à la clarification
        response_analysis = await self._analyze_clarification_response(
            original_query=original_query,
            clarification_response=request.query,
            context=context
        )

        return response_analysis.get("is_valid", False)

    async def _analyze_clarification_response(
        self,
        original_query: str,
        clarification_response: str,
        context: Dict
    ) -> Dict:
        """Analyse la réponse à une demande de clarification."""
        try:
            return {
                "is_valid": True if clarification_response else False,
                "adds_context": len(clarification_response.split()) > len(original_query.split()),
                "references_original": any(word in clarification_response.lower() 
                                        for word in original_query.lower().split()),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Erreur analyse clarification: {e}")
            return {"is_valid": False}
    

    async def _create_clarification_response(
        self,
        request: ChatRequest,
        chat_session: ChatSession,
        context_analysis: Dict
    ) -> ChatResponse:
        """Crée une réponse demandant des clarifications."""
        reason = context_analysis.get("clarification_reason", "general")
        templates = {
            "multiple_themes": (
                "Votre question touche plusieurs thèmes:\n"
                "{themes}\n\n"
                "Sur quel aspect spécifique souhaitez-vous des informations ?"
            ),
            "low_confidence": (
                "Je ne suis pas certain de bien comprendre votre demande.\n"
                "Voici ce que j'ai compris : {understanding}\n"
                "Pourriez-vous confirmer ou reformuler ?"
            ),
            "insufficient_context": (
                "Je n'ai pas trouvé suffisamment d'informations précises sur ce sujet.\n"
                "Pourriez-vous :\n"
                "1. Être plus spécifique sur {key_concepts}\n"
                "2. Donner un exemple concret\n"
                "3. Préciser le contexte d'utilisation"
            ),
            "query_ambiguity": (
                "Votre question pourrait avoir plusieurs interprétations. "
                "Voulez-vous dire :\n"
                "{interpretations}"
            )
        }

        # Construction du message de clarification
        clarification_text = templates.get(reason, templates["query_ambiguity"]).format(
            themes="\n".join(f"- {theme}" for theme in context_analysis.get("themes", [])),
            understanding=await self._generate_query_understanding(request.query),
            key_concepts=", ".join(context_analysis["query_analysis"]["key_concepts"]),
            interpretations="\n".join(
                f"- {interp}" for interp in await self._generate_query_interpretations(request.query)
            )
        )

        # Mise à jour du contexte de session
        if chat_session:
            await self._update_session_context(
                chat_session=chat_session,
                request=request,
                context_update={
                    "pending_clarification": True,
                    "original_query": request.query,
                    "clarification_reason": reason,
                    "clarification_attempt": chat_session.session_context.get("clarification_attempt", 0) + 1
                }
            )

        return ChatResponse(
            response=clarification_text,
            session_id=str(chat_session.session_id) if chat_session else None,
            conversation_id=str(uuid.uuid4()),
            confidence_score=context_analysis.get("context_confidence", 0.0),
            metadata={
                "needs_clarification": True,
                "clarification_reason": reason,
                "themes": context_analysis.get("themes", []),
                "query_analysis": context_analysis.get("query_analysis", {})
            }
        )

    async def _generate_final_response(
        self,
        request: ChatRequest,
        chat_session: ChatSession,
        relevant_docs: List[Dict],
        context_analysis: Dict,
        start_time: datetime
    ) -> ChatResponse:
        """Génère la réponse finale."""
        try:
            # Sélection du type de réponse et configuration
            response_type = context_analysis.get("response_type", "comprehensive")
            response_config = settings.RESPONSE_TYPES.get(response_type, settings.RESPONSE_TYPES["comprehensive"])

            # Construction du prompt avec style adapté
            prompt_prefix = response_config.get("style", "")

            # Génération de la réponse
            response = await self.components.model.generate_response(
                query=request.query,
                context_docs=relevant_docs,
                conversation_history=chat_session.session_context.get("history", []) if chat_session else [],
                response_type=response_type,
                prompt_prefix=prompt_prefix,
                generation_config={
                    "temperature": response_config.get("temperature", 0.7),
                    "max_tokens": response_config.get("max_tokens", 1024)
                }
            )

            # Construction de la réponse finale
            final_response = ChatResponse(
                response=response.get("response", ""),
                session_id=str(chat_session.session_id) if chat_session else None,
                conversation_id=str(uuid.uuid4()),
                documents=[
                    DocumentReference(
                        title=doc.get("title", ""),
                        content=doc.get("content", ""),
                        page=doc.get("page", 1),
                        score=float(doc.get("score", 0.0)),
                        metadata=doc.get("metadata", {})
                    ) for doc in relevant_docs
                ],
                confidence_score=context_analysis.get("context_confidence", 0.0),
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                tokens_used=response.get("tokens_used", {}).get("total", 0),
                tokens_details=response.get("tokens_used", {}),
                metadata={
                    "response_type": response_type,
                    "themes": context_analysis.get("themes", []),
                    "context_analysis": context_analysis,
                    "source": "model",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            if chat_session:
                await self._cleanup_session_context(chat_session)

            # Mise à jour des métriques
            metrics.increment_counter("responses_generated")
            metrics.record_time(
                "response_generation",
                (datetime.utcnow() - start_time).total_seconds()
            )

            return final_response

        except Exception as e:
            logger.error(f"Erreur génération réponse finale: {e}", exc_info=True)
            raise

    async def _update_session_context(
        self,
        chat_session: ChatSession,
        request: ChatRequest,
        context_update: Optional[Dict] = None,
        response: Optional[ChatResponse] = None,
        context_analysis: Optional[Dict] = None
    ) -> None:
        """Met à jour le contexte de la session."""
        try:
            current_context = chat_session.session_context or {}

            # Mise à jour de l'historique
            history = current_context.get("history", [])
            if request and response:
                history.append({
                    "role": "user",
                    "content": request.query,
                    "timestamp": datetime.utcnow().isoformat()
                })
                history.append({
                    "role": "assistant",
                    "content": response.response,
                    "timestamp": datetime.utcnow().isoformat()
                })
                # Limiter l'historique aux 5 dernières interactions
                history = history[-10:]

            # Construction du nouveau contexte
            new_context = {
                **current_context,
                "history": history,
                "last_interaction": datetime.utcnow().isoformat(),
                "last_query": request.query if request else None,
                **(context_update or {}),
            }

            # Ajout des informations d'analyse si présentes
            if context_analysis:
                new_context["analysis"] = {
                    "themes": context_analysis.get("themes", []),
                    "confidence": context_analysis.get("context_confidence"),
                    "response_type": context_analysis.get("response_type"),
                }

            # Mise à jour de la session
            chat_session.session_context = new_context
            chat_session.updated_at = datetime.utcnow()

            # Persistance des changements
            async with self.components.db.session_factory() as session:
                session.add(chat_session)
                await session.commit()

        except Exception as e:
            logger.error(f"Erreur mise à jour contexte session: {e}")
            raise

    async def _generate_query_understanding(self, query: str) -> str:
        """Génère une reformulation de la compréhension de la requête."""
        try:
            # Utilisation du modèle pour reformuler la question
            response = await self.components.model.generate_response(
                query=f"Reformule cette question pour vérifier la compréhension: {query}",
                response_type="concise",  # Utilisation du type de réponse au lieu de max_tokens
                language="fr"
            )
            return response.get("response", "")
        except Exception as e:
            logger.error(f"Erreur génération reformulation: {e}")
            return f"Vous demandez des informations sur {query}"

    async def _generate_query_interpretations(self, query: str) -> List[str]:
        """Génère différentes interprétations possibles de la requête."""
        try:
            response = await self.components.model.generate_response(
                query=f"Génère 3 interprétations possibles de cette question: {query}",
                max_tokens=150,
                temperature=0.4
            )
            
            interpretations = response.get("response", "").split("\n")
            return [i.strip() for i in interpretations if i.strip()][:3]
        except Exception as e:
            logger.error(f"Erreur génération interprétations: {e}")
            return []

    def _calculate_context_confidence(self, docs: List[Dict]) -> float:
        """Calcule le score de confiance moyen des documents."""
        if not docs:
            return 0.0
        scores = [doc.get("score", 0.0) for doc in docs]
        return sum(scores) / len(scores)

    async def _should_ask_clarification(self, chat_session: Optional[ChatSession]) -> bool:
        """Détermine si une clarification doit être demandée."""
        if not chat_session:
            return True

        context = chat_session.session_context
        clarification_attempts = context.get("clarification_attempt", 0)
        
        # Éviter les boucles de clarification
        if clarification_attempts >= self.max_clarification_attempts:
            return False

        # Vérifier le temps depuis la dernière clarification
        last_clarification = context.get("last_clarification")
        if last_clarification:
            last_time = datetime.fromisoformat(last_clarification)
            if (datetime.utcnow() - last_time).seconds < 300:  # 5 minutes
                return False

        return True
        
    async def _cleanup_session_context(
        self,
        chat_session: ChatSession
    ) -> None:
        """Nettoie le contexte de session des données temporaires."""
        if not chat_session or not chat_session.session_context:
            return

        context = chat_session.session_context
        # Supprimer les flags temporaires
        context.pop("pending_clarification", None)
        context.pop("clarification_attempt", None)
        context.pop("last_clarification", None)
        
        # Mise à jour de la session
        chat_session.session_context = context
    
    async def cleanup(self):
        """Nettoie les ressources du processeur."""
        try:
            # Nettoyage des composants si nécessaire
            if hasattr(self.components, 'cleanup'):
                await self.components.cleanup()
        except Exception as e:
            logger.error(f"Erreur nettoyage ressources: {e}")

    async def _log_error(self, request: ChatRequest, error: Exception) -> None:
        """Log une erreur dans la base de données."""
        try:
            error_data = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "query": request.query,
                "user_id": request.user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "application": request.application,
                    "session_id": request.session_id
                }
            }
            
            async with self.components.db.session_factory() as session:
                error_log = ErrorLog(**error_data)
                session.add(error_log)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Erreur lors du log d'erreur: {e}")