# api/routes/chat_routes.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Request
from fastapi.responses import StreamingResponse
from typing import Dict, Optional, List, Any
from datetime import datetime
import uuid
import asyncio
from sse_starlette.sse import EventSourceResponse
import json
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..models.requests import ChatRequest
from ..models.responses import ChatResponse, ErrorResponse, VectorStats
from ..dependencies import get_components
from core.utils.logger import get_logger
from core.database.base import get_session_manager
from core.utils.metrics import metrics
from core.database.models import User, ChatSession, ChatHistory
from core.database.base import DatabaseSession
from core.config import settings

logger = get_logger("chat_routes")
router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def process_chat_message(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    components=Depends(get_components)
) -> ChatResponse:
    """
    Traite une requête de chat et retourne une réponse complète.
    """
    start_time = datetime.utcnow()
    session_manager = get_session_manager()

    try:
        with metrics.timer("chat_processing"):
            # 1. Vérification et récupération de l'utilisateur
            async with DatabaseSession() as session:
                user = await session.execute(
                    select(User).where(User.id == request.user_id)
                )
                user = user.scalar_one_or_none()
                if not user:
                    raise HTTPException(
                        status_code=404,
                        detail="Utilisateur non trouvé"
                    )

                # 2. Gestion de la session
                chat_session = await get_or_create_session(
                    session,
                    request.user_id,
                    request.session_id,
                    request.metadata
                )

            # 3. Préparation du contexte
            query_vector = await components.model.create_embedding(request.query)
            context = await prepare_chat_context(components, request, chat_session)

            # 4. Recherche de questions similaires
            similar_questions = await find_similar_questions(
                components,
                query_vector,
                request.metadata
            )

            # 5. Génération de la réponse
            response_data = await generate_response(
                components,
                request,
                context,
                similar_questions
            )

            # 6. Sauvegarde en arrière-plan
            background_tasks.add_task(
                save_chat_interaction,
                components,
                request,
                response_data,
                query_vector,
                start_time
            )

            # 7. Retour de la réponse
            return format_chat_response(response_data, chat_session, start_time)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur traitement message: {e}", exc_info=True)
        metrics.increment_counter("chat_errors")
        await log_error(components, e, request)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail="Une erreur est survenue lors du traitement",
                error_code="PROCESSING_ERROR",
                path="/api/chat",
                metadata={"error": str(e)}
            ).model_dump()
        )

@router.get("/stream")
async def stream_chat_response(
    query: str = Query(..., min_length=1),
    user_id: str = Query(...),
    session_id: Optional[str] = None,
    language: str = "fr",
    components=Depends(get_components)
):
    """
    Stream une réponse de chat en utilisant Server-Sent Events.
    """
    async def event_generator():
        try:
            # Notification de début
            yield {
                "event": "start",
                "data": {"status": "started", "timestamp": datetime.utcnow().isoformat()}
            }

            # Préparation du contexte
            query_vector = await components.model.create_embedding(query)
            relevant_docs = await components.es_client.search_documents(
                query=query,
                vector=query_vector,
                size=settings.MAX_RELEVANT_DOCS
            )

            # Streaming de la réponse
            async for token in components.model.generate_streaming_response(
                query=query,
                context_docs=relevant_docs,
                language=language
            ):
                yield {
                    "event": "token",
                    "data": {"content": token, "type": "token"}
                }

            # Métadonnées finales
            yield {
                "event": "complete",
                "data": {
                    "status": "completed",
                    "metadata": {
                        "docs_used": len(relevant_docs),
                        "processing_time": metrics.get_timer_value("chat_processing")
                    }
                }
            }

        except Exception as e:
            logger.error(f"Erreur streaming: {e}")
            yield {
                "event": "error",
                "data": {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
            }

    return EventSourceResponse(event_generator())

@router.get("/similar", response_model=List[Dict[str, Any]])
async def get_similar_questions(
    query: str = Query(..., min_length=1),
    threshold: float = Query(0.8, ge=0.0, le=1.0),
    limit: int = Query(5, ge=1, le=20),
    components=Depends(get_components)
) -> List[Dict]:
    """
    Trouve des questions similaires basées sur la similarité vectorielle.
    """
    try:
        query_vector = await components.model.create_embedding(query)
        similar = await components.db.find_similar_questions(
            vector=query_vector,
            threshold=threshold,
            limit=limit
        )
        return similar
    except Exception as e:
        logger.error(f"Erreur recherche questions similaires: {e}")
        metrics.increment_counter("similar_search_errors")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = Query(50, ge=1, le=100),
    components=Depends(get_components)
) -> List[Dict]:
    """
    Récupère l'historique des conversations pour une session.
    """
    try:
        history = await components.db.get_session_history(
            session_id=session_id,
            limit=limit
        )
        return history
    except Exception as e:
        logger.error(f"Erreur récupération historique: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/vectors", response_model=VectorStats)
async def get_vector_statistics(components=Depends(get_components)) -> Dict:
    """
    Récupère les statistiques d'utilisation des vecteurs.
    """
    try:
        stats = await components.db.get_vector_statistics()
        return VectorStats(**stats)
    except Exception as e:
        logger.error(f"Erreur récupération statistiques vecteurs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Fonctions utilitaires

async def get_or_create_session(session, user_id: str, session_id: Optional[str], metadata: Dict) -> ChatSession:
    """Récupère ou crée une session de chat."""
    if session_id:
        result = await session.execute(
            select(ChatSession)
            .where(ChatSession.session_id == session_id)
            .options(selectinload(ChatSession.chat_history))
        )
        existing_session = result.scalar_one_or_none()
        if existing_session:
            return existing_session

    # Création d'une nouvelle session
    new_session = ChatSession(
        user_id=user_id,
        session_id=str(uuid.uuid4()),
        session_context={"created_at": datetime.utcnow().isoformat()},
        metadata=metadata
    )
    session.add(new_session)
    await session.flush()
    return new_session

async def prepare_chat_context(components, request: ChatRequest, session: ChatSession) -> Dict:
    """Prépare le contexte pour la génération de réponse."""
    return {
        "history": session.session_context.get("history", []),
        "metadata": request.metadata,
        "preferences": request.context.preferences if request.context else {},
        "session_id": session.session_id
    }

async def find_similar_questions(components, query_vector: List[float], metadata: Dict) -> List[Dict]:
    """Recherche des questions similaires."""
    return await components.db.find_similar_questions(
        vector=query_vector,
        threshold=settings.CONTEXT_CONFIDENCE_THRESHOLD,
        limit=3,
        metadata=metadata
    )

async def generate_response(components, request: ChatRequest, context: Dict, similar_questions: List[Dict]) -> Dict:
    """Génère une réponse basée sur le contexte et les questions similaires."""
    # Si une question très similaire existe
    if similar_questions and similar_questions[0]["similarity"] > 0.98:
        return {
            "response": similar_questions[0]["response"],
            "source": "cache",
            "confidence": similar_questions[0]["similarity"]
        }

    # Sinon, génération d'une nouvelle réponse
    relevant_docs = await components.es_client.search_documents(
        query=request.query,
        metadata={"application": request.application} if request.application else None
    )

    response = await components.model.generate_response(
        query=request.query,
        context_docs=relevant_docs,
        conversation_history=context["history"],
        language=request.language
    )

    return {
        "response": response,
        "source": "generation",
        "confidence": max((doc.get("score", 0) for doc in relevant_docs), default=0.0),
        "documents": relevant_docs
    }

async def save_chat_interaction(components, request: ChatRequest, response_data: Dict, query_vector: List[float], start_time: datetime):
    """Sauvegarde l'interaction dans la base de données."""
    try:
        response_vector = await components.model.create_embedding(response_data["response"])
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        await components.db.save_chat_interaction(
            session_id=request.session_id,
            user_id=request.user_id,
            query=request.query,
            response=response_data["response"],
            query_vector=query_vector,
            response_vector=response_vector,
            confidence_score=response_data["confidence"],
            processing_time=processing_time,
            metadata={
                "source": response_data["source"],
                "application": request.application,
                "language": request.language
            }
        )
    except Exception as e:
        logger.error(f"Erreur sauvegarde interaction: {e}")
        metrics.increment_counter("save_errors")

async def log_error(components, error: Exception, request: ChatRequest):
    """Log une erreur dans la base de données."""
    try:
        await components.db.log_error(
            error_type=type(error).__name__,
            error_message=str(error),
            endpoint="/api/chat",
            user_id=request.user_id,
            metadata={
                "query": request.query,
                "application": request.application
            }
        )
    except Exception as e:
        logger.error(f"Erreur lors du log d'erreur: {e}")

def format_chat_response(response_data: Dict, session: ChatSession, start_time: datetime) -> ChatResponse:
    """Formate la réponse finale."""
    processing_time = (datetime.utcnow() - start_time).total_seconds()
    
    return ChatResponse(
        response=response_data["response"],
        session_id=session.session_id,
        conversation_id=uuid.uuid4(),
        documents=response_data.get("documents", []),
        confidence_score=response_data["confidence"],
        processing_time=processing_time,
        metadata={
            "source": response_data["source"],
            "session_context": session.session_context
        }
    )
