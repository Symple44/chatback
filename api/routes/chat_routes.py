# api/routes/chat_routes.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Request
from fastapi.responses import StreamingResponse
from typing import Dict, Optional, List
from datetime import datetime
import uuid
import asyncio
from sse_starlette.sse import EventSourceResponse

from ..models.requests import ChatRequest
from ..models.responses import ChatResponse, ErrorResponse
from ..dependencies import get_components
from core.utils.logger import get_logger
from core.database.base import get_session_manager
from core.utils.metrics import metrics

logger = get_logger("chat_routes")
router = APIRouter(prefix="/chat", tags=["chat"])

# api/routes/chat_routes.py

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
        # 1. Vérification et récupération de l'utilisateur
        async with session_manager.get_session() as session:
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
            session_id = request.session_id or str(uuid.uuid4())
            chat_session = await session.execute(
                select(ChatSession).where(
                    ChatSession.session_id == session_id
                )
            )
            chat_session = chat_session.scalar_one_or_none()

            if not chat_session:
                chat_session = ChatSession(
                    user_id=str(request.user_id),
                    session_id=session_id,
                    session_context=request.context.model_dump() if request.context else {}
                )
                session.add(chat_session)
                await session.flush()

        # 3. Préparation du contexte de recherche
        query_vector = await components.model.create_embedding(request.query)
        
        # 4. Recherche de questions similaires
        similar_questions = await components.db.find_similar_questions(
            vector=query_vector,
            threshold=0.95,
            limit=3
        )

        # 5. Traitement de la réponse
        if similar_questions and similar_questions[0]['similarity'] > 0.98:
            # Utilisation d'une réponse existante très similaire
            response = similar_questions[0]['response']
            confidence_score = similar_questions[0]['similarity']
            metrics.increment_counter("cache_hits")
            relevant_docs = []
        else:
            # Recherche de documents pertinents
            relevant_docs = await components.es_client.search_documents(
                query=request.query,
                vector=query_vector,
                metadata={"application": request.application} if request.application else None,
                size=settings.MAX_RELEVANT_DOCS
            )

            # Génération de la réponse
            response = await components.model.generate_response(
                query=request.query,
                context_docs=relevant_docs,
                conversation_history=request.context.history if request.context else [],
                language=request.language
            )
            confidence_score = max((doc.get('score', 0) for doc in relevant_docs), default=0.0)
            metrics.increment_counter("new_responses")

        # 6. Création du vecteur de réponse
        response_vector = await components.model.create_embedding(response)
        
        # 7. Calcul du temps de traitement
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # 8. Préparation des documents référencés
        documents_used = [
            {
                "title": doc.get('title', 'Unknown'),
                "page": doc.get('metadata', {}).get('page', 'N/A'),
                "score": doc.get('score', 0.0),
                "content": doc.get('content', '')[:500]  # Limiter la taille du contenu
            }
            for doc in relevant_docs[:3]  # Limiter à 3 documents
        ] if relevant_docs else []

        # 9. Sauvegarde en arrière-plan
        background_tasks.add_task(
            components.db.save_chat_interaction,
            session_id=session_id,
            user_id=str(request.user_id),
            query=request.query,
            response=response,
            query_vector=query_vector,
            response_vector=response_vector,
            confidence_score=confidence_score,
            tokens_used=len(request.query.split()) + len(response.split()),
            processing_time=processing_time,
            referenced_docs=documents_used,
            additional_data={
                "application": request.application,
                "language": request.language,
                "similar_questions_found": len(similar_questions)
            }
        )

        # 10. Construction et renvoi de la réponse
        return ChatResponse(
            response=response,
            session_id=session_id,
            conversation_id=str(uuid.uuid4()),
            documents_used=documents_used,
            confidence_score=confidence_score,
            processing_time=processing_time,
            application=request.application,
            similar_questions=[{
                "query": q["query"],
                "response": q["response"],
                "similarity": q["similarity"],
                "timestamp": q["created_at"]
            } for q in similar_questions],
            metadata={
                "model_version": settings.VERSION,
                "tokens_used": len(response.split()),
                "source": "cache" if similar_questions and similar_questions[0]['similarity'] > 0.98 else "generation"
            },
            timestamp=datetime.utcnow()
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur traitement message: {e}", exc_info=True)
        metrics.increment_counter("chat_errors")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Une erreur est survenue lors du traitement",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
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
            # Préparation comme pour la requête normale
            query_vector = await components.model.create_embedding(query)
            relevant_docs = await components.es_client.search_documents(
                query=query,
                vector=query_vector
            )

            # Streaming de la réponse
            async for token in components.model.generate_streaming_response(
                query=query,
                context_docs=relevant_docs,
                language=language
            ):
                yield {
                    "event": "message",
                    "data": {"token": token, "type": "token"}
                }

            # Envoi des métadonnées à la fin
            yield {
                "event": "metadata",
                "data": {
                    "documents_used": len(relevant_docs),
                    "processing_time": time.time() - start_time
                }
            }

        except Exception as e:
            logger.error(f"Erreur streaming: {e}")
            yield {
                "event": "error",
                "data": {"error": str(e)}
            }

    return EventSourceResponse(event_generator())

@router.get("/similar")
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

async def save_interaction(
    components,
    request: ChatRequest,
    response: str,
    query_vector: List[float],
    response_vector: List[float],
    processing_time: float
):
    """
    Sauvegarde une interaction en arrière-plan.
    """
    try:
        async with get_session_manager().get_session() as session:
            await components.db.save_chat_interaction(
                session_id=request.session_id or str(uuid.uuid4()),
                user_id=str(request.user_id),
                query=request.query,
                response=response,
                query_vector=query_vector,
                response_vector=response_vector,
                confidence_score=0.0,  # À calculer selon le contexte
                tokens_used=len(request.query.split()) + len(response.split()),
                processing_time=processing_time
            )
        logger.info("Interaction sauvegardée avec succès")
    except Exception as e:
        logger.error(f"Erreur sauvegarde interaction: {e}")
        metrics.increment_counter("save_errors")
