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
from ..models.responses import ChatResponse, ErrorResponse, VectorStats, DocumentReference, SimilarQuestion
from ..dependencies import get_components
from core.utils.logger import get_logger
from core.database.base import get_session_manager
from core.utils.metrics import metrics
from core.database.models import User, ChatSession, ChatHistory, ReferencedDocument
from core.database.base import DatabaseSession
from core.database.manager import DatabaseManager
from core.config.config import settings
from core.chat.processor_factory import ProcessorFactory
from core.search.strategies import SearchMethod

#A supprimer
from core.chat.processor import ChatProcessor

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
    
    Args:
        request: Requête de chat
        vector_search: Active/désactive la recherche vectorielle
        background_tasks: Tâches d'arrière-plan
        components: Composants de l'application
        
    Returns:
        Réponse du chat
    """
    try:
        # 1. Vérification de l'utilisateur
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

        # 2. Récupération ou création de la session
        chat_session = await components.session_manager.get_or_create_session(
            str(request.session_id) if request.session_id else None,
            str(request.user_id),
            request.metadata
        )
        
        # Préparation du contexte avec paramètres de recherche
        session_context = {
            "history": chat_session.session_context.get("history", []),
            "metadata": request.metadata,
            "session_id": chat_session.session_id,
            "search_config": {
                "method": request.search_method,
                "params": request.search_params
            }
        }
        
        # Obtention du processeur approprié
        processor = await ProcessorFactory.get_processor(
            business_type=request.business,
            components=components
        )
        
        # Configuration de la recherche
        if hasattr(processor, 'search_manager'):
            processor.search_manager.configure(
                method=request.search_method,
                search_params=request.search_params,
                metadata_filter={"application": request.application} if request.application else None
            )

        # Traitement du message
        start_time = datetime.utcnow()
        response = await processor.process_message(
            request={
                **request.dict(exclude_unset=True),
                "session_id": chat_session.session_id,
                "context": session_context
            }
        )

        # Mise à jour des métriques
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        metrics.record_time("chat_processing", processing_time)

        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur traitement message: {e}", exc_info=True)
        metrics.increment_counter("chat_errors")
        await log_error(components, e, request)
        raise HTTPException(
            status_code=500,
            detail="Une erreur interne est survenue."
        )

@router.get("/stream")
async def stream_chat_response(
    query: str = Query(..., min_length=1),
    user_id: str = Query(...),
    search_method: SearchMethod = Query(
        default=SearchMethod.RAG,
        description="Méthode de recherche"
    ),
    search_params: Dict[str, Any] = Query(
        default={
            "max_docs": settings.MAX_RELEVANT_DOCS,
            "min_score": settings.CONFIDENCE_THRESHOLD
        },
        description="Paramètres de recherche"
    ),
    session_id: Optional[str] = None,
    language: str = "fr",
    application: Optional[str] = None,
    components=Depends(get_components)
):
    """
    Stream une réponse de chat avec stratégie de recherche configurable.
    
    Args:
        query: Question de l'utilisateur
        user_id: ID de l'utilisateur
        search_method: Méthode de recherche
        search_params: Paramètres de recherche
        session_id: ID de session optionnel
        language: Code de langue
        application: Application source
        components: Composants de l'application
    """
    async def event_generator():
        try:
            # Création de la requête
            chat_request = ChatRequest(
                query=query,
                user_id=user_id,
                session_id=session_id,
                search_method=search_method,
                search_params=search_params,
                application=application,
                language=language
            )

            # Récupération de la session
            chat_session = await components.session_manager.get_or_create_session(
                str(session_id) if session_id else None,
                str(user_id),
                metadata={
                    "search_method": search_method,
                    "application": application
                }
            )

            # Configuration du processeur
            processor = await ProcessorFactory.get_processor(
                business_type="generic",  # Toujours générique pour le streaming
                components=components
            )

            # Configuration de la recherche
            processor.search_manager.configure(
                method=search_method,
                search_params=search_params,
                metadata_filter={"application": application} if application else None
            )

            # Recherche du contexte
            relevant_docs = []
            if search_method != SearchMethod.DISABLED:
                relevant_docs = await processor.search_manager.search_context(
                    query=query,
                    metadata_filter={"application": application} if application else None,
                    **search_params
                )

            # Streaming de la réponse
            async for token in components.model.generate_streaming_response(
                query=query,
                context_docs=relevant_docs,
                language=language
            ):
                yield {
                    "event": "token",
                    "data": json.dumps({
                        "content": token,
                        "type": "token"
                    })
                }

            # Métadonnées finales
            yield {
                "event": "complete",
                "data": json.dumps({
                    "status": "completed",
                    "metadata": {
                        "search_method": search_method,
                        "docs_used": len(relevant_docs),
                        "search_params": search_params,
                        "processing_time": metrics.get_timer_value("chat_processing")
                    }
                })
            }

        except Exception as e:
            logger.error(f"Erreur streaming: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
            }

    return EventSourceResponse(event_generator())

@router.get("/search-methods")
async def get_search_methods():
    """Retourne les méthodes de recherche disponibles."""
    return {
        "methods": [method.value for method in SearchMethod],
        "default": SearchMethod.RAG.value,
        "configurations": {
            method.value: SEARCH_STRATEGIES_CONFIG.get(method.value, {}).get("search_params", {})
            for method in SearchMethod
        }
    }

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
        similar = await components.db_manager.find_similar_questions(
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
        history = await components.session_manager.get_session_history(session_id, limit)
        return [h._asdict() for h in history]
    except Exception as e:
        logger.error(f"Erreur récupération historique: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/vectors", response_model=VectorStats)
async def get_vector_statistics(components=Depends(get_components)) -> Dict:
    """
    Récupère les statistiques d'utilisation des vecteurs.
    """
    try:
        stats = await components.es_client.get_vector_statistics()
        return VectorStats(**stats)
    except Exception as e:
        logger.error(f"Erreur récupération statistiques vecteurs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Fonctions utilitaires

async def prepare_chat_context(components, request: ChatRequest, session: ChatSession) -> Dict:
    """Prépare le contexte pour la génération de réponse."""
    context = {
        "history": session.session_context.get("history", []),
        "metadata": request.metadata,
        "preferences": request.context.preferences if request.context else {},
        "session_id": session.session_id
    }
    
    # Ajout du contexte initial si présent
    if request.context and request.context.source_documents:
        context["source_documents"] = request.context.source_documents
        
    return context

async def find_similar_questions(components, query_vector: List[float], metadata: Dict) -> List[Dict]:
    """Recherche des questions similaires."""
    try:
        if hasattr(components.db_manager, "find_similar_questions"):
            return await components.db_manager.find_similar_questions(
                vector=query_vector,
                threshold=settings.CONTEXT_CONFIDENCE_THRESHOLD,
                limit=3,
                metadata=metadata
            )
        logger.warning("Méthode find_similar_questions non disponible")
        return []
    except Exception as e:
        logger.error(f"Erreur recherche similarité: {e}")
        return []

async def generate_response(components, request: ChatRequest, context: Dict, similar_questions: List[Dict]) -> Dict:
    """Génère une réponse basée sur le contexte et les questions similaires."""
    try:
        # Si une question très similaire existe
        if similar_questions and similar_questions[0]["similarity"] > 0.98:
            return {
                "response_text": similar_questions[0]["response"],
                "source": "cache",
                "confidence": similar_questions[0]["similarity"]
            }

        # Sinon, génération d'une nouvelle réponse
        relevant_docs = await components.es_client.search_documents(
            query=request.query,
            metadata={"application": request.application} if request.application else None
        )

        model_response = await components.model.generate_response(
            query=request.query,
            context_docs=relevant_docs,
            conversation_history=context["history"],
            language=request.language
        )

        # S'assurer que model_response est un string
        if isinstance(model_response, dict):
            response_text = model_response.get('response', '')
        else:
            response_text = str(model_response)

        return {
            "response_text": response_text,
            "source": "generation",
            "confidence": max((doc.get("score", 0) for doc in relevant_docs), default=0.0),
            "documents": relevant_docs,
            "tokens_used": model_response.get("tokens_used", 0) if isinstance(model_response, dict) else 0
        }
    except Exception as e:
        logger.error(f"Erreur génération réponse: {e}")
        return {
            "response_text": "Désolé, une erreur est survenue lors de la génération de la réponse.",
            "source": "error",
            "confidence": 0.0,
            "documents": []
        }

def format_chat_response(response_data: Dict, chat_session: ChatSession, start_time: datetime) -> ChatResponse:
    try:
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        documents = [DocumentReference(
            title=doc.get("title", "Unknown"),
            page=doc.get("page", 1),
            score=float(doc.get("score", 0.0)),
            content=doc.get("content"),
            metadata=doc.get("metadata", {})
        ) for doc in response_data.get("documents", [])]

        return ChatResponse(
            response=response_data.get("response", ""),
            session_id=str(chat_session.session_id),
            conversation_id=str(uuid.uuid4()),
            documents=documents,
            confidence_score=float(response_data.get("confidence_score", 0.0)),
            processing_time=float(processing_time),
            tokens_used=response_data["tokens_used"]["total"],
            tokens_details=response_data["tokens_used"],
            metadata={
                "source": response_data.get("source", "model"),
                "timestamp": datetime.utcnow().isoformat(),
                "session_context": chat_session.session_context
            }
        )
    except Exception as e:
        logger.error(f"Erreur formatage réponse: {e}")
        return ChatResponse(
            response="Une erreur est survenue lors du formatage de la réponse.",
            session_id=str(chat_session.session_id),
            conversation_id=str(uuid.uuid4()),
            documents=[],
            confidence_score=0.0,
            processing_time=0.0,
            tokens_used=0,
            metadata={"error": str(e)}
        )

async def log_error(components, error: Exception, request: ChatRequest):
    """Log une erreur dans la base de données."""
    try:
        if hasattr(components, 'db_manager') and hasattr(components.db_manager, 'log_error'):
            await components.db_manager.log_error(
                error_type=type(error).__name__,
                error_message=str(error),
                endpoint="/api/chat",
                user_id=str(request.user_id),
                metadata={
                    "query": request.query,
                    "application": request.application,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        else:
            # Fallback vers le logger si la méthode log_error n'est pas disponible
            logger.error(
                f"Error in chat processing - "
                f"Type: {type(error).__name__}, "
                f"Message: {str(error)}, "
                f"User: {request.user_id}, "
                f"Query: {request.query}"
            )
    except Exception as e:
        logger.error(f"Erreur lors du log d'erreur: {e}")

# Helpers pour la gestion des types et validations

def validate_session_id(session_id: Optional[str]) -> bool:
    """Valide le format d'un ID de session."""
    if not session_id:
        return False
    try:
        uuid.UUID(session_id)
        return True
    except ValueError:
        return False

def safe_get_dict_value(data: dict, key: str, default: Any = None) -> Any:
    """Récupère une valeur d'un dictionnaire de manière sécurisée."""
    try:
        return data.get(key, default)
    except Exception:
        return default

def ensure_string(value: Any) -> str:
    """Convertit une valeur en string de manière sécurisée."""
    if value is None:
        return ""
    return str(value)

def ensure_float(value: Any, default: float = 0.0) -> float:
    """Convertit une valeur en float de manière sécurisée."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def ensure_int(value: Any, default: int = 0) -> int:
    """Convertit une valeur en int de manière sécurisée."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Formate un timestamp en string ISO."""
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat()
