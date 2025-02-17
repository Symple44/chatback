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

from ..models.requests import ChatRequest, SearchConfig  
from ..models.responses import (
    ChatResponse, 
    SearchMetrics,
    SearchMetadata, 
    ErrorResponse,
    DocumentReference,
    SimilarQuestion,
    VectorStats
)
from ..dependencies import get_components
from core.utils.logger import get_logger
from core.database.base import get_session_manager
from core.utils.metrics import metrics
from core.database.models import User, ChatSession, ChatHistory, ReferencedDocument
from core.database.base import DatabaseSession
from core.database.manager import DatabaseManager
from core.config.config import settings
from core.config.search_config import SEARCH_STRATEGIES_CONFIG
from core.chat.processor_factory import ProcessorFactory
from core.search.strategies import SearchMethod
from core.streaming.stream_manager import StreamManager



#A supprimer
from core.chat.processor import ChatProcessor

logger = get_logger("chat_routes")
router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def process_chat_message(
    request: ChatRequest,
    components=Depends(get_components)
) -> ChatResponse:
    """
    Traite une requête de chat avec configuration de recherche personnalisée.
    """
    try:
        # Initialisation des métriques de requête
        request_id = str(uuid.uuid4())
        metrics.start_request_tracking(request_id)

        # Vérification et obtention de la session
        chat_session = await components.session_manager.get_or_create_session(
            session_id=request.session_id,
            user_id=request.user_id,
            metadata={
                "search_config": request.search_config.dict() if request.search_config else None,
                "application": request.application
            }
        )

        # Obtention du processeur approprié
        processor = await components.processor_factory.get_processor(
            business_type=request.business,
            components=components
        )

        # Traitement du message
        response = await processor.process_message(request=request)

        # Mise à jour des métriques
        metrics.finish_request_tracking(request_id)
        
        return response

    except Exception as e:
        metrics.increment_counter("chat_errors")
        logger.error(f"Erreur traitement message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/metrics")
async def get_search_metrics() -> SearchMetrics:
    """
    Retourne les métriques détaillées de recherche.
    """
    metrics_data = metrics.get_search_metrics()
    
    return SearchMetrics(
        total_searches=metrics_data["total_searches"],
        success_rate=metrics_data["success_rate"],
        average_time=metrics_data["average_time"],
        cache_hit_rate=metrics_data["cache_hit_rate"],
        methods_usage={
            method.value: metrics_data["methods"].get(method.value, {})
            for method in SearchMethod
        },
        timestamp=datetime.utcnow()
    )

@router.post("/search/test")
async def test_search_configuration(
    config: SearchConfig,
    query: str = Query(..., min_length=1),
    components=Depends(get_components)
) -> Dict[str, Any]:
    """Teste une configuration de recherche spécifique."""
    try:
        # Les paramètres sont déjà validés et du bon type
        search_manager = components.search_manager.__class__(components)
        search_manager.configure(
            method=config.method,
            search_params=config.params.dict(),  # Conversion en dict pour l'API
            metadata_filter=config.metadata_filter
        )

        results = await search_manager.search_context(
            query=query,
            metadata_filter=config.metadata_filter
        )

        return {
            "success": True,
            "results_count": len(results),
            "results": results[:3],
            "configuration_used": config.dict()
        }

    except Exception as e:
        logger.error(f"Erreur test configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur test configuration: {str(e)}"
        )

@router.get("/stream")
async def stream_chat_response(
    query: str = Query(..., min_length=1),
    user_id: str = Query(...),
    search_method: SearchMethod = Query(default=SearchMethod.RAG),
    session_id: Optional[str] = None,
    language: str = Query(default="fr"),
    application: Optional[str] = None,
    components=Depends(get_components)
):
    """Stream une réponse de chat avec optimisation des ressources."""
    
    stream_manager = StreamManager()
    
    async def event_generator():
        try:
            # Configuration de la recherche
            search_params = {
                "max_docs": settings.MAX_RELEVANT_DOCS,
                "min_score": settings.CONFIDENCE_THRESHOLD
            }
            
            # Récupération de la session
            chat_session = await components.session_manager.get_or_create_session(
                session_id=session_id,
                user_id=user_id,
                metadata={
                    "search_method": search_method,
                    "application": application
                }
            )
            
            # Configuration du processeur
            processor = await ProcessorFactory.get_processor(
                business_type="generic",
                components=components
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
            async for event in stream_manager.stream_response(
                query=query,
                model=components.model,
                context_docs=relevant_docs,
                language=language,
                session=chat_session
            ):
                yield event
                
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

@router.get("/search/config")
async def get_search_configuration() -> Dict[str, Any]:
    """
    Retourne la configuration complète des méthodes de recherche.
    """
    return {
        "methods": {
            method.value: {
                "name": method.value,
                "description": {
                    "rag": "Recherche par RAG (Retrieval Augmented Generation)",
                    "hybrid": "Recherche hybride combinant RAG et sémantique",
                    "semantic": "Recherche purement sémantique",
                    "disabled": "Recherche désactivée"
                }.get(method.value, ""),
                "config": SEARCH_STRATEGIES_CONFIG.get(method.value, {}),
                "default_params": SEARCH_STRATEGIES_CONFIG.get(method.value, {}).get("search_params", {})
            }
            for method in SearchMethod
            if method != SearchMethod.DISABLED
        },
        "defaults": {
            "method": settings.DEFAULT_SEARCH_METHOD,
            "max_docs": settings.SEARCH_MAX_DOCS,
            "min_score": settings.SEARCH_MIN_SCORE,
            "rag": {
                "vector_weight": settings.RAG_VECTOR_WEIGHT,
                "semantic_weight": settings.RAG_SEMANTIC_WEIGHT
            },
            "hybrid": {
                "rerank_top_k": settings.HYBRID_RERANK_TOP_K,
                "window_size": settings.HYBRID_WINDOW_SIZE
            },
            "semantic": {
                "max_concepts": settings.SEMANTIC_MAX_CONCEPTS,
                "boost_exact": settings.SEMANTIC_BOOST_EXACT
            }
        },
        "constraints": {
            "max_docs_limit": 20,
            "min_score_range": [0.0, 1.0],
            "vector_weight_range": [0.0, 1.0],
            "semantic_weight_range": [0.0, 1.0]
        },
        "performance": {
            "cache_enabled": True,
            "cache_ttl": settings.SEARCH_CACHE_TTL,
            "max_concurrent_searches": settings.SEARCH_MAX_CONCURRENT
        },
        "metrics": await get_search_metrics()
    }
@router.get("/search/methods")
async def get_search_methods():
    """
    Retourne la liste simplifiée des méthodes de recherche disponibles.
    """
    methods_info = {
        method.value: SEARCH_STRATEGIES_CONFIG.get(method.value, {}).get("search_params", {})
        for method in SearchMethod
        if method != SearchMethod.DISABLED
    }

    return {
        "methods": [method.value for method in SearchMethod if method != SearchMethod.DISABLED],
        "default": SearchMethod.RAG.value,
        "configurations": methods_info,
        "defaults": {
            method: config.get("search_params", {})
            for method, config in SEARCH_STRATEGIES_CONFIG.items()
        }
    }

@router.get("/search/status")
async def get_search_status(components=Depends(get_components)) -> Dict[str, Any]:
    """
    Retourne le statut actuel du système de recherche.
    """
    try:
        status = {
            "active_method": components.search_manager.current_method.value,
            "enabled": components.search_manager.enabled,
            "current_params": components.search_manager.current_params,
            "cache": {
                "size": len(components.search_manager.cache),
                "hit_rate": metrics.get_cache_hit_rate()
            },
            "performance": {
                "average_response_time": metrics.get_timer_value("search_time"),
                "total_searches": metrics.counters.get("total_searches", 0),
                "successful_searches": metrics.counters.get("successful_searches", 0)
            }
        }
        return status
    except Exception as e:
        logger.error(f"Erreur récupération statut recherche: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur récupération statut: {str(e)}"
        )

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
