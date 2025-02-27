# api/routes/chat_routes.py
from sqlalchemy import select
from core.database.base import DatabaseSession
from core.database.models import User
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from typing import Dict, Optional, List, Any
from datetime import datetime
import uuid
import asyncio
from sse_starlette.sse import EventSourceResponse
import json
from pydantic import ValidationError
import torch
import psutil
from ..models.requests import ChatRequest, SearchConfig  
from ..models.responses import (
    ChatResponse, 
    SearchMetrics,
    DocumentReference,
    VectorStats,
    SearchTestResponse,  
    SearchError,
    SearchDebugInfo,
    SearchValidationError
)
from ..dependencies import get_components
from main import CustomJSONEncoder
from main import CustomJSONResponse
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings
from core.config.search_config import SEARCH_STRATEGIES_CONFIG
from core.chat.processor_factory import ProcessorFactory
from core.search.strategies import SearchMethod
from core.streaming.stream_manager import StreamManager
from core.config.chat import BusinessType

logger = get_logger("chat_routes")
router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def process_chat_message(
    request: ChatRequest,
    components=Depends(get_components)
) -> ChatResponse:
    try:
        request_id = str(uuid.uuid4())
        metrics.start_request_tracking(request_id)
        
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

        # Mise à jour des paramètres de recherche dans le SearchManager
        if request.search_config:
            await components.search_manager.configure(
                method=request.search_config.method,
                search_params=request.search_config.params.dict(),
                metadata_filter=request.search_config.metadata_filter
            )

        # Obtention du processeur avec les paramètres
        processor = await components.processor_factory.get_processor(
            business_type=request.business,
            components=components
        )

        # Traitement avec tous les paramètres
        response = await processor.process_message(
            request={
                **request.dict(exclude_unset=True),
                "session_id": chat_session.session_id
            },
            context={
                "history": chat_session.history if hasattr(chat_session, "history") else [],
                "metadata": chat_session.metadata if hasattr(chat_session, "metadata") else {}
            }
        )

        metrics.finish_request_tracking(request_id)
        return response

    except Exception as e:
        metrics.increment_counter("chat_errors")
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

@router.post("/search/test", response_model=SearchTestResponse)
async def test_search_configuration(
    request: Request,
    config: SearchConfig,
    query: str = Query(
        ..., 
        min_length=1,
        max_length=500,
        description="Requête de test pour la recherche"
    ),
    debug: bool = Query(
        False,
        description="Active les informations de debug"
    ),
    components=Depends(get_components)
) -> SearchTestResponse:
    """
    Teste une configuration de recherche spécifique.
    
    Args:
        request: Requête FastAPI
        config: Configuration de recherche à tester
        query: Requête de test
        debug: Mode debug pour plus d'informations
        components: Composants de l'application
        
    Returns:
        SearchTestResponse: Résultats détaillés du test
        
    Raises:
        HTTPException: En cas d'erreur de validation ou d'exécution
    """
    try:
        # Création d'un identifiant unique pour le test
        test_id = uuid.uuid4()
        start_time = datetime.utcnow()
        
        # Démarrage du suivi des métriques
        metrics.start_request_tracking(str(test_id))

        # 1. Validation des paramètres
        await validate_search_config(config)
        
        # 2. Configuration du SearchManager
        search_manager = components.search_manager.__class__(components)

        # Simplifier la fusion des paramètres
        search_params = config.params.dict()  # Commencer avec les paramètres utilisateur
        logger.debug(f"User params: {search_params}")

        await search_manager.configure(
            method=config.method,
            search_params=search_params,  # Passer directement les paramètres utilisateur
            metadata_filter=config.metadata_filter
        )

        # 3. Collecte des métriques de performance initiales
        initial_memory = {
            "gpu": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            "gpu_reserved": torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_available() else 0,
            "ram": psutil.Process().memory_info().rss / 1024**2
        }

        # 4. Exécution de la recherche
        vector_start = datetime.utcnow()
        query_vector = await components.model.create_embedding(query)
        query_vector_time = (datetime.utcnow() - vector_start).total_seconds()

        search_start = datetime.utcnow()
        results = await search_manager.search_context(
            query=query,
            metadata_filter=config.metadata_filter
        )
        search_time = (datetime.utcnow() - search_start).total_seconds()

        # 5. Collecte des métriques finales
        final_memory = {
            "gpu": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            "gpu_reserved": torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_available() else 0,
            "ram": psutil.Process().memory_info().rss / 1024**2
        }

        memory_usage = {
            "gpu_delta": final_memory["gpu"] - initial_memory["gpu"],
            "ram_delta": final_memory["ram"] - initial_memory["ram"]
        }

        # 6. Préparation des résultats
        processed_results = []
        for result in results:
            doc_ref = DocumentReference(
                title=result.content[:50] if result.content else "Sans titre",  # Utilise les premiers caractères comme titre
                score=result.score,
                content=result.content[:200],  # Limite pour l'aperçu
                metadata={
                    **result.metadata,
                    "relevance_score": result.score,
                    "source": "search_result"  # Valeur par défaut
                },
                vector_id=str(uuid.uuid4()),
                last_updated=datetime.utcnow()
            )
            processed_results.append(doc_ref)

        # 7. Collecte des métriques globales
        metrics_data = {
            "cache_hit": metrics.get_cache_hit_rate() if hasattr(metrics, 'get_cache_hit_rate') else False,
            "processing_time": metrics.get_timer_value("search_time")
        }
        cache_info = search_manager.get_cache_info() if hasattr(search_manager, 'get_cache_info') else {}

        # 8. Construction de la réponse
        max_docs = search_params.get('max_docs', settings.search.SEARCH_MAX_DOCS)
        response = SearchTestResponse(
            test_id=test_id,
            success=True,
            method=config.method,
            configuration_used=search_params,
            results=processed_results[:max_docs],
            metrics=SearchMetrics(
                processing_time=search_time,
                results_count=len(results),
                cache_hit=metrics_data.get("cache_hit", False),
                memory_usage=memory_usage,
                query_vector_time=query_vector_time,
                gpu_metrics={  # Ajout des métriques GPU
                    "total_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
                    "used_memory": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                    "peak_memory": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                } if torch.cuda.is_available() else None
            ),
            timestamp=datetime.utcnow()
        )

        # 9. Ajout des informations de debug si demandées
        if debug:
            response.debug_info = SearchDebugInfo(
                raw_config=config.dict(),
                search_strategy=search_manager.current_method.value,
                cache_status=cache_info,
                memory_details={
                    "initial": initial_memory,
                    "final": final_memory,
                    "delta": memory_usage
                },
                request_headers=dict(request.headers),
                timing_breakdown={
                    "total_time": (datetime.utcnow() - start_time).total_seconds(),
                    "vector_generation": query_vector_time,
                    "search_execution": search_time,
                    "processing_overhead": metrics_data.get("processing_time", 0)
                }
            )

        # 10. Nettoyage et retour
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        return response

    except ValidationError as e:
        error = SearchValidationError(
            field=e.errors()[0]["loc"][0],
            value=e.errors()[0]["input"],
            constraint=e.errors()[0]["msg"]
        )
        raise HTTPException(
            status_code=422,
            detail=error.dict()
        )

    except Exception as e:
        metrics.increment_counter("search_errors")
        error = SearchError(
            error_code="SEARCH_ERROR",
            detail=str(e),
            metadata={
                "method": config.method.value if config else None,
                "error_type": type(e).__name__,
                "query": query
            }
        )
        # Utilisation de CustomJSONResponse pour gérer la sérialisation
        return CustomJSONResponse(
            status_code=500,
            content=error.dict()
        )

    finally:
        # Nettoyage final
        metrics.finish_request_tracking(str(test_id))
        
async def validate_search_config(config: SearchConfig) -> None:
    """Valide la configuration de recherche."""
    base_config = SEARCH_STRATEGIES_CONFIG.get(config.method.value)
    if not base_config:
        raise SearchValidationError(
            field="method",
            value=config.method.value,
            constraint="unsupported_method"
        )

    method_params = base_config["search_params"]
    for key, value in config.params.dict().items():
        # Vérification des paramètres supportés
        if key not in method_params:
            raise SearchValidationError(
                field=key,
                value=value,
                constraint="unsupported_parameter"
            )
        
        # Validation des plages de valeurs
        if isinstance(value, (int, float)):
            if key in ["min_score", "vector_weight", "semantic_weight"]:
                if not 0 <= value <= 1:
                    raise SearchValidationError(
                        field=key,
                        value=value,
                        constraint="value_out_of_range_0_1"
                    )
            elif key == "max_docs":
                if not 1 <= value <= 20:
                    raise SearchValidationError(
                        field=key,
                        value=value,
                        constraint="value_out_of_range_1_20"
                    )

@router.get("/stream")
async def stream_chat_response(
    query: str = Query(..., min_length=1),
    user_id: str = Query(...),
    business: BusinessType = Query(default=BusinessType.GENERIC),
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
                "max_docs": settings.search.SEARCH_MAX_DOCS,
                "min_score": settings.chat.CONFIDENCE_THRESHOLD  
            }
            
            # Récupération de la session
            chat_session = await components.session_manager.get_or_create_session(
                session_id=session_id,
                user_id=user_id,
                metadata={
                    "search_method": search_method,
                    "application": application,
                    "business": business
                }
            )
            
            # Configuration du processeur
            processor = await components.processor_factory.get_processor(
                business_type=business,
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
                business=business,  
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
            "method": settings.search.DEFAULT_SEARCH_METHOD,
            "max_docs": settings.search.SEARCH_MAX_DOCS,
            "min_score": settings.search.SEARCH_MIN_SCORE,
            "rag": {
                "vector_weight": settings.search.RAG_VECTOR_WEIGHT,
                "semantic_weight": settings.search.RAG_SEMANTIC_WEIGHT
            },
            "hybrid": {
                "rerank_top_k": settings.search.HYBRID_RERANK_TOP_K,
                "window_size": settings.search.HYBRID_WINDOW_SIZE
            },
            "semantic": {
                "max_concepts": settings.search.SEMANTIC_MAX_CONCEPTS,
                "boost_exact": settings.search.SEMANTIC_BOOST_EXACT
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
            "cache_ttl": settings.search.SEARCH_CACHE_TTL,
            "max_concurrent_searches": settings.search.SEARCH_MAX_CONCURRENT
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
    """Retourne le statut actuel du système de recherche."""
    try:
        cache_hit_rate = metrics.get_cache_hit_rate() if hasattr(metrics, 'get_cache_hit_rate') else 0.0
        
        status = {
            "active_method": components.search_manager.current_method.value,
            "enabled": components.search_manager.enabled,
            "current_params": components.search_manager.current_params,
            "cache": {
                "size": len(components.search_manager.cache),
                "hit_rate": cache_hit_rate
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
