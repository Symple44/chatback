# api/routes/router.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import Dict
from datetime import datetime

from .chat_routes import router as chat_router
from .session_routes import router as session_router
from .user_routes import router as user_router
from .health_routes import router as health_router
from .history_routes import router as history_router
from core.utils.logger import get_logger
from core.config import settings

logger = get_logger("router")
router = APIRouter(prefix="/api")

# Configuration des tags pour la documentation
tags_metadata = [
    {
        "name": "chat",
        "description": "Opérations de conversation et de traitement des messages"
    },
    {
        "name": "sessions",
        "description": "Gestion des sessions utilisateur"
    },
    {
        "name": "users",
        "description": "Gestion des utilisateurs et de leurs données"
    },
    {
        "name": "history",
        "description": "Accès à l'historique des conversations"
    },
    {
        "name": "health",
        "description": "Surveillance et monitoring du système"
    }
]

# Inclusion des routers spécifiques
router.include_router(chat_router, prefix="/chat", tags=["chat"])
router.include_router(session_router, prefix="/sessions", tags=["sessions"])
router.include_router(user_router, prefix="/users", tags=["users"])
router.include_router(health_router, prefix="/health", tags=["health"])
router.include_router(history_router, prefix="/history", tags=["history"])

@router.get("/", tags=["info"])
async def api_info() -> Dict:
    """
    Retourne les informations générales sur l'API.
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "description": "API de chat avec support vectoriel",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": "/docs",
        "environment": os.getenv("ENV", "production")
    }

@router.get("/routes", tags=["info"])
async def list_routes() -> Dict:
    """
    Liste toutes les routes disponibles.
    """
    available_routes = []
    for r in router.routes:
        route_info = {
            "path": r.path,
            "methods": list(r.methods) if hasattr(r, 'methods') else [],
            "name": r.name,
            "tags": r.tags if hasattr(r, 'tags') else []
        }
        available_routes.append(route_info)
    
    return {
        "total_routes": len(available_routes),
        "routes": available_routes
    }

# Gestionnaires d'erreurs globaux
@router.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestionnaire d'erreurs global."""
    logger.error(f"Erreur non gérée: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Une erreur interne est survenue",
            "type": type(exc).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@router.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Gestionnaire pour les erreurs de validation."""
    logger.warning(f"Erreur de validation: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": str(exc),
            "type": "ValidationError",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Middleware pour le logging des requêtes
@router.middleware("http")
async def log_requests(request, call_next):
    """Middleware pour logger les requêtes."""
    start_time = datetime.utcnow()
    response = await call_next(request)
    
    # Calcul du temps de réponse
    process_time = (datetime.utcnow() - start_time).total_seconds()
    
    # Log de la requête
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Duration: {process_time:.3f}s"
    )
    
    # Ajout des headers de monitoring
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-API-Version"] = settings.VERSION
    
    return response

# Configuration de la documentation
def setup_docs(app):
    """Configure la documentation OpenAPI."""
    app.title = settings.APP_NAME
    app.description = """
    API de chat avec support vectoriel et traitement du langage naturel.
    
    Fonctionnalités principales:
    * Gestion des conversations avec contexte
    * Recherche vectorielle de documents
    * Sessions utilisateur
    * Historique des conversations
    * Monitoring et métriques
    """
    app.version = settings.VERSION
    app.openapi_tags = tags_metadata
