# api/routes/router.py
from fastapi import APIRouter, Depends, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse
from datetime import datetime
import os
from typing import Dict, List

from .chat_routes import router as chat_router
from .session_routes import router as session_router
from .user_routes import router as user_router
from .health_routes import router as health_router
from .history_routes import router as history_router
from .model_routes import router as model_router
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings
from ..dependencies import get_components

logger = get_logger("router")

# Configuration des tags pour la documentation OpenAPI
tags_metadata = [
    {
        "name": "chat",
        "description": "Opérations de conversation et de traitement des messages",
    },
    {
        "name": "models",
        "description": "Gestion des modèes",
    },
    {
        "name": "sessions",
        "description": "Gestion des sessions utilisateur et contextes",
    },
    {
        "name": "users",
        "description": "Gestion des utilisateurs et préférences",
    },
    {
        "name": "history",
        "description": "Accès à l'historique des conversations",
    },
    {
        "name": "health",
        "description": "Surveillance et monitoring du système",
    },
    {
        "name": "info",
        "description": "Informations générales sur l'API"
    }
]

# Création du router principal avec préfixe
router = APIRouter(prefix="/api")

# Inclusion des sous-routers avec leurs préfixes
router.include_router(
    chat_router,
    #prefix="/chat",
    tags=["chat"]
)
router.include_router(
    model_router,
    #prefix="/models",
    tags=["models"]
)
router.include_router(
    session_router,
    #prefix="/sessions",
    tags=["sessions"]
)
router.include_router(
    user_router,
    #prefix="/users",
    tags=["users"]
)
router.include_router(
    health_router,
    #prefix="/health",
    tags=["health"]
)
router.include_router(
    history_router,
    #prefix="/history",
    tags=["history"]
)


@router.get("/", tags=["info"])
async def api_info() -> Dict:
    """
    Retourne les informations générales sur l'API.

    Returns:
        Dict: Informations sur l'API incluant version, statut et environnement
    """
    with metrics.timer("api_info_request"):
        return {
            "name": settings.APP_NAME,
            "version": settings.VERSION,
            "description": "API de chat avec support vectoriel",
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "documentation": "/docs",
            "environment": os.getenv("ENV", "production"),
            "build_date": settings.BUILD_DATE,
            "contact": settings.CONTACT_EMAIL
        }

@router.get("/routes", tags=["info"])
async def list_routes() -> Dict:
    """
    Liste toutes les routes disponibles dans l'API.

    Returns:
        Dict: Liste des routes avec leurs méthodes et tags
    """
    with metrics.timer("list_routes_request"):
        available_routes = []
        for r in router.routes:
            route_info = {
                "path": r.path,
                "methods": list(r.methods) if hasattr(r, 'methods') else [],
                "name": r.name,
                "tags": r.tags if hasattr(r, 'tags') else [],
                "description": r.description if hasattr(r, 'description') else None
            }
            available_routes.append(route_info)
        
        # Tri des routes par chemin
        available_routes.sort(key=lambda x: x["path"])
        
        return {
            "total_routes": len(available_routes),
            "routes": available_routes
        }

@router.get("/ping", tags=["health"])
async def ping() -> Dict:
    """
    Endpoint simple pour vérifier que l'API répond.

    Returns:
        Dict: Message de confirmation avec timestamp
    """
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/sync", tags=["documents"])
async def sync_documents(
    background_tasks: BackgroundTasks,
    components=Depends(get_components)
) -> Dict:
    """Lance une synchronisation manuelle avec Google Drive."""
    background_tasks.add_task(components.sync_drive_documents)
    return {
        "message": "Synchronisation lancée",
        "status": "pending",
        "timestamp": datetime.utcnow().isoformat()
    }