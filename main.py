#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import asyncio
import uvicorn
from datetime import datetime
from contextlib import asynccontextmanager
import uuid
import json
import shutil
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Import internes avant setup
from core.config import settings
from core.utils.logger import get_logger, logger_manager
from core.utils.metrics import metrics
from core.utils.system_optimizer import SystemOptimizer
from core.llm.cuda_manager import CUDAManager
from core.storage.google_drive import GoogleDriveManager

logger = get_logger("main")

class CustomJSONEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)

class CustomJSONResponse(JSONResponse):
    """Réponse JSON personnalisée."""
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            cls=CustomJSONEncoder
        ).encode("utf-8")

async def setup_environment():
    """Configure l'environnement complet avant le démarrage."""
    try:
        # Configuration CUDA
        cuda_manager = CUDAManager()
        await cuda_manager.initialize()
        logger.info("Configuration CUDA initialisée")

        # Configuration des bibliothèques numériques
        os.environ["MKL_NUM_THREADS"] = os.getenv("MKL_NUM_THREADS", "16")
        os.environ["NUMEXPR_NUM_THREADS"] = os.getenv("NUMEXPR_NUM_THREADS", "16")
        os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS", "16")
        os.environ["OPENBLAS_NUM_THREADS"] = os.getenv("OPENBLAS_NUM_THREADS", "16")
        
        # Création des répertoires
        REQUIRED_DIRS = [
            "offload_folder",
            "static",
            "documents",
            "model_cache",
            "logs",
            "data",
            "temp"
        ]
        for dir_name in REQUIRED_DIRS:
            path = Path(dir_name)
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o755)  # Permissions sécurisées
            
        logger.info("Environnement configuré avec succès")
            
    except Exception as e:
        logger.error(f"Erreur configuration environnement: {e}")
        raise

# Instance globale des composants
components = ComponentManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    try:
        # Activation du tracemalloc en dev
        if settings.DEBUG:
            import tracemalloc
            tracemalloc.start()
            
        # Configuration initiale de l'environnement
        await setup_environment()
            
        # Initialisation des composants
        await logger_manager.initialize()
        await metrics.initialize()
        
        try:
            await components.initialize()
        except Exception as e:
            logger.critical(f"Erreur critique lors de l'initialisation des composants: {e}")
            await components.cleanup()
            raise

        # Import et ajout des routes
        from api.routes.router import router as api_router
        app.include_router(api_router)
        
        if settings.GOOGLE_DRIVE_SYNC_INTERVAL:
            asyncio.create_task(periodic_sync())

        yield

    except Exception as e:
        logger.critical(f"Erreur fatale pendant le démarrage: {e}")
        raise
    finally:
        # Nettoyage
        try:
            await components.cleanup()
            if settings.DEBUG and tracemalloc.is_tracing():
                tracemalloc.stop()
        except Exception as e:
            logger.error(f"Erreur pendant le nettoyage final: {e}")
        logger.info("Application arrêtée")

async def periodic_sync():
    """Tâche périodique de synchronisation."""
    while True:
        await asyncio.sleep(settings.GOOGLE_DRIVE_SYNC_INTERVAL)
        try:
            await components.sync_drive_documents()
        except Exception as e:
            logger.error(f"Erreur sync périodique: {e}")

# Configuration FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    default_response_class=CustomJSONResponse
)

# Middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Fichiers statiques
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Gestionnaire d'erreurs global."""
    error_id = str(uuid.uuid4())
    logger.error(f"Erreur [{error_id}]: {exc}", exc_info=True)
    return CustomJSONResponse(
        status_code=500,
        content={
            "error_id": error_id,
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
async def shutdown():
    """Nettoie proprement les ressources lors de l'arrêt."""
    try:
        logger.info("Arrêt de l'application...")
        
        # Nettoyage des composants
        await components.cleanup()
        
        # Nettoyage des répertoires temporaires
        temp_dirs = ["temp", "temp/pdf", "offload_folder"]
        for dir_path in temp_dirs:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.info(f"Répertoire {dir_path} nettoyé")
        
        logger.info("Application arrêtée proprement")
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt: {e}")

if __name__ == "__main__":
    try:
        # Configuration uvicorn optimisée
        config = uvicorn.Config(
            "main:app",
            host="0.0.0.0",
            port=settings.PORT,
            reload=settings.DEBUG,
            workers=1 if settings.DEBUG else settings.WORKERS,
            log_level="info",
            loop="uvloop",
            timeout_keep_alive=settings.KEEPALIVE,
            limit_max_requests=settings.MAX_REQUESTS,
            limit_concurrency=settings.MAX_CONCURRENT_REQUESTS,
            backlog=2048
        )
        
        server = uvicorn.Server(config)
        server.run()

    except KeyboardInterrupt:
        logger.info("Arrêt manuel détecté")
        asyncio.run(shutdown())
    except Exception as e:
        logger.critical(f"Erreur fatale: {e}")
        sys.exit(1)
