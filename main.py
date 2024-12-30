#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def setup_cuda_environment():
    """Configure l'environnement CUDA avant le chargement des bibliothèques."""
    
    # Désactive les avertissements TF
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    # Force l'ordre de chargement CUDA
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    
    # Configuration des bibliothèques numériques
    os.environ["MKL_NUM_THREADS"] = os.getenv("MKL_NUM_THREADS", "16")
    os.environ["NUMEXPR_NUM_THREADS"] = os.getenv("NUMEXPR_NUM_THREADS", "16")
    os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS", "16")
    os.environ["OPENBLAS_NUM_THREADS"] = os.getenv("OPENBLAS_NUM_THREADS", "16")
    
    # Configuration CUDA pour la RTX 3090
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Configuration de la mémoire PyTorch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"
    
    # Désactive JIT PyTorch pour éviter les conflits
    os.environ["PYTORCH_JIT"] = "0"
    
    # Configuration du provider CUDA pour ONNX
    os.environ["ONNXRUNTIME_CUDA_PROVIDER_OPTIONS"] = """
    {
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_use_max_workspace": "1",
        "do_copy_in_default_stream": "1"
    }
    """

def setup_environment():
    """Configure l'environnement complet avant le démarrage."""
    # Configuration CUDA
    setup_cuda_environment()
    
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

# Exécution de la configuration avant tout import
if __name__ == "__main__":
    setup_environment()
    
# Import des dépendances après la configuration
import asyncio
import uvicorn
from datetime import datetime
from contextlib import asynccontextmanager
import logging
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

# Imports internes
from core.config import settings
from core.utils.logger import get_logger, logger_manager
from core.utils.metrics import metrics
from core.utils.system_optimizer import SystemOptimizer
from core.storage.google_drive import GoogleDriveManager

# Configuration du logger
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

class ComponentManager:
    """Gestionnaire des composants de l'application."""
    
    def __init__(self):
        """Initialisation du gestionnaire."""
        self.start_time = datetime.utcnow()
        self.initialized = False
        self._components = {}
        self._init_lock = asyncio.Lock()
        self.system_optimizer = SystemOptimizer()

    async def initialize(self):
        """Initialise tous les composants."""
        async with self._init_lock:
            if self.initialized:
                return

            try:
                # Optimisation système
                await self.system_optimizer.optimize()

                # Base de données
                from core.database.base import get_session_manager
                self._components["db"] = get_session_manager(settings.get_database_url())
                logger.info("Base de données initialisée")

                # Session Manager
                from core.session_manager import SessionManager
                self._components["session_manager"] = SessionManager(settings.get_database_url())
                logger.info("Session manager initialisé")

                # Cache Redis
                from core.cache import RedisCache
                cache = RedisCache()
                await cache.initialize()
                self._components["cache"] = cache
                logger.info("Cache Redis initialisé")

                # Elasticsearch
                from core.vectorstore.search import ElasticsearchClient
                es_client = ElasticsearchClient()
                await es_client.initialize()
                self._components["es_client"] = es_client
                logger.info("Elasticsearch initialisé")

                # Modèle LLM
                from core.llm.model import ModelInference
                model = ModelInference()
                await model.initialize()  
                self._components["model"] = model
                logger.info("Modèle LLM initialisé")

                # Processeurs de documents
                from core.document_processing.extractor import DocumentExtractor
                from core.document_processing.pdf_processor import PDFProcessor
                self._components["doc_extractor"] = DocumentExtractor()
                self._components["pdf_processor"] = PDFProcessor(es_client)
                logger.info("Processeurs de documents initialisés")
                
                # Google Drive (optionnel)
                if settings.GOOGLE_DRIVE_CREDENTIALS_PATH:
                    drive_manager = GoogleDriveManager(
                        credentials_path=settings.GOOGLE_DRIVE_CREDENTIALS_PATH
                    )
                    if await drive_manager.initialize():
                        self._components["drive_manager"] = drive_manager
                        await self.sync_drive_documents()
                        logger.info("Google Drive manager initialisé")
                
                self.initialized = True
                logger.info("Initialisation des composants terminée")

            except Exception as e:
                logger.critical(f"Erreur critique lors de l'initialisation: {e}")
                await self.cleanup()
                raise

    async def sync_drive_documents():
        """Synchronise les documents depuis Google Drive."""
        if "drive_manager" not in components._components:
            return
    
        try:
            downloaded_files = await components.drive_manager.sync_drive_folder(
                settings.GOOGLE_DRIVE_FOLDER_ID,
                save_path="documents"
            )
            
            if not downloaded_files:
                logger.info("Aucun nouveau fichier à indexer")
                return
                
            # Index les fichiers téléchargés
            for file_path in downloaded_files:
                app_name = Path(file_path).parent.name
                await components.pdf_processor.index_pdf(
                    file_path,
                    metadata={
                        "application": app_name,
                        "sync_date": datetime.utcnow().isoformat()
                    }
                )
                
            logger.info(f"{len(downloaded_files)} documents synchronisés et indexés")
            
        except Exception as e:
            logger.error(f"Erreur synchronisation Drive: {e}", exc_info=True)

    async def cleanup(self):
        """Nettoie les ressources."""
        for name, component in self._components.items():
            try:
                if hasattr(component, 'cleanup'):
                    await component.cleanup()
                logger.info(f"Composant {name} nettoyé")
            except Exception as e:
                logger.error(f"Erreur nettoyage {name}: {e}")

        self._components.clear()
        self.initialized = False

    def __getattr__(self, name):
        if name in self._components:
            return self._components[name]
        raise AttributeError(f"Composant {name} non trouvé")

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
            
        # Initialisation
        await logger_manager.initialize()
        await metrics.initialize()
        try:
            await components.initialize()
        except Exception as e:
            logger.critical(f"Erreur critique lors de l'initialisation des composants: {e}")
            # Nettoyage explicite en cas d'erreur
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
            workers=1 if settings.DEBUG else int(os.getenv("WORKERS", "1")),
            log_level="info",
            loop="auto",
            timeout_keep_alive=settings.KEEPALIVE,
            limit_concurrency=int(os.getenv("MAX_CONCURRENT_REQUESTS", "100")),
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
