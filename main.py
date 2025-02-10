#!/usr/bin/env python3
import os
import sys
import signal
import asyncio
import uvicorn
from datetime import datetime
from contextlib import asynccontextmanager
import uuid
import json
import shutil
from typing import Dict, Optional, List
from pathlib import Path

from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from core.config.config import settings
from core.utils.logger import get_logger, logger_manager
from core.utils.metrics import metrics
from core.utils.system_optimizer import SystemOptimizer

from core.database.manager import DatabaseManager
from core.database.base import get_session_manager
from core.database.session_manager import SessionManager
from core.cache import RedisCache
from core.vectorstore import ElasticsearchClient
from core.llm.auth_manager import HuggingFaceAuthManager
from core.llm.model_manager import ModelManager
from core.llm.model_loader import ModelLoader, ModelType
from core.llm.cuda_manager import CUDAManager
from core.llm.embedding_manager import EmbeddingManager 
from core.llm.summarizer import DocumentSummarizer
from core.llm.tokenizer_manager import TokenizerManager
from core.llm.model import ModelInference

from core.storage.google_drive import GoogleDriveManager
from core.document_processing.extractor import DocumentExtractor
from core.document_processing.pdf_processor import PDFProcessor

logger = get_logger("main")

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, ModelType):
            return obj.value
        return super().default(obj)

class CustomJSONResponse(JSONResponse):
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
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.initialized = False
        self._components = {}
        self._init_lock = asyncio.Lock()
        self.system_optimizer = SystemOptimizer()
        self._shutdown_event = asyncio.Event()

    async def initialize(self):
        async with self._init_lock:
            if self.initialized:
                return

            try:
                # Optimisation système
                await self.system_optimizer.optimize()

                # Base de données
                db_session_manager = get_session_manager(settings.get_database_url())
                db_manager = DatabaseManager(db_session_manager.session_factory)
                self._components["db_manager"] = db_manager
                self._components["db"] = db_session_manager
                logger.info("Base de données initialisée")

                # Session Manager
                self._components["session_manager"] = SessionManager(settings.get_database_url())
                logger.info("Session manager initialisé")

                # Cache Redis
                cache = RedisCache()
                await cache.initialize()
                self._components["cache"] = cache
                logger.info("Cache Redis initialisé")

                # Elasticsearch
                es_client = ElasticsearchClient()
                await es_client.initialize()
                self._components["es_client"] = es_client
                logger.info("Elasticsearch initialisé")
                
                # Auth Manager
                auth_manager = HuggingFaceAuthManager()
                await auth_manager.setup_auth()
                self._components["auth_manager"] = auth_manager
                logger.info("Auth Manager initialisé")

                # CUDA Manager
                cuda_manager = CUDAManager()
                await cuda_manager.initialize()
                self._components["cuda_manager"] = cuda_manager
                logger.info("CUDA Manager initialisé")

                # Tokenizer Manager
                tokenizer_manager = TokenizerManager()
                await tokenizer_manager.initialize()
                self._components["tokenizer_manager"] = tokenizer_manager
                logger.info("Tokenizer Manager initialisé")

                # Model Loader
                model_loader = ModelLoader(self.cuda_manager, self.tokenizer_manager)
                self._components["model_loader"] = model_loader

                # Model Manager
                model_manager = ModelManager(self.cuda_manager, self.tokenizer_manager)
                await model_manager.initialize()
                self._components["model_manager"] = model_manager

                # Embedding Manager
                embedding_manager = EmbeddingManager()
                await embedding_manager.initialize(model_manager)
                self._components["embedding_manager"] = embedding_manager

                # Summarizer
                summarizer = DocumentSummarizer()
                await summarizer.initialize(model_manager)
                self._components["summarizer"] = summarizer

                # Model Inference
                model = ModelInference()
                await model.initialize(self)
                self._components["model"] = model
                logger.info("Model Inference initialisé")

                # Document Processing
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
                        logger.info("Google Drive manager initialisé")
                
                self.initialized = True
                logger.info("Initialisation des composants terminée")

            except Exception as e:
                logger.critical(f"Erreur critique lors de l'initialisation: {e}")
                await self.cleanup()
                raise

    async def cleanup(self):
        """Nettoie proprement toutes les ressources."""
        try:
            logger.info("Début du nettoyage des composants...")
            cleanup_tasks = []

            # Nettoyage des composants avec gestion des erreurs individuelles
            for name, component in self._components.items():
                try:
                    if hasattr(component, 'cleanup'):
                        task = asyncio.create_task(component.cleanup())
                        cleanup_tasks.append(task)
                        logger.info(f"Tâche de nettoyage créée pour {name}")
                except Exception as e:
                    logger.error(f"Erreur lors de la création de la tâche de nettoyage pour {name}: {e}")

            # Attendre la fin de tous les nettoyages avec timeout
            if cleanup_tasks:
                done, pending = await asyncio.wait(cleanup_tasks, timeout=30)
                
                # Annuler les tâches qui n'ont pas terminé
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        logger.warning(f"Une tâche de nettoyage a été annulée après timeout")

            # Nettoyage des répertoires temporaires
            temp_dirs = ["temp", "offload_folder", "cache"]
            for dir_path in temp_dirs:
                try:
                    if os.path.exists(dir_path):
                        shutil.rmtree(dir_path)
                        logger.info(f"Répertoire {dir_path} nettoyé")
                except Exception as e:
                    logger.error(f"Erreur lors du nettoyage du répertoire {dir_path}: {e}")

            self._components.clear()
            self.initialized = False
            logger.info("Nettoyage des composants terminé")

        except Exception as e:
            logger.error(f"Erreur pendant le nettoyage final: {e}")
            raise

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
            
        # Configuration initiale
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

    finally:
        # Nettoyage
        try:
            await components.cleanup()
            if settings.DEBUG and tracemalloc.is_tracing():
                tracemalloc.stop()
        except Exception as e:
            logger.error(f"Erreur pendant le nettoyage final: {e}")
        logger.info("Application arrêtée")

async def setup_environment():
    """Configure l'environnement."""
    try:
        # Configuration des bibliothèques numériques
        os.environ.update({
            "MKL_NUM_THREADS": str(settings.MKL_NUM_THREADS),
            "NUMEXPR_NUM_THREADS": str(settings.NUMEXPR_NUM_THREADS),
            "OMP_NUM_THREADS": str(settings.OMP_NUM_THREADS),
            "OPENBLAS_NUM_THREADS": str(settings.OPENBLAS_NUM_THREADS),
            "PYTORCH_CUDA_ALLOC_CONF": settings.PYTORCH_CUDA_ALLOC_CONF,
            "PYTORCH_ENABLE_MEM_EFFICIENT_OFFLOAD": str(settings.PYTORCH_ENABLE_MEM_EFFICIENT_OFFLOAD).lower(),
            "CUDA_MODULE_LOADING": settings.CUDA_MODULE_LOADING,
            "CUDA_VISIBLE_DEVICES": str(settings.CUDA_VISIBLE_DEVICES)
        })
        
        # Création des répertoires
        dirs = [
            "offload_folder",
            "static",
            "documents",
            "model_cache",
            "logs",
            "data",
            "temp",
            str(settings.MODELS_DIR)
        ]
        
        for dir_name in dirs:
            path = Path(dir_name)
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o755)
            
        logger.info("Environnement configuré avec succès")
            
    except Exception as e:
        logger.error(f"Erreur configuration environnement: {e}")
        raise

async def periodic_sync():
    """Tâche périodique de synchronisation."""
    while True:
        try:
            await asyncio.sleep(settings.GOOGLE_DRIVE_SYNC_INTERVAL)
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
    
    if isinstance(exc, RuntimeError) and "CUDA" in str(exc):
        logger.error(f"Erreur CUDA [{error_id}]: {exc}", exc_info=True)
        if hasattr(components, 'cuda_manager'):
            await components.cuda_manager.cleanup_memory()
        status_code = 503
        detail = "Ressources GPU temporairement indisponibles"
    else:
        logger.error(f"Erreur [{error_id}]: {exc}", exc_info=True)
        status_code = 500
        detail = str(exc)

    metrics.increment_counter("errors")
    
    return CustomJSONResponse(
        status_code=status_code,
        content={
            "error_id": error_id,
            "detail": detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def shutdown():
    """Arrête proprement l'application."""
    try:
        logger.info("Début de l'arrêt de l'application...")
        
        # Nettoyage des composants
        await components.cleanup()
        
        # Attente courte pour laisser le temps aux connexions de se fermer
        await asyncio.sleep(1)
        
        logger.info("Application arrêtée proprement")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt: {e}")
    finally:
        # Force l'arrêt si nécessaire
        sys.exit(0)

def signal_handler(signum, frame):
    """Gère les signaux d'interruption."""
    logger.info(f"Signal {signum} reçu, arrêt en cours...")
    asyncio.run(shutdown())

if __name__ == "__main__":
    try:
        # Configuration des handlers de signaux
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Configuration uvicorn
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
        logger.info("Interruption manuelle détectée")
        asyncio.run(shutdown())
    except Exception as e:
        logger.critical(f"Erreur fatale: {e}")
        asyncio.run(shutdown())
        sys.exit(1)