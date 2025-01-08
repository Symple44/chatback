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
from typing import Dict, Optional, List

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
from core.llm.cuda_manager import CUDAManager
from core.storage.google_drive import GoogleDriveManager
from core.llm.model_loader import ModelType

logger = get_logger("main")

class CustomJSONEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, ModelType):
            return obj.value
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
                from core.database.manager import DatabaseManager
                from core.database.base import get_session_manager

                db_session_manager = get_session_manager(settings.get_database_url())
                db_manager = DatabaseManager(db_session_manager.session_factory)
                self._components["db_manager"] = db_manager
                self._components["db"] = db_session_manager
                logger.info("Base de données initialisée")

                # Session Manager
                from core.database.session_manager import SessionManager
                self._components["session_manager"] = SessionManager(settings.get_database_url())
                logger.info("Session manager initialisé")

                # Cache Redis
                from core.cache import RedisCache
                cache = RedisCache()
                await cache.initialize()
                self._components["cache"] = cache
                logger.info("Cache Redis initialisé")

                # Elasticsearch
                from core.vectorstore import ElasticsearchClient
                es_client = ElasticsearchClient()
                await es_client.initialize()
                self._components["es_client"] = es_client
                logger.info("Elasticsearch initialisé")
                
                # Auth Manager
                from core.llm.auth_manager import HuggingFaceAuthManager
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
                from core.llm.tokenizer_manager import TokenizerManager
                tokenizer_manager = TokenizerManager()
                await tokenizer_manager.initialize()
                self._components["tokenizer_manager"] = tokenizer_manager
                logger.info("Tokenizer Manager initialisé")

                # Initialisation du model loader avant tout
                self.model_loader = ModelLoader(self.cuda_manager, self.tokenizer_manager)
                await self.model_loader.initialize()
                self._components["model_loader"] = self.model_loader

                # Initialisation du model manager
                model_manager = ModelManager(self.cuda_manager, self.tokenizer_manager)
                await model_manager.initialize()
                self._components["model_manager"] = model_manager

                # Initialisation de l'embedding manager avec le model manager
                embedding_manager = EmbeddingManager()
                await embedding_manager.initialize(model_manager)
                self._components["embedding_manager"] = embedding_manager

                # Initialisation du summarizer avec le model manager
                summarizer = DocumentSummarizer()
                await summarizer.initialize(model_manager)
                self._components["summarizer"] = summarizer

                # Model Inference (utilise les composants déjà initialisés)
                from core.llm.model import ModelInference
                model = ModelInference()
                await model.initialize(self)  # Passage de self pour accéder aux composants
                self._components["model"] = model
                logger.info("Model Inference initialisé")

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

    async def sync_drive_documents(self):
        """Synchronise les documents."""
        try:
            downloaded_files = set()
            if "drive_manager" in self._components:
                downloaded_files = await self._components["drive_manager"].sync_drive_folder(
                    settings.GOOGLE_DRIVE_FOLDER_ID,
                    save_path="documents"
                )

            indexed_count = 0
            for app_dir in Path("documents").iterdir():
                if not app_dir.is_dir():
                    continue

                app_name = app_dir.name
                logger.info(f"Traitement de l'application: {app_name}")

                for pdf_path in app_dir.glob("*.pdf"):
                    try:
                        if str(pdf_path) not in downloaded_files:
                            success = await self._components["pdf_processor"].index_pdf(
                                str(pdf_path),
                                metadata={
                                    "application": app_name,
                                    "sync_date": datetime.utcnow().isoformat(),
                                    "source": "local_filesystem"
                                }
                            )
                            if success:
                                indexed_count += 1
                                logger.info(f"Document indexé: {app_name}/{pdf_path.name}")
                    except Exception as e:
                        logger.error(f"Erreur indexation {pdf_path}: {e}")

            logger.info(f"Indexation terminée: {indexed_count} documents indexés")
            return indexed_count

        except Exception as e:
            logger.error(f"Erreur synchronisation: {e}")
            return 0

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

async def setup_environment():
    """Configure l'environnement complet avant le démarrage."""
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
        REQUIRED_DIRS = [
            "offload_folder",
            "static",
            "documents",
            "model_cache",
            "logs",
            "data",
            "temp",
            str(settings.MODELS_DIR)
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
        from api.routes.chat_routes import router as chat_router
        from api.routes.model_routes import router as model_router
        
        app.include_router(api_router)
        app.include_router(chat_router)
        app.include_router(model_router)
        
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
    """Gestionnaire d'erreurs global avec gestion spéciale CUDA."""
    error_id = str(uuid.uuid4())
    
    # Gestion spécifique selon le type d'erreur
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