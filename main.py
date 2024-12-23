#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Configuration des variables d'environnement critiques au démarrage
def setup_environment():
    # Optimisations CUDA et GPU depuis les variables d'environnement
    cuda_env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
        "NVIDIA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
        "PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:2048"),
    }
    os.environ.update(cuda_env)

    # Configuration des threads depuis les variables d'environnement
    thread_env = {
        "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS", "16"),
        "NUMEXPR_NUM_THREADS": os.getenv("NUMEXPR_NUM_THREADS", "16"),
        "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", "16"),
        "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS", "16"),
    }
    os.environ.update(thread_env)

    # Configuration TensorFlow
    tf_env = {
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_ENABLE_ONEDNN_OPTS": "1",
        "TF_GPU_THREAD_MODE": "gpu_private",
        "TF_GPU_THREAD_COUNT": "1",
        "TF_XLA_FLAGS": "--xla_gpu_cuda_data_dir=/usr/local/cuda",
    }
    os.environ.update(tf_env)

# Setup initial de l'environnement
setup_environment()

# Ajout du répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent))

# Imports après la configuration de l'environnement
import torch
import torch.backends.cuda
import torch.backends.cudnn
import asyncio
import uvicorn
from datetime import datetime
from contextlib import asynccontextmanager
import logging
import uuid
import json

from fastapi import FastAPI, WebSocket, HTTPException, Request, WebSocketDisconnect
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

def setup_torch_config():
    """Configure PyTorch selon les variables d'environnement."""
    if torch.cuda.is_available():
        # Optimisations CUDA
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True

        # Configuration FP16 si activée
        if os.getenv("USE_FP16", "true").lower() == "true":
            torch.set_float32_matmul_precision('high')

        # Gestion mémoire CUDA
        if os.getenv("CUDA_MEMORY_FRACTION"):
            fraction = float(os.getenv("CUDA_MEMORY_FRACTION"))
            torch.cuda.set_per_process_memory_fraction(fraction)

class ComponentManager:
    """Gestionnaire des composants de l'application."""
    
    def __init__(self):
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
                self._components["model"] = model
                logger.info("Modèle LLM initialisé")

                # Document Processors
                from core.document_processing.extractor import DocumentExtractor
                from core.document_processing.pdf_processor import PDFProcessor
                self._components["doc_extractor"] = DocumentExtractor()
                self._components["pdf_processor"] = PDFProcessor(es_client)
                logger.info("Processeurs de documents initialisés")
                
                # Google Drive (optionnel)
                if os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH"):
                    drive_manager = GoogleDriveManager(
                        credentials_path=os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH")
                    )
                    if await drive_manager.initialize():
                        self._components["drive_manager"] = drive_manager
                        await self.sync_drive_documents()
                
                self.initialized = True
                logger.info("Tous les composants sont initialisés")

            except Exception as e:
                logger.critical(f"Erreur critique lors de l'initialisation: {e}")
                await self.cleanup()
                raise

    async def sync_drive_documents(self):
        """Synchronise les documents depuis Google Drive."""
        if "drive_manager" not in self._components:
            return

        try:
            downloaded_files = await self._components["drive_manager"].sync_drive_folder(
                os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
                save_path="documents"
            )
            
            for file_path in downloaded_files:
                await self._components["pdf_processor"].index_pdf(file_path)
                
        except Exception as e:
            logger.error(f"Erreur synchronisation Drive: {e}")

    async def cleanup(self):
        """Nettoie les ressources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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

async def startup_event():
    """Événements de démarrage de l'application."""
    setup_torch_config()
    await logger_manager.initialize()
    await metrics.initialize()
    await components.initialize()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    try:
        await startup_event()
        
        # Import des routes
        from api.routes.router import router as api_router
        app.include_router(api_router)
        
        # Démarrage des tâches de fond
        if os.getenv("GOOGLE_DRIVE_SYNC_INTERVAL"):
            asyncio.create_task(periodic_sync())
        
        yield
    finally:
        await components.cleanup()

async def periodic_sync():
    """Tâche périodique de synchronisation."""
    interval = int(os.getenv("GOOGLE_DRIVE_SYNC_INTERVAL", "3600"))
    while True:
        try:
            await components.sync_drive_documents()
        except Exception as e:
            logger.error(f"Erreur sync périodique: {e}")
        await asyncio.sleep(interval)

# Configuration FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
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

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Gestionnaire d'erreurs global."""
    error_id = str(uuid.uuid4())
    logger.error(f"Erreur [{error_id}]: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error_id": error_id,
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=int(os.getenv("PORT", "8000")),
            workers=int(os.getenv("WORKERS", "16")),
            limit_concurrency=int(os.getenv("MAX_CONCURRENT_REQUESTS", "100")),
            timeout_keep_alive=int(os.getenv("KEEPALIVE", "5")),
            log_level="info" if not settings.DEBUG else "debug"
        )
    except KeyboardInterrupt:
        logger.info("Arrêt manuel détecté")
    except Exception as e:
        logger.critical(f"Erreur fatale: {e}")
        sys.exit(1)
