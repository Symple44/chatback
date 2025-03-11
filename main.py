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
import torch
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

# Importation de la nouvelle structure de cache
from core.cache import redis_cache, cache_manager, MemoryCache, FileCache, CacheManager
from core.vectorstore import ElasticsearchClient
from core.llm.auth_manager import HuggingFaceAuthManager
from core.llm.model_manager import ModelManager
from core.llm.model_loader import ModelLoader, ModelType
from core.llm.cuda_manager import CUDAManager
from core.llm.embedding_manager import EmbeddingManager 
from core.llm.summarizer import DocumentSummarizer
from core.llm.tokenizer_manager import TokenizerManager
from core.llm.model import ModelInference
from core.chat.processor_factory import ProcessorFactory
from core.search.search_manager import SearchManager

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
                db_session_manager = get_session_manager(settings.db.get_database_url())
                db_manager = DatabaseManager(db_session_manager.session_factory)
                self._components["db_manager"] = db_manager
                self._components["db"] = db_session_manager
                logger.info("Base de données initialisée")

                # Session Manager
                self._components["session_manager"] = SessionManager(settings.db.get_database_url())
                logger.info("Session manager initialisé")

                # Cache Redis - Utilisation de la nouvelle structure
                await redis_cache.initialize()
                self._components["redis_cache"] = redis_cache
                logger.info("Cache Redis initialisé")
                
                # Cache Manager multi-niveaux - Nouveau
                await cache_manager.initialize()
                self._components["cache_manager"] = cache_manager
                logger.info("Cache Manager multi-niveaux initialisé")
                
                # Caches spécialisés pour différents usages
                pdf_cache = CacheManager(redis_cache=redis_cache, namespace="pdf_extraction")
                await pdf_cache.initialize()
                self._components["pdf_cache"] = pdf_cache
                logger.info("Cache PDF initialisé")
                
                models_cache = CacheManager(redis_cache=redis_cache, namespace="models")
                await models_cache.initialize()
                self._components["models_cache"] = models_cache
                logger.info("Cache modèles initialisé")

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

                # Processor Factory
                processor_factory = ProcessorFactory()
                self._components["processor_factory"] = processor_factory
                logger.info("Processor Factory initialisé")

                # SearchManager
                search_manager = SearchManager(self)
                self._components["search_manager"] = search_manager
                logger.info("Search Manager initialisé")

                # Document Processing
                self._components["doc_extractor"] = DocumentExtractor()
                self._components["pdf_processor"] = PDFProcessor(es_client)
                logger.info("Processeurs de documents initialisés")
                
                # Initialisation du détecteur de tableaux IA
                if hasattr(settings, 'table_extraction') and \
                    hasattr(settings.table_extraction, 'AI_DETECTION') and \
                    getattr(getattr(settings.table_extraction, 'AI_DETECTION', None), 'ENABLED', False):
                    from core.document_processing.table_detection import TableDetectionModel
                    table_detector = TableDetectionModel(cuda_manager=self.cuda_manager)
                    await table_detector.initialize()
                    self._components["table_detector"] = table_detector
                    logger.info("Détecteur de tableaux par IA initialisé")
                    
                    # Initialisation du nouveau pipeline d'extraction de tableaux
                    from core.document_processing.table_extraction.pipeline import TableExtractionPipeline
                    table_extraction_pipeline = TableExtractionPipeline(
                        cache_manager=self._components.get("cache_manager"),
                        table_detector=table_detector
                    )
                    self._components["table_extraction_pipeline"] = table_extraction_pipeline
                    logger.info("Pipeline d'extraction de tableaux initialisé")
                
                # Google Drive (optionnel)
                if settings.document.GOOGLE_DRIVE_CREDENTIALS_PATH:
                    drive_manager = GoogleDriveManager(
                        credentials_path=settings.document.GOOGLE_DRIVE_CREDENTIALS_PATH
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
                    settings.document.GOOGLE_DRIVE_FOLDER_ID,
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
                    except Exception as e:
                        logger.error(f"Erreur indexation {pdf_path}: {e}")

            logger.info(f"Indexation terminée: {indexed_count} documents indexés")
            return indexed_count

        except Exception as e:
            logger.error(f"Erreur synchronisation: {e}")
            return 0
        
    async def cleanup(self):
        """Nettoie proprement toutes les ressources."""
        try:
            logger.info("Début du nettoyage des composants...")
            
            # Ordre explicite de nettoyage - du plus haut niveau au plus bas niveau
            cleanup_order = [
                "search_manager", "doc_extractor", "pdf_processor",
                "table_extraction_pipeline", "table_detector",
                "model", "summarizer", "embedding_manager", "model_manager", 
                "model_loader", "tokenizer_manager", "cuda_manager", "auth_manager", 
                "es_client", "pdf_cache", "models_cache", "cache_manager", "redis_cache", 
                "db", "session_manager"
            ]
            
            # Nettoyer les composants dans l'ordre spécifié
            for name in cleanup_order:
                if name in self._components and hasattr(self._components[name], 'cleanup'):
                    try:
                        component = self._components[name]
                        await component.cleanup()
                        logger.info(f"Composant {name} nettoyé avec succès")
                    except Exception as e:
                        logger.error(f"Erreur lors du nettoyage de {name}: {e}")

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
                
        if settings.document.GOOGLE_DRIVE_SYNC_INTERVAL:
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
        # Utilisation directe de la configuration hardware depuis settings
        # Plus besoin d'importer get_hardware_config
        
        # Récupérer les configurations CPU et CUDA directement
        thread_config = settings.hardware.thread_config
        cuda_config = settings.hardware.cuda
        
        # Configuration des bibliothèques numériques
        os.environ.update({
            # Configurations CPU - maintenant accessibles via settings.hardware.thread_config
            "MKL_NUM_THREADS": str(thread_config.get("workers", settings.hardware.thread_config.get("mkl_num_threads", 16))),
            "NUMEXPR_NUM_THREADS": str(thread_config.get("workers", settings.hardware.thread_config.get("numexpr_num_threads", 16))),
            "OMP_NUM_THREADS": str(thread_config.get("inference_threads", settings.hardware.thread_config.get("omp_num_threads", 16))),
            "OPENBLAS_NUM_THREADS": str(thread_config.get("inference_threads", settings.hardware.thread_config.get("openblas_num_threads", 16))),
            
            # Configurations CUDA - maintenant accessibles via settings.hardware.cuda
            "PYTORCH_CUDA_ALLOC_CONF": f"max_split_size_mb:{settings.hardware.cuda.max_split_size_mb},garbage_collection_threshold:{settings.hardware.cuda.gc_threshold}",
            "PYTORCH_ENABLE_MEM_EFFICIENT_OFFLOAD": str(settings.hardware.cuda.efficient_offload).lower(),
            "CUDA_MODULE_LOADING": settings.hardware.cuda.module_loading,
            "CUDA_VISIBLE_DEVICES": str(settings.hardware.cuda.device_id)
        })
        
        # Log de la configuration matérielle
        hardware = settings.hardware.detector.detect_hardware()
        cpu_info = hardware['cpu']
        gpu_info = hardware['gpu']
        
        # Logs de configuration matérielle
        logger.info(f"Configuration matérielle chargée:")
        logger.info(f"CPU: {cpu_info['model']}")
        logger.info(f"Cores physiques: {cpu_info['physical_cores']}")
        logger.info(f"Cores logiques: {cpu_info['logical_cores']}")
        logger.info(f"Architecture CPU: {cpu_info['architecture']}")
        logger.info(f"RAM totale: {cpu_info['ram_total_gb']} Go")

        logger.info(f"GPU: {gpu_info['name']}")
        logger.info(f"CUDA disponible: {gpu_info['available']}")
        logger.info(f"VRAM: {gpu_info['vram_gb']} Go")
        logger.info(f"Compute Capability: {gpu_info['compute_capability']}")
        
        # Création des répertoires
        dirs = [
            cuda_config.offload_folder,
            "static",
            "documents",
            "model_cache",
            "logs",
            "data",
            "temp",
            str(settings.MODELS_DIR)  # Chemin des modèles maintenant dans settings.models
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
            await asyncio.sleep(settings.document.GOOGLE_DRIVE_SYNC_INTERVAL)
            if hasattr(components, 'drive_manager'):
                downloaded_files = await components.drive_manager.sync_drive_folder(
                    settings.document.GOOGLE_DRIVE_FOLDER_ID,
                    save_path="documents"
                )
                if downloaded_files:
                    logger.info(f"Synchronisation terminée: {len(downloaded_files)} fichiers téléchargés")
        except Exception as e:
            logger.error(f"Erreur sync périodique: {e}")
            await asyncio.sleep(60)  # Attendre une minute avant de réessayer

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
    allow_origins=settings.security.CORS_ORIGINS,
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
        
        # Import ici pour éviter les problèmes de cycle d'import
        import torch
        
        # Nettoyage des composants
        await components.cleanup()
        
        # Nettoyage CUDA si disponible
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass
        
        # Attente courte pour laisser le temps aux connexions de se fermer
        await asyncio.sleep(1)
        
        logger.info("Application arrêtée proprement")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt: {e}")
    finally:
        # Force l'arrêt si nécessaire
        sys.exit(0)

def signal_handler(signum, frame):
    """Gère les signaux d'interruption avec une meilleure gestion des tâches."""
    logger.info(f"Signal {signum} reçu, arrêt en cours...")
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Création de la tâche sans sys.exit() direct
            shutdown_task = loop.create_task(components.cleanup())
            
            # Ajouter un callback pour sortir proprement quand la tâche est terminée
            def exit_after_cleanup(_):
                try:
                    # Sortie différée pour permettre aux tâches de se terminer
                    loop.call_later(0.5, sys.exit, 0)
                except:
                    pass
                
            shutdown_task.add_done_callback(exit_after_cleanup)
        else:
            loop.run_until_complete(components.cleanup())
            sys.exit(0)
    except Exception as e:
        logger.error(f"Erreur dans le gestionnaire de signal: {e}")
        sys.exit(1)

async def safe_shutdown():
    """Version sécurisée de l'arrêt en cas de problème critique."""
    try:
        logger.info("Début de l'arrêt de l'application...")
        
        # Désactiver CUDA d'abord pour éviter les erreurs
        if torch.cuda.is_available():
            try:
                # Forcer la libération de la mémoire CUDA
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                logger.warning("Impossible de vider correctement la mémoire CUDA")
        
        # Nettoyage minimal des composants essentiels
        if hasattr(components, '_components'):
            for name in ['model', 'cuda_manager']:
                if name in components._components and hasattr(components._components[name], 'cleanup'):
                    try:
                        await components._components[name].cleanup()
                    except Exception as e:
                        logger.warning(f"Erreur nettoyage sécurisé {name}: {e}")
        
        # Attente courte
        await asyncio.sleep(1)
        
        logger.info("Application arrêtée en mode sécurisé")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt sécurisé: {e}")
    finally:
        # Force l'arrêt
        sys.exit(0)

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
            workers=1 if settings.DEBUG else settings.server.WORKERS,
            log_level="info",
            loop="uvloop",
            timeout_keep_alive=settings.server.KEEPALIVE,
            limit_max_requests=settings.server.MAX_REQUESTS,
            limit_concurrency=settings.server.MAX_CONCURRENT_REQUESTS,
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