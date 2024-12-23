#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Ajout du répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent))

try:
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

    # Création des répertoires nécessaires
    REQUIRED_DIRS = ["static", "documents", "model_cache", "logs", "data", "temp"]
    for dir_name in REQUIRED_DIRS:
        Path(dir_name).mkdir(parents=True, exist_ok=True)

    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, UUID):
                return str(obj)
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
        """Gestionnaire des composants de l'application."""
        
        def __init__(self):
            self.start_time = datetime.utcnow()
            self.initialized = False
            self._components = {}
            self._initializing = False
            self._init_lock = asyncio.Lock()
            self.system_optimizer = SystemOptimizer()

        async def initialize(self):
            """Initialise tous les composants."""
            async with self._init_lock:
                if self.initialized or self._initializing:
                    return

                logger.info("Début de l'initialisation des composants")
                self._initializing = True
                components_initialized = []

                try:
                    # 1. Optimisation système
                    await self.system_optimizer.optimize()

                    # 2. Base de données
                    from core.database.base import get_session_manager
                    self._components["db"] = get_session_manager(settings.DATABASE_URL)
                    logger.info("Base de données initialisée")

                    # 3. Initialisation du DatabaseManager
                    from core.database.manager import DatabaseManager
                    self._components["db_manager"] = DatabaseManager(self._components["db"])
                    logger.info("Database manager initialisé")
                    
                    # 4. Initialisation du SessionManager
                    from core.session_manager import SessionManager
                    self._components["session_manager"] = SessionManager(settings.DATABASE_URL)
                    logger.info("Session manager initialisé")

                    # 5. Cache Redis
                    from core.cache import RedisCache
                    cache = RedisCache()
                    await cache.initialize()
                    self._components["cache"] = cache
                    logger.info("Cache Redis initialisé")

                    # 6. Elasticsearch
                    from core.vectorstore.search import ElasticsearchClient
                    es_client = ElasticsearchClient()
                    await es_client.initialize()
                    self._components["es_client"] = es_client
                    logger.info("Elasticsearch initialisé")

                    # 7. Modèle LLM
                    from core.llm.model import ModelInference
                    model = ModelInference()
                    self._components["model"] = model
                    logger.info("Modèle LLM initialisé")

                    # 8. Processeurs de documents
                    from core.document_processing.extractor import DocumentExtractor
                    from core.document_processing.pdf_processor import PDFProcessor
                    doc_extractor = DocumentExtractor()
                    pdf_processor = PDFProcessor(es_client)
                    self._components["doc_extractor"] = doc_extractor
                    self._components["pdf_processor"] = pdf_processor
                    logger.info("Processeurs de documents initialisés")
                    
                     # 9. Google Drive Manager
                    # Configuration plus résiliente de Google Drive
                    if os.path.exists(settings.GOOGLE_DRIVE_CREDENTIALS_PATH):
                        drive_manager = GoogleDriveManager(
                            credentials_path=settings.GOOGLE_DRIVE_CREDENTIALS_PATH
                        )
                        if await drive_manager.initialize():
                            self._components["drive_manager"] = drive_manager
                            components_initialized.append("drive_manager")
                            # Première synchronisation seulement si l'initialisation a réussi
                            await self.sync_drive_documents()
                            logger.info("Google Drive manager initialisé avec succès")
                        else:
                            logger.warning("Google Drive manager non initialisé - sync désactivée")
                    else:
                        logger.warning(f"Fichier credentials Google Drive non trouvé: {settings.GOOGLE_DRIVE_CREDENTIALS_PATH}")
    
                    self.initialized = True
                    logger.info("Initialisation des composants terminée avec succès")

                except Exception as e:
                    logger.critical(f"Erreur critique lors de l'initialisation: {e}")
                    # Nettoyage sélectif des composants initialisés
                    for component in components_initialized:
                        try:
                            if hasattr(self._components[component], 'cleanup'):
                                await self._components[component].cleanup()
                            logger.info(f"Composant {component} nettoyé")
                        except Exception as cleanup_error:
                            logger.error(f"Erreur nettoyage {component}: {cleanup_error}")
                    raise
                finally:
                    self._initializing = False
                    
        async def sync_drive_documents(self):
            """Synchronise les documents depuis Google Drive."""
            try:
                downloaded_files = await self._components["drive_manager"].sync_drive_folder(
                    settings.GOOGLE_DRIVE_FOLDER_ID,
                    save_path="documents"
                )
                
                # Indexer les nouveaux documents
                for file_path in downloaded_files:
                    await self._components["pdf_processor"].index_pdf(file_path)
                    
                logger.info(f"{len(downloaded_files)} documents synchronisés depuis Drive")
                
            except Exception as e:
                logger.error(f"Erreur synchronisation Drive: {e}")

        async def cleanup(self):
            """Nettoie les ressources."""
            cleanup_errors = []
            for name, component in self._components.items():
                try:
                    if hasattr(component, 'cleanup'):
                        await component.cleanup()
                    logger.info(f"Composant {name} nettoyé")
                except Exception as e:
                    cleanup_errors.append(f"{name}: {str(e)}")
                    logger.error(f"Erreur nettoyage {name}: {e}")

            self._components.clear()
            self.initialized = False

            if cleanup_errors:
                logger.error(f"Erreurs lors du nettoyage: {', '.join(cleanup_errors)}")

        def __getattr__(self, name):
            """Permet d'accéder aux composants comme des attributs."""
            if name in self._components:
                return self._components[name]
            raise AttributeError(f"Composant {name} non trouvé")

    # Instance globale des composants
    components = ComponentManager()
    
    async def start_background_tasks(app: FastAPI):
        async def sync_documents():
            await asyncio.sleep(settings.GOOGLE_DRIVE_SYNC_INTERVAL)
            while True:
                try:
                    logger.info("Début de la synchronisation des documents")
                    await components.sync_drive_documents()
                    logger.info("Synchronisation des documents terminée")
                except Exception as e:
                    logger.error(f"Erreur synchro documents: {e}")
                logger.info(f"Attente de {settings.GOOGLE_DRIVE_SYNC_INTERVAL} secondes avant la prochaine synchronisation")
                await asyncio.sleep(settings.GOOGLE_DRIVE_SYNC_INTERVAL)

        asyncio.create_task(sync_documents())
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Gestion du cycle de vie de l'application."""
        logger.info("Démarrage de l'application")
        try:
            # Services de base
            logger.info("Initialisation des services de base")
            await logger_manager.initialize()
            await metrics.initialize()

            # Composants
            logger.info("Initialisation des composants")
            await components.initialize()

            # Import et ajout des routes
            from api.routes.router import router as api_router
            app.include_router(api_router)
            
            # Tâches de fond
            await start_background_tasks(app)

            logger.info("Application prête")
            yield

        finally:
            logger.info("Arrêt de l'application")
            await components.cleanup()
            
    # Configuration de l'application FastAPI
    app = FastAPI(
        title=settings.APP_NAME,
        description="API de chat avec support vectoriel et traitement documentaire",
        version=settings.VERSION,
        lifespan=lifespan,
        default_response_class=CustomJSONResponse
        #docs_url=None,
        #redoc_url=None
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
    if Path("static").exists():
        app.mount("/static", StaticFiles(directory="static"), name="static")

    # Exception Handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Gestionnaire d'erreurs global."""
        error_id = str(uuid.uuid4())
        logger.error(f"Erreur non gérée [{error_id}]: {exc}", exc_info=True)
        return CustomJSONResponse(  
            status_code=500,
            content={
                "error_id": error_id,
                "detail": "Une erreur interne est survenue",
                "type": type(exc).__name__,
                "timestamp": datetime.utcnow()
            }
        )

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """Page de documentation personnalisée."""
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{settings.APP_NAME} - Documentation API",
            swagger_js_url="/static/swagger/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger/swagger-ui.css"
        )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Middleware pour ajouter le temps de traitement dans les headers."""
        start_time = datetime.utcnow()
        response = await call_next(request)
        process_time = (datetime.utcnow() - start_time).total_seconds()

        # Ajout des headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-API-Version"] = settings.VERSION

        # Log et métriques
        if not request.url.path.endswith(("/ping", "/health")):
            logger.info(
                f"Request: {request.method} {request.url.path} "
                f"Status: {response.status_code} "
                f"Duration: {process_time:.3f}s"
            )
            metrics.increment_counter("http_requests")
            if response.status_code >= 400:
                metrics.increment_counter("http_errors")

        return response

    if __name__ == "__main__":
        try:
            print("Démarrage de l'application...")
            logger.info("Démarrage de l'application...")

            # Configuration uvicorn
            log_config = uvicorn.config.LOGGING_CONFIG
            log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
            log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"

            # Configuration du serveur
            config = uvicorn.Config(
                "main:app",
                host="0.0.0.0",
                port=8000,
                reload=settings.DEBUG,
                workers=1,
                log_level="info",
                log_config=log_config,
                timeout_keep_alive=60
            )

            # Démarrage du serveur
            server = uvicorn.Server(config)
            server.run()

        except KeyboardInterrupt:
            print("Interruption manuelle détectée")
            logger.warning("Interruption manuelle détectée")
        except Exception as e:
            print(f"Erreur lors du démarrage: {str(e)}")
            logger.critical(f"Erreur lors du démarrage: {str(e)}", exc_info=True)
            sys.exit(1)

except Exception as e:
    print(f"Erreur critique au démarrage: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
