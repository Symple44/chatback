import sys
import os
from pathlib import Path

# Ajout du répertoire parent au PYTHONPATH
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

print("Démarrage du debug...")

try:
    import uvicorn
    from datetime import datetime
    from fastapi import FastAPI
    from core.config import settings
    print("Imports de base OK")

    # Test des imports des composants
    try:
        from core.utils.logger import get_logger
        print("Logger importé avec succès")
    except Exception as e:
        print(f"Erreur import logger: {str(e)}")
        traceback.print_exc()

    try:
        from core.utils.metrics import metrics
        print("Metrics importé avec succès")
    except Exception as e:
        print(f"Erreur import metrics: {str(e)}")
        traceback.print_exc()

import asyncio
import uvicorn
from datetime import datetime
from contextlib import asynccontextmanager
import logging
import uuid

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

# Configuration du logger
logger = get_logger("main")

# Création des répertoires nécessaires
REQUIRED_DIRS = ["static", "documents", "model_cache", "logs", "data", "temp"]
for dir_name in REQUIRED_DIRS:
    Path(dir_name).mkdir(parents=True, exist_ok=True)

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

            try:
                # 1. Optimisation système
                await self.system_optimizer.optimize()

                # 2. Base de données
                from core.database.base import get_session_manager
                self._components["db"] = get_session_manager(settings.DATABASE_URL)
                logger.info("Base de données initialisée")

                # 3. Cache Redis
                from core.cache import RedisCache
                cache = RedisCache()
                await cache.initialize()
                self._components["cache"] = cache
                logger.info("Cache Redis initialisé")

                # 4. Elasticsearch
                from core.vectorstore.search import ElasticsearchClient
                es_client = ElasticsearchClient()
                await es_client.initialize()
                self._components["es_client"] = es_client
                logger.info("Elasticsearch initialisé")

                # 5. Modèle LLM
                from core.llm.model import ModelInference
                model = ModelInference()
                self._components["model"] = model
                logger.info("Modèle LLM initialisé")

                # 6. Processeurs de documents
                from core.document_processing.extractor import DocumentExtractor
                from core.document_processing.pdf_processor import PDFProcessor
                doc_extractor = DocumentExtractor()
                pdf_processor = PDFProcessor(es_client)
                self._components["doc_extractor"] = doc_extractor
                self._components["pdf_processor"] = pdf_processor
                logger.info("Processeurs de documents initialisés")

                # 7. Gestionnaire de synchronisation
                from core.storage.sync_manager import DocumentSyncManager
                sync_manager = DocumentSyncManager(es_client)
                self._components["sync_manager"] = sync_manager
                logger.info("Gestionnaire de synchronisation initialisé")

                # 8. Gestionnaire de mémoire
                from core.utils.memory_manager import MemoryManager
                memory_manager = MemoryManager()
                await memory_manager.start_monitoring()
                self._components["memory_manager"] = memory_manager
                logger.info("Gestionnaire de mémoire initialisé")

                self.initialized = True
                logger.info("Initialisation des composants terminée avec succès")

            except Exception as e:
                logger.critical(f"Erreur critique lors de l'initialisation: {e}")
                await self.cleanup()
                raise
            finally:
                self._initializing = False

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

class WebSocketManager:
    def __init__(self):
        self.active_connections = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"Nouvelle connexion WebSocket (total: {len(self.active_connections)})")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"Déconnexion WebSocket (restant: {len(self.active_connections)})")

    async def broadcast(self, message: str):
        """Diffuse un message à tous les clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            await self.disconnect(conn)

# Instance du gestionnaire WebSocket
ws_manager = WebSocketManager()

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
    docs_url=None,
    redoc_url=None
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
    return JSONResponse(
        status_code=500,
        content={
            "error_id": error_id,
            "detail": "Une erreur interne est survenue",
            "type": type(exc).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket pour le streaming des réponses."""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                response = await components.model.generate_streaming_response(data)
                async for token in response:
                    await websocket.send_text(token)
            except Exception as e:
                logger.error(f"Erreur traitement WebSocket: {e}")
                await websocket.send_json({
                    "error": "Erreur de traitement",
                    "details": str(e)
                })
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Erreur WebSocket: {e}")
        await ws_manager.disconnect(websocket)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Page de documentation personnalisée."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{settings.APP_NAME} - Documentation API",
        swagger_js_url="/static/swagger/js/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger/css/swagger-ui.css",
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
            port=settings.PORT,
            reload=settings.DEBUG,
            workers=1,  # Un seul worker en développement
            log_level="info",
            log_config=log_config,
            timeout_keep_alive=settings.KEEPALIVE
        )

        # Démarrage du serveur
        print("Lancement du serveur Uvicorn...")
        server = uvicorn.Server(config)
        server.run()

    except KeyboardInterrupt:
        print("Interruption manuelle détectée")
        logger.warning("Interruption manuelle détectée")
    except Exception as e:
        print(f"Erreur lors du démarrage: {str(e)}")
        logger.critical(f"Erreur lors du démarrage: {str(e)}", exc_info=True)
        sys.exit(1)
