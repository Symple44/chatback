from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

import asyncio
import os
import uvicorn
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import logging
from pathlib import Path

# Imports internes
from core.config import settings
from core.cache import RedisCache
from core.vectorstore.search import ElasticsearchClient
from core.llm.model import ModelInference
from core.document_processing.extractor import DocumentExtractor
from core.document_processing.pdf_processor import PDFProcessor
from core.storage.sync_manager import DocumentSyncManager
from core.utils.logger import get_logger, logger_manager
from core.utils.metrics import metrics
from core.utils.memory_manager import MemoryManager
from core.utils.system_optimizer import SystemOptimizer
from api.routes.router import router as api_router
from api.routes.docs import tags_metadata

# Configuration du logger
logger = get_logger("main")

class ComponentInitError(Exception):
    """Exception personnalisée pour les erreurs d'initialisation des composants."""
    pass

class ComponentManager:
    """Gestionnaire des composants de l'application."""

    def __init__(self):
        self.start_time = datetime.utcnow()
        self.initialized = False
        self._components = {}
        self._initializing = False
        self._init_lock = asyncio.Lock()
        self.system_optimizer = SystemOptimizer()

    async def _initialize_component(self, name: str, component: Any, timeout: int = 30) -> None:
        """Initialise un composant avec timeout."""
        try:
            if hasattr(component, 'initialize'):
                async with asyncio.timeout(timeout):
                    await component.initialize()
            self._components[name] = component
            logger.info(f"Composant {name} initialisé avec succès")
        except asyncio.TimeoutError:
            raise ComponentInitError(f"Timeout lors de l'initialisation de {name}")
        except Exception as e:
            raise ComponentInitError(f"Erreur lors de l'initialisation de {name}: {e}")

    async def initialize(self) -> None:
        """Initialise tous les composants de l'application."""
        async with self._init_lock:
            if self.initialized or self._initializing:
                return

            self._initializing = True
            try:
                logger.info("Début de l'initialisation des composants")

                # Création des répertoires nécessaires
                for dir_name in ["documents", "model_cache", "logs", "data", "temp"]:
                    os.makedirs(dir_name, exist_ok=True)

                # Initialisation dans un ordre spécifique
                init_sequence = [
                    ("cache", RedisCache()),
                    ("es_client", ElasticsearchClient()),
                    ("model", ModelInference()),
                    ("doc_extractor", DocumentExtractor()),
                ]

                # Première phase d'initialisation
                for name, component in init_sequence:
                    await self._initialize_component(name, component)

                # Initialisation des composants dépendants
                dependent_components = [
                    ("pdf_processor", PDFProcessor(self._components["es_client"])),
                    ("sync_manager", DocumentSyncManager(self._components["es_client"])),
                    ("memory_manager", MemoryManager())
                ]

                for name, component in dependent_components:
                    await self._initialize_component(name, component)

                # Démarrage du monitoring
                await self._components["memory_manager"].start_monitoring()

                self.initialized = True
                logger.info("Initialisation des composants terminée avec succès")

            except Exception as e:
                logger.critical(f"Erreur critique lors de l'initialisation: {e}")
                await self.cleanup()
                raise
            finally:
                self._initializing = False

    async def cleanup(self) -> None:
        """Nettoie les ressources des composants."""
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
            raise ComponentInitError(f"Erreurs lors du nettoyage: {', '.join(cleanup_errors)}")

    async def check_health(self) -> Dict[str, Any]:
        """Vérifie l'état de santé des composants."""
        health_status = {
            "status": "healthy",
            "components": {},
            "uptime": (datetime.utcnow() - self.start_time).total_seconds(),
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            for name, component in self._components.items():
                try:
                    if hasattr(component, 'health_check'):
                        status = await component.health_check()
                    else:
                        status = bool(component)
                    health_status["components"][name] = status
                except Exception as e:
                    logger.error(f"Erreur health check {name}: {e}")
                    health_status["components"][name] = False
                    health_status["status"] = "degraded"

            health_status["metrics"] = metrics.get_current_metrics()
            return health_status

        except Exception as e:
            logger.error(f"Erreur health check global: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def __getattr__(self, name: str) -> Any:
        """Permet d'accéder aux composants comme des attributs."""
        if name in self._components:
            return self._components[name]
        raise AttributeError(f"Composant {name} non trouvé")

# Instance globale des composants
components = ComponentManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    try:
        logger.info("Initialisation des services de base")
        await logger_manager.initialize()
        await metrics.initialize()
        logger.info("Initialisation des composants")
        await components.initialize()
        yield
    finally:
        logger.info("Nettoyage des composants")
        await components.cleanup()

# Configuration de l'application FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    description="API de chat avec support vectoriel et traitement documentaire",
    version=settings.VERSION,
    lifespan=lifespan,
    openapi_tags=tags_metadata,
    docs_url=None,
    redoc_url=None
)

# Configuration des middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Fichiers statiques
# Création du répertoire static s'il n'existe pas
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
swagger_dir = static_dir / "swagger"
swagger_dir.mkdir(exist_ok=True)

# Mount static files only if directory exists
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    logger.warning("Répertoire 'static' non trouvé - les fichiers statiques ne seront pas servis")

# Routes API
app.include_router(api_router)

# Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
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

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Gestionnaire pour les erreurs de validation."""
    error_id = str(uuid.uuid4())
    logger.warning(f"Erreur de validation [{error_id}]: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error_id": error_id,
            "detail": str(exc),
            "type": "ValidationError",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Documentation personnalisée
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Page de documentation personnalisée."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{settings.APP_NAME} - Documentation API",
        swagger_js_url="/static/swagger/js/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger/css/swagger-ui.css",
    )

# WebSocket Manager
class WebSocketManager:
    def __init__(self):
        self.active_connections = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: str) -> None:
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
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
    except Exception as e:
        logger.error(f"Erreur WebSocket: {e}")
    finally:
        await ws_manager.disconnect(websocket)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware pour ajouter le temps de traitement dans les headers."""
    start_time = datetime.utcnow()
    response = await call_next(request)
    process_time = (datetime.utcnow() - start_time).total_seconds()

    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-API-Version"] = settings.VERSION

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
        logger.info("Démarrage du serveur Uvicorn")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=settings.PORT,
            reload=settings.DEBUG,
            workers=settings.WORKERS,
            log_level=settings.LOG_LEVEL.lower(),
            loop="uvloop",
            timeout_keep_alive=settings.KEEPALIVE
        )
    except KeyboardInterrupt:
        logger.warning("Interruption manuelle détectée")
