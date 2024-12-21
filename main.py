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
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid
import json

# Imports internes
from core.config import settings
from core.utils.logger import get_logger, logger_manager
from core.utils.metrics import metrics
from core.utils.system_optimizer import SystemOptimizer

logger = get_logger("main")

# Création du répertoire static s'il n'existe pas
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
swagger_dir = static_dir / "swagger"
swagger_dir.mkdir(exist_ok=True)

class DatabaseComponent:
    """Composant Base de données avec initialisation retardée"""
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._session_manager = None

    async def initialize(self):
        # Import retardé pour éviter la dépendance circulaire
        from core.database.base import DatabaseSessionManager
        self._session_manager = DatabaseSessionManager(self.database_url)
        await self._session_manager.initialize()

    def get_session_manager(self):
        return self._session_manager

    async def cleanup(self):
        if self._session_manager:
            await self._session_manager.close()

class ComponentManager:
    """Gestionnaire des composants avec initialisation séquentielle."""
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.initialized = False
        self._components = {}
        self._initializing = False
        self._init_lock = asyncio.Lock()
        self.system_optimizer = SystemOptimizer()

    def _create_initial_components(self) -> Dict:
        """Crée les instances de composants sans les initialiser."""
        from core.cache import RedisCache
        from core.vectorstore.search import ElasticsearchClient
        from core.llm.model import ModelInference
        from core.document_processing.extractor import DocumentExtractor
        from core.document_processing.pdf_processor import PDFProcessor
        from core.storage.sync_manager import DocumentSyncManager
        from core.utils.memory_manager import MemoryManager

        return {
            "db": DatabaseComponent(settings.DATABASE_URL),
            "cache": RedisCache(),
            "es_client": ElasticsearchClient(),
            "model": ModelInference(),
            "doc_extractor": DocumentExtractor(),
            "memory_manager": MemoryManager()
        }

    async def _initialize_component(self, name: str, component: Any, 
                                 dependencies: Optional[List[str]] = None) -> None:
        """Initialise un composant avec ses dépendances."""
        if dependencies:
            for dep in dependencies:
                if dep not in self._components or not getattr(self._components[dep], 'initialized', True):
                    raise RuntimeError(f"Dépendance {dep} non initialisée pour {name}")

        try:
            async with asyncio.timeout(30):  # Timeout de 30 secondes
                if hasattr(component, 'initialize'):
                    await component.initialize()
                self._components[name] = component
                logger.info(f"Composant {name} initialisé")
        except Exception as e:
            logger.error(f"Erreur initialisation {name}: {e}")
            raise

    async def initialize(self) -> None:
        """Initialise les composants dans le bon ordre."""
        async with self._init_lock:
            if self.initialized or self._initializing:
                return

            self._initializing = True
            try:
                # Créer les répertoires nécessaires
                for dir_name in ["documents", "model_cache", "logs", "data", "temp"]:
                    os.makedirs(dir_name, exist_ok=True)

                # Créer les instances de composants
                components = self._create_initial_components()

                # Ordre d'initialisation avec dépendances
                init_sequence = [
                    ("db", components["db"], []),
                    ("cache", components["cache"], []),
                    ("es_client", components["es_client"], []),
                    ("model", components["model"], ["cache"]),
                    ("doc_extractor", components["doc_extractor"], ["es_client"]),
                    ("memory_manager", components["memory_manager"], [])
                ]

                for name, component, deps in init_sequence:
                    await self._initialize_component(name, component, deps)

                # Composants dépendants
                pdf_processor = PDFProcessor(self._components["es_client"])
                sync_manager = DocumentSyncManager(self._components["es_client"])

                await self._initialize_component("pdf_processor", pdf_processor, ["es_client"])
                await self._initialize_component("sync_manager", sync_manager, ["es_client"])

                # Démarrage du monitoring
                await self._components["memory_manager"].start_monitoring()

                self.initialized = True
                logger.info("Initialisation des composants terminée")

            except Exception as e:
                logger.critical(f"Erreur critique lors de l'initialisation: {e}")
                await self.cleanup()
                raise
            finally:
                self._initializing = False

    async def cleanup(self) -> None:
        """Nettoie les ressources des composants."""
        for name, component in self._components.items():
            try:
                if hasattr(component, 'cleanup'):
                    await component.cleanup()
                logger.info(f"Composant {name} nettoyé")
            except Exception as e:
                logger.error(f"Erreur nettoyage {name}: {e}")

        self._components.clear()
        self.initialized = False

    def __getattr__(self, name: str) -> Any:
        """Accès aux composants comme attributs."""
        if name in self._components:
            return self._components[name]
        raise AttributeError(f"Composant {name} non trouvé")

# Instance globale du gestionnaire de composants
components = ComponentManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    try:
        logger.info("Initialisation des services")
        await logger_manager.initialize()
        await metrics.initialize()
        await components.initialize()
        
        # Import retardé des routes pour éviter les dépendances circulaires
        from api.routes.router import router as api_router
        app.include_router(api_router)
        
        yield
    finally:
        logger.info("Nettoyage des ressources")
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
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

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
