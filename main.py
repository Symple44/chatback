# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
import asyncio
import os
import uvicorn
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import Optional, List, Dict

# Imports internes
from core.config import settings
from core.database.base import get_session_manager, cleanup_database
from core.cache import RedisCache
from core.vectorstore.search import ElasticsearchClient
from core.llm.model import ModelInference
from core.document_processing.extractor import DocumentExtractor
from core.storage.sync_manager import DocumentSyncManager
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.utils.memory_manager import MemoryManager
from core.utils.system_optimizer import SystemOptimizer
from api.routes.router import router as api_router
from api.routes.docs import tags_metadata

logger = get_logger("main")

class ComponentManager:
    """Gestionnaire des composants de l'application."""
    
    def __init__(self):
        """Initialise le gestionnaire de composants."""
        self.start_time = datetime.utcnow()
        self.initialized = False
        self.system_optimizer = SystemOptimizer()
        self._components = {}

    async def initialize(self):
        """Initialise tous les composants de l'application."""
        if self.initialized:
            logger.warning("Composants déjà initialisés")
            return

        try:
            logger.info("Démarrage de l'initialisation des composants")

            # Optimisation système
            await self.system_optimizer.optimize()

            # Création des répertoires nécessaires
            for dir_name in ["documents", "model_cache", "logs", "data"]:
                os.makedirs(dir_name, exist_ok=True)

            # Initialisation des composants principaux
            components_to_init = {
                "db": get_session_manager(settings.DATABASE_URL),
                "cache": RedisCache(),
                "es_client": ElasticsearchClient(),
                "model": ModelInference(),
                "doc_extractor": DocumentExtractor(),
                "sync_manager": DocumentSyncManager(),
                "memory_manager": MemoryManager()
            }

            # Initialisation séquentielle avec gestion d'erreurs
            for name, component in components_to_init.items():
                try:
                    if hasattr(component, 'initialize'):
                        await component.initialize()
                    self._components[name] = component
                    logger.info(f"Composant {name} initialisé")
                except Exception as e:
                    logger.error(f"Erreur initialisation {name}: {e}")
                    await self.cleanup()
                    raise

            # Démarrage du monitoring
            await self._components["memory_manager"].start_monitoring()

            self.initialized = True
            logger.info("Initialisation des composants terminée avec succès")

        except Exception as e:
            logger.critical(f"Erreur critique lors de l'initialisation: {e}")
            await self.cleanup()
            raise

    async def cleanup(self):
        """Nettoie les ressources des composants."""
        logger.info("Démarrage du nettoyage des composants")
        
        for name, component in self._components.items():
            try:
                if hasattr(component, 'cleanup'):
                    await component.cleanup()
                logger.info(f"Composant {name} nettoyé")
            except Exception as e:
                logger.error(f"Erreur nettoyage {name}: {e}")

        self._components.clear()
        self.initialized = False
        logger.info("Nettoyage des composants terminé")

    async def check_health(self) -> Dict:
        """Vérifie l'état de santé des composants."""
        health_status = {
            "status": "healthy",
            "components": {},
            "uptime": (datetime.utcnow() - self.start_time).total_seconds(),
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            # Vérification de chaque composant
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

            # Ajout des métriques système
            health_status["metrics"] = metrics.get_current_metrics()
            
            return health_status

        except Exception as e:
            logger.error(f"Erreur health check global: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def __getattr__(self, name):
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
        await components.initialize()
        yield
    finally:
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

# Documentation personnalisée
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{settings.APP_NAME} - Documentation API",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

# Routes API
app.include_router(api_router)

# WebSocket pour le streaming
class ConnectionManager:
    """Gestionnaire des connexions WebSocket."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"Nouvelle connexion WebSocket: {len(self.active_connections)} active(s)")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.remove(websocket)
        logger.info(f"Connexion WebSocket fermée: {len(self.active_connections)} active(s)")

    async def broadcast(self, message: str):
        """Diffuse un message à tous les clients connectés."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"Erreur broadcast WebSocket: {e}")
                disconnected.append(connection)

        # Nettoyage des connexions fermées
        for conn in disconnected:
            await self.disconnect(conn)

# Instance du gestionnaire de connexions
manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket pour le streaming des réponses."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                # Traitement du message
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
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Erreur WebSocket: {e}")
        await manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        loop="uvloop",
        timeout_keep_alive=settings.KEEPALIVE
    )
