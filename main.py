from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import json
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
import time
import psutil
from typing import Optional, List, Dict

# Imports internes
from core.config import settings
from core.database.base import get_session_manager, cleanup_database
from core.cache import RedisCache
from core.vectorstore.search import ElasticsearchClient
from core.llm.model import ModelInference
from core.document_processing.extractor import DocumentExtractor
from core.storage.sync_manager import DocumentSyncManager
from core.storage.google_drive import GoogleDriveManager
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.utils.memory_manager import MemoryManager
from core.utils.system_optimizer import SystemOptimizer
from api.routes.router import router as api_router

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parent

# Configuration du logging
logger = get_logger("main")

class ConnectionManager:
    def __init__(self):
        """Gestionnaire des connexions WebSocket."""
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Établit une nouvelle connexion WebSocket."""
        try:
            await websocket.accept()
            async with self._lock:
                self.active_connections.append(websocket)
            logger.info(f"Nouvelle connexion WebSocket: {websocket.client}")
            metrics.increment_counter("websocket_connections")
        except Exception as e:
            logger.error(f"Erreur lors de la connexion: {e}")
            metrics.increment_counter("websocket_connection_errors")
            raise

    def disconnect(self, websocket: WebSocket):
        """Ferme une connexion WebSocket."""
        try:
            self.active_connections.remove(websocket)
            logger.info(f"Connexion WebSocket fermée: {websocket.client}")
            metrics.increment_counter("websocket_disconnections")
        except ValueError:
            pass
        except Exception as e:
            logger.error(f"Erreur lors de la déconnexion: {e}")

    async def broadcast(self, message: str):
        """Diffuse un message à tous les clients connectés."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"Erreur lors du broadcast: {e}")
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

class ComponentManager:
    def __init__(self):
        """Initialise le gestionnaire de composants."""
        self.start_time = datetime.utcnow()
        self.initialized = False
        self.system_optimizer = SystemOptimizer()

    async def initialize(self):
        """Initialise tous les composants."""
        try:
            if self.initialized:
                return
                
            logger.info("Initialisation des composants...")
            
            # Optimisation système
            await self.system_optimizer.optimize()
            
            # Création des répertoires nécessaires
            for dir_path in ["documents", "model_cache", "logs"]:
                os.makedirs(dir_path, exist_ok=True)
            
            # Base de données
            self.db = get_session_manager(settings.DATABASE_URL)
            await self.db.create_all()
            
            # Cache Redis
            self.cache = RedisCache()
            
            # Elasticsearch
            self.es_client = ElasticsearchClient()
            
            # Modèle LLM
            self.model = ModelInference()
            
            # Extracteur de documents
            self.doc_extractor = DocumentExtractor()
            
            # Google Drive
            if os.path.exists(settings.GOOGLE_DRIVE_CREDENTIALS_PATH):
                self.drive_manager = GoogleDriveManager(
                    settings.GOOGLE_DRIVE_CREDENTIALS_PATH
                )
                logger.info("Google Drive initialisé")
            
            # Gestionnaire de synchronisation
            self.sync_manager = DocumentSyncManager(
                self.es_client,
                settings.DOCUMENTS_DIR,
                settings.SYNC_STATUS_FILE
            )
            
            # Gestionnaire de mémoire
            self.memory_manager = MemoryManager()
            
            # Démarrage du monitoring
            asyncio.create_task(self.memory_manager.start_monitoring())
            
            self.initialized = True
            logger.info("Initialisation des composants terminée")
            
        except Exception as e:
            logger.critical(f"Erreur critique lors de l'initialisation: {e}")
            await self.cleanup()
            raise

    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            components_to_cleanup = [
                'db', 'cache', 'es_client', 'model',
                'drive_manager', 'sync_manager',
                'memory_manager'
            ]
            
            for component_name in components_to_cleanup:
                component = getattr(self, component_name, None)
                if component and hasattr(component, 'cleanup'):
                    await component.cleanup()
            
            logger.info("Nettoyage des ressources terminé")
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")

    async def check_health(self) -> Dict:
        """Vérifie l'état des services."""
        try:
            status = {
                'status': 'healthy',
                'components': {},
                'uptime': (datetime.utcnow() - self.start_time).total_seconds(),
                'system': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent
                }
            }
            
            # Vérification des composants
            components_to_check = [
                'db', 'cache', 'es_client', 'model',
                'drive_manager', 'sync_manager'
            ]
            
            for component_name in components_to_check:
                component = getattr(self, component_name, None)
                if component and hasattr(component, 'health_check'):
                    status['components'][component_name] = await component.health_check()
                
            return status
            
        except Exception as e:
            logger.error(f"Erreur health check: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

# Instances globales
components = ComponentManager()
manager = ConnectionManager()

# Application FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    description="Assistant documentaire avec support vectoriel",
    version=settings.VERSION
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket pour le streaming des réponses."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Message reçu: {data}")
            
            try:
                message_data = json.loads(data)
                
                if message_data.get('type') == 'user':
                    query = message_data.get('content')
                    
                    # Génération de la réponse avec streaming
                    async for token in components.model.generate_streaming_response(
                        query=query,
                        context_docs=[],
                        language="fr"
                    ):
                        response = {
                            "type": "assistant",
                            "content": token,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        await websocket.send_json(response)
                
            except json.JSONDecodeError as e:
                logger.error(f"Erreur parsing message: {e}")
            except Exception as e:
                logger.error(f"Erreur traitement message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "content": "Une erreur est survenue lors du traitement"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client déconnecté")
    except Exception as e:
        logger.error(f"Erreur WebSocket: {e}")
        manager.disconnect(websocket)

# Routes API
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        reload_dirs=[str(BASE_DIR)]
    )
