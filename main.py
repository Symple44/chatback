from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
import time
import psutil
from typing import Optional

# Imports internes
from core.config import settings
from core.database.manager import DatabaseManager
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
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            logger.info(f"Nouvelle connexion WebSocket: {websocket.client}")
        except Exception as e:
            logger.error(f"Erreur lors de la connexion: {e}")
            raise

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
            logger.info(f"Connexion WebSocket fermée: {websocket.client}")
        except ValueError:
            pass
        except Exception as e:
            logger.error(f"Erreur lors de la déconnexion: {e}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Erreur lors du broadcast: {e}")

class ComponentManager:
    def __init__(self):
        """Initialise le gestionnaire de composants."""
        self.start_time = datetime.utcnow()
        self.initialized = False

    async def initialize(self):
        """Initialise tous les composants."""
        try:
            if self.initialized:
                return
                
            logger.info("Initialisation des composants...")
            
            # Création des répertoires nécessaires
            for dir_path in ["documents", "model_cache", "logs"]:
                os.makedirs(dir_path, exist_ok=True)
            
            # Initialisation de la base de données en premier
            self.db_manager = DatabaseManager(settings.DATABASE_URL)
            await self.db_manager.initialize_tables()
            
            # Initialisation du cache Redis
            self.cache = RedisCache()
            
            # Initialisation d'Elasticsearch
            self.es_client = ElasticsearchClient()
            
            # Initialisation du modèle
            self.model = ModelInference()
            
            # Initialisation du gestionnaire de sessions après la BDD
            from core.session_manager import SessionManager
            self.session_manager = SessionManager(settings.DATABASE_URL)
            await self.session_manager.initialize()
            
            # Initialisation de l'extracteur de documents
            self.doc_extractor = DocumentExtractor()
            
            # Initialisation des gestionnaires de stockage
            if os.path.exists(settings.GOOGLE_DRIVE_CREDENTIALS_PATH):
                self.drive_manager = GoogleDriveManager(
                    settings.GOOGLE_DRIVE_CREDENTIALS_PATH
                )
                logger.info("Google Drive initialisé")
            
            self.sync_manager = DocumentSyncManager(
                self.es_client,
                settings.DOCUMENTS_DIR,
                settings.SYNC_STATUS_FILE
            )
            
            # Initialisation des utilitaires
            self.memory_manager = MemoryManager()
            self.system_optimizer = SystemOptimizer()
            
            # Démarrage du monitoring mémoire
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
                'db_manager', 'cache', 'es_client', 'model',
                'session_manager', 'drive_manager', 'sync_manager',
                'memory_manager'
            ]
            
            for component_name in components_to_cleanup:
                component = getattr(self, component_name, None)
                if component and hasattr(component, 'cleanup'):
                    await component.cleanup()
            
            logger.info("Nettoyage des ressources terminé")
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")

    async def check_health(self) -> dict:
        """Vérifie l'état des services."""
        try:
            status = {
                'status': 'healthy',
                'components': {},
                'uptime': (datetime.utcnow() - self.start_time).total_seconds()
            }
            
            components_to_check = [
                'db_manager', 'cache', 'es_client', 'model',
                'session_manager', 'drive_manager', 'sync_manager'
            ]
            
            for component_name in components_to_check:
                component = getattr(self, component_name, None)
                if component and hasattr(component, 'check_health'):
                    status['components'][component_name] = await component.check_health()
                
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
    description="Assistant documentaire avec support streaming",
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    await components.initialize()
    yield
    await components.cleanup()

app.router.lifespan_context = lifespan

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Réception du message
            data = await websocket.receive_text()
            logger.info(f"Message reçu: {data}")
            
            try:
                # Parse du message
                message_data = json.loads(data)
                
                # Traitement du message
                if message_data.get('type') == 'user':
                    # Récupération du contenu
                    query = message_data.get('content')
                    
                    # Génération de la réponse avec le modèle
                    response_text = await components.model.generate_response(
                        query=query,
                        context_docs=[],  # Ajoutez le contexte si nécessaire
                        language="fr"
                    )
                    
                    # Envoi de la réponse
                    response = {
                        "type": "assistant",
                        "content": response_text,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    logger.info(f"Envoi réponse: {response}")
                    await websocket.send_json(response)
                
            except json.JSONDecodeError as e:
                logger.error(f"Erreur parsing message: {e}")
            except Exception as e:
                logger.error(f"Erreur traitement message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "content": "Une erreur est survenue lors du traitement de votre message"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client déconnecté")
    except Exception as e:
        logger.error(f"Erreur WebSocket: {e}")
        manager.disconnect(websocket)


# Inclusion des routes API
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
