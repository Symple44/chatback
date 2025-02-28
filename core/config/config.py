# core/config/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import validator, Field
from pathlib import Path
import os
import logging
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Configuration du logger
logger = logging.getLogger(__name__)

# Chemin absolu vers le fichier .env
env_path = Path(__file__).resolve().parent.parent.parent / ".env"

# Chargement explicite du fichier .env avec indication de succès/échec
if env_path.exists():
    logger.info(f"Chargement du fichier .env: {env_path}")
    load_dotenv(dotenv_path=env_path, override=True)
    
    # Vérification que certaines variables critiques sont bien chargées
    critical_vars = ["ELASTICSEARCH_HOST", "DATABASE_URL", "REDIS_HOST"]
    for var in critical_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"Variable chargée: {var}={value[:10]}...")
        else:
            logger.warning(f"Variable non trouvée: {var}")
else:
    logger.warning(f"Fichier .env non trouvé: {env_path}")

class Settings(BaseSettings):
    """Configuration unifiée de l'application."""
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    LOG_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "logs")
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data")
    CACHE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "cache")
    MODELS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "models")
    TEMP_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "temp")
    
    # Application settings - utilisation de Field avec default_factory pour évaluation tardive
    APP_NAME: str = Field(default_factory=lambda: os.getenv("APP_NAME", "AI Chat Assistant"))
    VERSION: str = Field(default_factory=lambda: os.getenv("VERSION", "1.0.0"))
    BUILD_DATE: str = Field(default_factory=lambda: os.getenv("BUILD_DATE", "2025"))
    CONTACT_EMAIL: str = Field(default_factory=lambda: os.getenv("CONTACT_EMAIL", "contact@symple.fr"))
    DEBUG: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    ENV: str = Field(default_factory=lambda: os.getenv("ENV", "production"))
    PORT: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    
    # Formats
    SUPPORTED_FORMATS: List[str] = ['.pdf', '.docx', '.txt', '.md', '.csv']
    
    # Configurations spécialisées
    db: Optional[Any] = None
    cache: Optional[Any] = None
    server: Optional[Any] = None
    security: Optional[Any] = None
    hardware: Optional[Any] = None
    models: Optional[Any] = None
    search: Optional[Any] = None
    document: Optional[Any] = None
    chat: Optional[Any] = None
    
    model_config = SettingsConfigDict(
        env_file = str(env_path),
        case_sensitive = True,
        extra = 'ignore',
        env_file_encoding = 'utf-8'
    )

    @validator('PORT')
    def validate_port(cls, port):
        """Validate that port is within valid range."""
        if not 1 <= port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return port

    @validator('SUPPORTED_FORMATS')
    def validate_formats(cls, formats):
        """Validate file formats."""
        valid_formats = {'.pdf', '.docx', '.txt', '.md', '.csv'}
        invalid_formats = set(formats) - valid_formats
        if invalid_formats:
            raise ValueError(f"Invalid formats: {invalid_formats}. Supported formats are: {valid_formats}")
        return formats

    def __init__(self, **data):
        # Créer les répertoires requis avant l'initialisation de Pydantic
        required_dirs = [
            Path(__file__).resolve().parent.parent.parent / "logs",
            Path(__file__).resolve().parent.parent.parent / "data",
            Path(__file__).resolve().parent.parent.parent / "cache",
            Path(__file__).resolve().parent.parent.parent / "models",
            Path(__file__).resolve().parent.parent.parent / "temp"
        ]
        
        for path in required_dirs:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialiser Pydantic
        super().__init__(**data)
        
        # Importer les configurations spécialisées ici pour éviter les problèmes d'importation circulaire
        from .database import DatabaseConfig
        from .cache import CacheConfig
        from .server import ServerConfig
        from .security import SecurityConfig
        from .hardware import HardwareConfig
        from .models import ModelsConfig
        from .search import SearchConfig
        from .document import DocumentConfig
        from .chat import ChatConfig
        
        # Créer les configurations spécialisées APRÈS l'initialisation de Pydantic
        try:
            self.db = DatabaseConfig()
            logger.info("Configuration de base de données chargée")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration DB: {e}")
            
        try:
            self.cache = CacheConfig()
            logger.info("Configuration du cache chargée")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration Cache: {e}")
        
        try:
            self.server = ServerConfig()
            logger.info("Configuration du serveur chargée")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration Server: {e}")
            
        try:
            self.security = SecurityConfig()
            logger.info("Configuration de sécurité chargée")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration Security: {e}")
            
        try:
            self.hardware = HardwareConfig()
            logger.info("Configuration hardware chargée")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration Hardware: {e}")
            
        try:
            self.models = ModelsConfig()
            logger.info("Configuration des modèles chargée")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration Models: {e}")
            
        try:
            self.search = SearchConfig()
            logger.info("Configuration de recherche chargée")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration Search: {e}")
            
        try:
            self.document = DocumentConfig()
            # Vérification spécifique pour Elasticsearch
            if not self.document.ELASTICSEARCH_HOST:
                logger.error("ELASTICSEARCH_HOST est vide après chargement")
            else:
                logger.info(f"ELASTICSEARCH_HOST chargé: {self.document.ELASTICSEARCH_HOST}")
            logger.info("Configuration des documents chargée")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration Document: {e}")
            
        try:
            self.chat = ChatConfig()
            logger.info("Configuration du chat chargée")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration Chat: {e}")
        
        # Log pour le débogage
        logger.info(f"Configuration initialisée avec succès: DEBUG={self.DEBUG}, ENV={self.ENV}")

    def reload(self):
        """Recharge les configurations depuis les variables d'environnement."""
        # Recharger les variables d'environnement
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info("Variables d'environnement rechargées")
        
        # Réinitialiser les configurations spécialisées
        from .database import DatabaseConfig
        from .cache import CacheConfig
        from .server import ServerConfig
        from .security import SecurityConfig
        from .hardware import HardwareConfig
        from .models import ModelsConfig
        from .search import SearchConfig
        from .document import DocumentConfig
        from .chat import ChatConfig
        
        self.db = DatabaseConfig()
        self.cache = CacheConfig()
        self.server = ServerConfig()
        self.security = SecurityConfig()
        self.hardware = HardwareConfig()
        self.models = ModelsConfig()
        self.search = SearchConfig()
        self.document = DocumentConfig()
        self.chat = ChatConfig()
        
        logger.info("Configurations rechargées")
        return self

# Instance unique des configurations
settings = Settings()