# core/config/config.py
from pydantic_settings import BaseSettings
from pathlib import Path
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Importer les configurations spécialisées
from .database import DatabaseConfig
from .cache import CacheConfig
from .server import ServerConfig
from .security import SecurityConfig
from .hardware import HardwareConfig
from .models import ModelsConfig
from .search import SearchConfig
from .document import DocumentConfig
from .chat import ChatConfig

load_dotenv()

class Settings(BaseSettings):
    """Configuration unifiée de l'application."""
    
    # Application settings
    APP_NAME: str = os.getenv("APP_NAME", "AI Chat Assistant")
    VERSION: str = os.getenv("VERSION", "1.0.0")
    BUILD_DATE: str = os.getenv("BUILD_DATE", "2025")
    CONTACT_EMAIL : str = os.getenv("CONTACT_EMAIL", "contact@symple.fr")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENV: str = os.getenv("ENV", "production")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    LOG_DIR: Path = BASE_DIR / "logs"
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_DIR: Path = BASE_DIR / "cache"
    MODELS_DIR: Path = BASE_DIR / "models"
    TEMP_DIR: Path = BASE_DIR / "temp"
    
    # Formats
    SUPPORTED_FORMATS: List[str] = ['.pdf', '.docx', '.txt']
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Créer les configurations spécialisées
        self.db = DatabaseConfig()
        self.cache = CacheConfig()
        self.server = ServerConfig()
        self.security = SecurityConfig()
        self.hardware = HardwareConfig()
        self.models = ModelsConfig()
        self.search = SearchConfig()
        self.document = DocumentConfig()
        self.chat = ChatConfig()
        
        # Créer les répertoires requis
        for path in [self.LOG_DIR, self.DATA_DIR, self.CACHE_DIR, self.MODELS_DIR, self.TEMP_DIR]:
            path.mkdir(parents=True, exist_ok=True)

# Instance unique des configurations
settings = Settings()