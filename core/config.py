# core/config.py
from pydantic_settings import BaseSettings
from typing import Dict, Optional, List, Union
from pathlib import Path
import os
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

class Settings(BaseSettings):
    # Paramètres de base
    APP_NAME: str = "AI Chat Assistant"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENV: str = os.getenv("ENV", "production")
    
    # Sécurité
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key")
    API_KEY: str = os.getenv("API_KEY", "your-api-key")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    CORS_ORIGINS: List[str] = ["*"]

    # Base de données
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/db")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    DB_ECHO: bool = DEBUG
    DB_SSL: bool = os.getenv("DB_SSL", "false").lower() == "true"
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_SSL: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    
    # Elasticsearch
    ELASTICSEARCH_HOST: str = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
    ELASTICSEARCH_USER: str = os.getenv("ELASTICSEARCH_USER", "")
    ELASTICSEARCH_PASSWORD: str = os.getenv("ELASTICSEARCH_PASSWORD", "")
    ELASTICSEARCH_VERIFY_CERTS: bool = os.getenv("ELASTICSEARCH_VERIFY_CERTS", "false").lower() == "true"
    ELASTICSEARCH_CA_CERTS: Optional[str] = os.getenv("ELASTICSEARCH_CA_CERTS")
    ELASTICSEARCH_INDEX_PREFIX: str = os.getenv("ELASTICSEARCH_INDEX_PREFIX", "chat")
    ELASTICSEARCH_EMBEDDING_DIM: int = 384  # Dimension des embeddings

    # Chemins
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    LOG_DIR: Path = BASE_DIR / "logs"
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_DIR: Path = BASE_DIR / "cache"
    MODEL_DIR: Path = BASE_DIR / "models"
    
    # Configuration du modèle
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt2")
    MODEL_REVISION: str = os.getenv("MODEL_REVISION", "main")
    USE_AUTH_TOKEN: bool = os.getenv("USE_AUTH_TOKEN", "false").lower() == "true"
    DEVICE: str = os.getenv("DEVICE", "cpu")
    FP16: bool = os.getenv("FP16", "false").lower() == "true"
    
    # Paramètres de génération
    MAX_NEW_TOKENS: int = 150
    MIN_NEW_TOKENS: int = 30
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    TOP_K: int = 50
    DO_SAMPLE: bool = True
    NUM_BEAMS: int = 1
    
    # Configuration du chat
    MAX_HISTORY_LENGTH: int = 10
    MAX_PROMPT_LENGTH: int = 1000
    MAX_RESPONSE_LENGTH: int = 2000
    STREAM_CHUNK_SIZE: int = 20
    STREAM_DELAY: float = 0.05

    # Traitement des documents
    PDF_TEMP_DIR: str = "temp/pdf"
    PDF_IMAGE_QUALITY: int = 85
    PDF_MAX_IMAGE_SIZE: Tuple[int, int] = (800, 800)
    SUPPORTED_FORMATS: List[str] = ['.pdf', '.docx', '.txt']
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT: int = 5
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9090"))
    HEALTH_CHECK_INTERVAL: int = 60  # secondes
    
    # Performance
    WORKERS: int = int(os.getenv("WORKERS", "4"))
    TIMEOUT: int = int(os.getenv("TIMEOUT", "60"))
    KEEPALIVE: int = int(os.getenv("KEEPALIVE", "5"))
    MAX_REQUESTS: int = int(os.getenv("MAX_REQUESTS", "10000"))
    
    # Cache
    CACHE_TTL: int = 3600  # secondes
    CACHE_PREFIX: str = "chat:"
    
    # Nettoyage automatique
    CLEANUP_ENABLED: bool = True
    CLEANUP_INTERVAL: int = 24 * 60 * 60  # 24 heures
    CLEANUP_THRESHOLD_DAYS: int = 30
    
    # Messages système
    SYSTEM_MESSAGES = {
        "welcome": "Bienvenue ! Comment puis-je vous aider ?",
        "error": "Désolé, une erreur est survenue.",
        "rate_limit": "Vous avez atteint la limite de requêtes.",
        "maintenance": "Le système est en maintenance."
    }

    class Config:
        env_file = ".env"
        case_sensitive = True
        
        # Configuration personnalisée
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings
        ):
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )

    def get_database_url(self, include_db: bool = True) -> str:
        """Retourne l'URL de la base de données."""
        url = self.DATABASE_URL
        if not include_db:
            # Retire le nom de la base de données pour les migrations
            url = url.rsplit('/', 1)[0]
        return url

    @property
    def model_path(self) -> Path:
        """Retourne le chemin du modèle."""
        return self.MODEL_DIR / self.MODEL_NAME.replace('/', '_')

    def get_log_path(self, name: str) -> Path:
        """Retourne le chemin d'un fichier de log."""
        return self.LOG_DIR / f"{name}.log"

    @property
    def generation_config(self) -> Dict:
        """Retourne la configuration de génération."""
        return {
            "max_new_tokens": self.MAX_NEW_TOKENS,
            "min_new_tokens": self.MIN_NEW_TOKENS,
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "top_k": self.TOP_K,
            "do_sample": self.DO_SAMPLE,
            "num_beams": self.NUM_BEAMS
        }

# Instance unique des paramètres
settings = Settings()

# Création des répertoires nécessaires
for path in [settings.LOG_DIR, settings.DATA_DIR, settings.CACHE_DIR, settings.MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)
