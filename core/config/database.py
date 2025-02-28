# core/config/database.py
import os
import logging
from pydantic import BaseModel

# Ajout du logger
logger = logging.getLogger(__name__)

class DatabaseConfig(BaseModel):
    """Configuration de la base de données."""
    
    # Connection string
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # Pool configuration
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "32"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "64"))
    DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))
    DB_POOL_PRE_PING: bool = True
    DB_ECHO: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    def get_database_url(self, include_db: bool = True) -> str:
        """Retourne l'URL de la base de données."""
        url = self.DATABASE_URL
        
        # Vérification et notification si l'URL est vide
        if not url:
            logger.error("DATABASE_URL est vide - vérifiez votre fichier .env")
            # Utiliser une URL par défaut en développement ou lever une exception
            # Remplacez par vos valeurs spécifiques
            url = "postgresql+asyncpg://aioweoadmin:X%7D5%7Bk6n%409fQguVWE76%2C%5B@192.168.0.25:5432/AICHAT"
            logger.warning(f"Utilisation de l'URL de base de données codée en dur : {url}")
        
        if not include_db and url:
            url = url.rsplit('/', 1)[0]
            
        return url