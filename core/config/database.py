# core/config/database.py
import os
from pydantic import BaseModel

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
        if not include_db and url:
            url = url.rsplit('/', 1)[0]
        return url