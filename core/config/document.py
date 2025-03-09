# core/config/document.py
import os
import logging
from pydantic import BaseModel, Field, validator
from typing import Optional

# Configuration du logger
logger = logging.getLogger(__name__)

class DocumentConfig(BaseModel):
    """Configuration pour le traitement des documents."""
    
    # Paramètres de chunking
    CHUNK_SIZE: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1500"))
    )
    CHUNK_OVERLAP: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150"))
    )
    
    # Intégration Google Drive
    GOOGLE_DRIVE_FOLDER_ID: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
    )
    GOOGLE_DRIVE_CREDENTIALS_PATH: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH", "")
    )
    GOOGLE_DRIVE_SYNC_INTERVAL: int = Field(
        default_factory=lambda: int(os.getenv("GOOGLE_DRIVE_SYNC_INTERVAL", "3600"))
    )
    
    # Elasticsearch
    ELASTICSEARCH_HOST: str = Field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_HOST", "")
    )
    ELASTICSEARCH_USER: str = Field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_USER", "elastic")
    )
    ELASTICSEARCH_PASSWORD: str = Field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_PASSWORD", "")
    )
    ELASTICSEARCH_CA_CERT: Optional[str] = Field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_CA_CERT")
    )
    ELASTICSEARCH_CLIENT_CERT: Optional[str] = Field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_CLIENT_CERT")
    )
    ELASTICSEARCH_CLIENT_KEY: Optional[str] = Field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_CLIENT_KEY")
    )
    ELASTICSEARCH_VERIFY_CERTS: bool = Field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_VERIFY_CERTS", "true").lower() == "true"
    )
    ELASTICSEARCH_INDEX_PREFIX: str = Field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_INDEX_PREFIX", "owai")
    )
    ELASTICSEARCH_EMBEDDING_DIM: int = Field(
        default_factory=lambda: int(os.getenv("ELASTICSEARCH_EMBEDDING_DIM", "768"))
    )
    ELASTICSEARCH_NUMBER_OF_SHARDS: int = 1
    ELASTICSEARCH_NUMBER_OF_REPLICAS: int = 0
    ELASTICSEARCH_REFRESH_INTERVAL: str = "30s"

    
    ENABLE_AI_TABLE_DETECTION: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_AI_TABLE_DETECTION", "true").lower() == "true"
    )

    TABLE_DETECTION_MODEL: str = Field(
        default_factory=lambda: os.getenv("TABLE_DETECTION_MODEL", "microsoft/table-transformer-detection")
    )

    TABLE_DETECTION_THRESHOLD: float = Field(
        default_factory=lambda: float(os.getenv("TABLE_DETECTION_THRESHOLD", "0.7"))
    )

    TABLE_DETECTION_MAX_TABLES: int = Field(
        default_factory=lambda: int(os.getenv("TABLE_DETECTION_MAX_TABLES", "10"))
    )

    @validator('ELASTICSEARCH_HOST')
    def validate_elasticsearch_host(cls, v):
        """Valide l'URL Elasticsearch."""
        if not v:
            logger.warning("ELASTICSEARCH_HOST est vide")
            return v
        
        # Vérifier si l'URL commence par http:// ou https://
        if not (v.startswith('http://') or v.startswith('https://')):
            logger.warning(f"ELASTICSEARCH_HOST ne commence pas par http:// ou https:// : {v}")
            v = f"https://{v}"
            logger.info(f"URL corrigée: {v}")
        
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Log pour débogage
        logger.info(f"DocumentConfig initialisé avec:")
        logger.info(f"  - ELASTICSEARCH_HOST: '{self.ELASTICSEARCH_HOST}'")
        logger.info(f"  - ELASTICSEARCH_USER: '{self.ELASTICSEARCH_USER}'")
        logger.info(f"  - CHUNK_SIZE: {self.CHUNK_SIZE}")
        logger.info(f"  - CHUNK_OVERLAP: {self.CHUNK_OVERLAP}")
        
        # Vérification des valeurs critiques
        if not self.ELASTICSEARCH_HOST:
            logger.error("ELASTICSEARCH_HOST est vide - vérifiez votre fichier .env")