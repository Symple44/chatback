# core/config/security.py
import os
from typing import List
from pydantic import BaseModel

class SecurityConfig(BaseModel):
    """Configuration de sécurité."""
    
    # API and authentication
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    API_KEY: str = os.getenv("API_KEY", "")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # SSL configuration
    SSL_ENABLED: bool = os.getenv("SSL_ENABLED", "true").lower() == "true"
    SSL_CERT_PATH: str = os.getenv("SSL_CERT_PATH", "/path/to/cert.pem")
    SSL_KEY_PATH: str = os.getenv("SSL_KEY_PATH", "/path/to/key.pem")
    
    # Hugging Face authentication
    HUGGING_FACE_HUB_TOKEN: str = os.getenv("HUGGING_FACE_HUB_TOKEN", "")