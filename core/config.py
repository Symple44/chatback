# core/config.py
from pydantic_settings import BaseSettings
from typing import Dict, Optional, List, Tuple, ClassVar, Union
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Base - Chargés depuis .env
    APP_NAME: str = os.getenv("APP_NAME", "AI Chat Assistant")
    VERSION: str = os.getenv("VERSION", "1.0.0") 
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENV: str = os.getenv("ENV", "production")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Security - Paramètres structurels
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    CORS_ORIGINS: List[str] = ["*"]

    # Paths - Paramètres structurels
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    LOG_DIR: Path = BASE_DIR / "logs"
    DATA_DIR: Path = BASE_DIR / "data" 
    CACHE_DIR: Path = BASE_DIR / "cache"
    MODEL_DIR: Path = BASE_DIR / "models"
    PDF_TEMP_DIR: str = "temp/pdf"

    # Templates & Messages - Paramètres structurels
    CHAT_TEMPLATE: str = """System: {system}\nQuestion: {query}\nContexte: {context}\n\nRéponse:"""
    SYSTEM_PROMPT: str = """Je suis un assistant IA français, conçu pour aider et répondre aux questions de manière claire et précise."""
    SYSTEM_MESSAGES: ClassVar[Dict[str, str]] = {
        "welcome": "Bienvenue ! Comment puis-je vous aider ?",
        "error": "Désolé, une erreur est survenue.",
        "rate_limit": "Vous avez atteint la limite de requêtes.", 
        "maintenance": "Le système est en maintenance."
    }

    # Formats supportés - Paramètres structurels
    SUPPORTED_FORMATS: List[str] = ['.pdf', '.docx', '.txt']

    # Fonction utilitaire pour l'URL de base de données
    def get_database_url(self, include_db: bool = True) -> str:
        url = os.getenv("DATABASE_URL", "")
        if not include_db:
            url = url.rsplit('/', 1)[0]
        return url

    @property
    def model_path(self) -> Path:
        model_name = os.getenv("MODEL_NAME", "").replace('/', '_')
        return self.MODEL_DIR / model_name

    def get_log_path(self, name: str) -> Path:
        return self.LOG_DIR / f"{name}.log"

    @property 
    def generation_config(self) -> Dict:
        """Configuration de génération basée sur les variables d'environnement."""
        return {
            "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "1024")),
            "min_new_tokens": int(os.getenv("MIN_NEW_TOKENS", "30")),
            "temperature": float(os.getenv("TEMPERATURE", "0.7")),
            "top_p": float(os.getenv("TOP_P", "0.95")),
            "top_k": int(os.getenv("TOP_K", "50")),
            "do_sample": os.getenv("DO_SAMPLE", "true").lower() == "true",
            "num_beams": int(os.getenv("NUM_BEAMS", "1")),
            "repetition_penalty": float(os.getenv("REPETITION_PENALTY", "1.1")),
            "length_penalty": float(os.getenv("LENGTH_PENALTY", "1.0")),
            "no_repeat_ngram_size": int(os.getenv("NO_REPEAT_NGRAM_SIZE", "3"))
        }

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

# Instance unique des paramètres
settings = Settings()

# Création des répertoires requis
for path in [settings.LOG_DIR, settings.DATA_DIR, settings.CACHE_DIR, settings.MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)
