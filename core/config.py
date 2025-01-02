# core/config.py
from pydantic_settings import BaseSettings
from pydantic import validator 
from typing import Dict, Optional, List, Tuple, ClassVar, Union, Any
from pathlib import Path
import os
import json
from dotenv import load_dotenv

# Ajout de l'import de torch avec gestion d'erreur
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

load_dotenv()

class Settings(BaseSettings):
    # Base - Paramètres fondamentaux
    APP_NAME: str = os.getenv("APP_NAME", "AI Chat Assistant")
    VERSION: str = os.getenv("VERSION", "1.0.0") 
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENV: str = os.getenv("ENV", "production")
    PORT: int = int(os.getenv("PORT", "8000"))
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    API_KEY: str = os.getenv("API_KEY", "")

    # Base Security
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    CORS_ORIGINS: List[str] = ["*"]

    # Database - Paramètres critiques
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "16"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "32"))
    DB_ECHO: bool = DEBUG
    DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))
    DB_POOL_PRE_PING: bool = True

    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_SSL: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    REDIS_SOCKET_TIMEOUT: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", "2.0"))
    REDIS_MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))

    # Elasticsearch
    ELASTICSEARCH_HOST: str = os.getenv("ELASTICSEARCH_HOST", "")
    ELASTICSEARCH_USER: str = os.getenv("ELASTICSEARCH_USER", "")
    ELASTICSEARCH_PASSWORD: str = os.getenv("ELASTICSEARCH_PASSWORD", "")
    ELASTICSEARCH_CA_CERT: Optional[str] = os.getenv("ELASTICSEARCH_CA_CERT")
    ELASTICSEARCH_CLIENT_CERT: Optional[str] = os.getenv("ELASTICSEARCH_CLIENT_CERT")
    ELASTICSEARCH_CLIENT_KEY: Optional[str] = os.getenv("ELASTICSEARCH_CLIENT_KEY")
    ELASTICSEARCH_INDEX_PREFIX: str = os.getenv("ELASTICSEARCH_INDEX_PREFIX", "chat")
    ELASTICSEARCH_VERIFY_CERTS: bool = os.getenv("ELASTICSEARCH_VERIFY_CERTS", "true").lower() == "true"
    ELASTICSEARCH_EMBEDDING_DIM: int = int(os.getenv("ELASTICSEARCH_EMBEDDING_DIM", "384"))
    ELASTICSEARCH_NUMBER_OF_SHARDS: int = int(os.getenv("ELASTICSEARCH_NUMBER_OF_SHARDS", "3"))
    ELASTICSEARCH_NUMBER_OF_REPLICAS: int = int(os.getenv("ELASTICSEARCH_NUMBER_OF_REPLICAS", "1"))
    ELASTICSEARCH_REFRESH_INTERVAL: str = os.getenv("ELASTICSEARCH_REFRESH_INTERVAL", "5s")

    # Machine Learning
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "mistralai/Mixtral-8x7B-Instruct-v0.1")
    MODEL_NAME_SUMMARIZER: str = os.getenv("MODEL_NAME_SUMMARIZER", "csebuetnlp/mT5_multilingual_XLSum")
    MODEL_REVISION: str = os.getenv("MODEL_REVISION", "main")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "768"))

    # GPU Configuration
    USE_CPU_ONLY: bool = os.getenv("USE_CPU_ONLY", "false").lower() == "true"
    CUDA_VISIBLE_DEVICES: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    CUDA_MEMORY_FRACTION: float = float(os.getenv("CUDA_MEMORY_FRACTION", "0.85"))
    GPU_MEMORY_FRACTION: float = float(os.getenv("GPU_MEMORY_FRACTION", "0.9"))
    USE_FP16: bool = os.getenv("USE_FP16", "true").lower() == "true"
    USE_8BIT: bool = os.getenv("USE_8BIT", "false").lower() == "true"
    USE_4BIT: bool = os.getenv("USE_4BIT", "false").lower() == "true"
    MODEL_PARALLEL_SIZE: int = int(os.getenv("MODEL_PARALLEL_SIZE", "1"))
    DEVICE: str = "cuda" if not USE_CPU_ONLY and torch.cuda.is_available() else "cpu"
    PYTORCH_CUDA_ALLOC_CONF: str = os.getenv(
        "PYTORCH_CUDA_ALLOC_CONF", 
        "max_split_size_mb:2048,garbage_collection_threshold:0.8"
    )
    PYTORCH_ENABLE_MEM_EFFICIENT_OFFLOAD: bool = os.getenv(
        "PYTORCH_ENABLE_MEM_EFFICIENT_OFFLOAD", 
        "true"
    ).lower() == "true"

    # Performance
    MAX_THREADS: int = int(os.getenv("MAX_THREADS", "16"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))
    MAX_PARALLEL_PROCESSES: int = int(os.getenv("MAX_PARALLEL_PROCESSES", "8"))
    MAX_MEMORY: str = os.getenv("MAX_MEMORY", '{"0":"20GiB","cpu":"24GB"}')
    MEMORY_LIMIT: int = int(os.getenv("MEMORY_LIMIT", "51200"))
    OFFLOAD_FOLDER: str = os.getenv("OFFLOAD_FOLDER", "offload_folder")

    @validator('MAX_MEMORY')
    def validate_max_memory(cls, v: str) -> str:
        """Valide la configuration mémoire."""
        try:
            config = json.loads(v)
            if not isinstance(config, dict):
                raise ValueError("MAX_MEMORY doit être un dictionnaire JSON")
            
            # Vérification des clés et valeurs
            valid_keys = set(['cpu', 'disk']) | {str(i) for i in range(8)}  # Accepte 'cpu', 'disk' et '0'-'7'
            for key in config.keys():
                if key not in valid_keys:
                    raise ValueError(f"Clé invalide dans MAX_MEMORY: {key}")
                
                # Vérification du format des valeurs (ex: "22GiB", "24GB")
                value = config[key]
                if not isinstance(value, str) or not any(value.endswith(unit) for unit in ['GB', 'GiB']):
                    raise ValueError(f"Format de mémoire invalide pour {key}: {value}")
            
            return v
        except json.JSONDecodeError:
            raise ValueError("MAX_MEMORY doit être un JSON valide")
        except Exception as e:
            raise ValueError(f"Configuration mémoire invalide: {e}")

    # Paths - Structure du projet
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    LOG_DIR: Path = BASE_DIR / "logs"
    DATA_DIR: Path = BASE_DIR / "data" 
    CACHE_DIR: Path = BASE_DIR / "cache"
    MODEL_DIR: Path = BASE_DIR / "models"
    PDF_TEMP_DIR: str = "temp/pdf"

    # Formats
    SUPPORTED_FORMATS: List[str] = ['.pdf', '.docx', '.txt']

    # Google Drive Configuration
    GOOGLE_DRIVE_FOLDER_ID: str = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
    GOOGLE_DRIVE_CREDENTIALS_PATH: str = os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH", "")
    GOOGLE_DRIVE_SYNC_INTERVAL: int = int(os.getenv("GOOGLE_DRIVE_SYNC_INTERVAL", "3600"))

    # Model et Document Processing
    MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "4096"))
    MAX_OUTPUT_LENGTH: int = int(os.getenv("MAX_OUTPUT_LENGTH", "4096"))
    MAX_PROMPT_LENGTH: int = int(os.getenv("MAX_PROMPT_LENGTH", "2048"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "8192"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "2000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    MAX_RELEVANT_DOCS: int = int(os.getenv("MAX_RELEVANT_DOCS", "8"))
    MAX_CHUNKS_PER_DOC: int = int(os.getenv("MAX_CHUNKS_PER_DOC", "100"))
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "5"))
    PROMPT_TEMPERATURE: float = float(os.getenv("PROMPT_TEMPERATURE", "0.7"))
    CONTEXT_TOKENS_LIMIT: int = int(os.getenv("CONTEXT_TOKENS_LIMIT", "2047"))

    # Templates
    CHAT_TEMPLATE: str = """<|system|>
    {system}
    </|system|>

    <|context|>
    === CONTEXTE PERTINENT ===
    {context}

    === ANALYSE DU CONTEXTE ===
    Thèmes identifiés:
    {themes}

    Points à clarifier:
    {clarifications}

    Questions suggérées:
    {questions}
    </|context|>

    <|history|>
    === HISTORIQUE DE LA CONVERSATION ===
    {history}
    </|history|>

    <|instructions|>
    === INSTRUCTIONS SPÉCIFIQUES ===
    {instructions}
    </|instructions|>

    <|user|>
    Question: {query}
    </|user|>

    {assistant_section}"""

    # Rôles système
    SYSTEM_ROLES: Dict[str, str] = {
        "system": "Je suis {app_name}, votre assistant IA spécialisé.",
        "user": "En tant qu'utilisateur, je demande :",
        "assistant": "En tant qu'assistant technique, je vais vous aider :",
        "context": "Voici le contexte pertinent :",
    }

    # Types de réponses
    RESPONSE_TYPES: Dict[str, Dict[str, Any]] = {
        "comprehensive": {
            "description": "Réponse détaillée et complète",
            "temperature": 0.7,
            "max_tokens": 1024
        },
        "concise": {
            "description": "Réponse courte et directe",
            "temperature": 0.5,
            "max_tokens": 512
        },
        "technical": {
            "description": "Réponse technique avec détails",
            "temperature": 0.3,
            "max_tokens": 1024
        }
    }

    # Format des clarifications
    CLARIFICATION_FORMAT: str = """
    Pour mieux vous aider, j'aurais besoin de quelques précisions:
    {clarifications}

    Suggestions de reformulation:
    {suggestions}
    """

    # Server Configuration
    WORKERS: int = int(os.getenv("WORKERS", "24"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "32"))
    TIMEOUT: int = int(os.getenv("TIMEOUT", "120"))
    KEEPALIVE: int = int(os.getenv("KEEPALIVE", "5"))
    MAX_REQUESTS: int = int(os.getenv("MAX_REQUESTS", "20000"))

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

    def get_database_url(self, include_db: bool = True) -> str:
        url = self.DATABASE_URL
        if not include_db and url:
            url = url.rsplit('/', 1)[0]
        return url

    @property
    def model_path(self) -> Path:
        return self.MODEL_DIR / self.MODEL_NAME.replace('/', '_')

    def get_log_path(self, name: str) -> Path:
        return self.LOG_DIR / f"{name}.log"

    @property
    def generation_config(self) -> Dict:
        """Configuration de génération pour le modèle."""
        return {
            "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "1024")),
            "min_new_tokens": int(os.getenv("MIN_NEW_TOKENS", "30")),
            "temperature": float(os.getenv("TEMPERATURE", "0.7")),
            "top_p": float(os.getenv("TOP_P", "0.95")),
            "top_k": int(os.getenv("TOP_K", "50")),
            "do_sample": os.getenv("DO_SAMPLE", "true").lower() == "true",
            "num_beams": int(os.getenv("NUM_BEAMS", "1")),
            "repetition_penalty": float(os.getenv("REPETITION_PENALTY", "1.1")),
            "length_penalty": float(os.getenv("LENGTH_PENALTY", "1.0"))
        }
        
settings = Settings()

# Création des répertoires requis
for path in [settings.LOG_DIR, settings.DATA_DIR, settings.CACHE_DIR, settings.MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)
