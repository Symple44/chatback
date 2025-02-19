# core/config.py
from pydantic_settings import BaseSettings
from pydantic import validator
from typing import Dict, Optional, List, Tuple, Union, Any
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
    """Configuration générale de l'application."""
    
    # === Application Settings ===
    APP_NAME: str = os.getenv("APP_NAME", "AI Chat Assistant")
    VERSION: str = os.getenv("VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENV: str = os.getenv("ENV", "production")
    PORT: int = int(os.getenv("PORT", "8000"))
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    API_KEY: str = os.getenv("API_KEY", "")

    # === Base Security ===
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    CORS_ORIGINS: List[str] = ["*"]
    SSL_ENABLED: bool = os.getenv("SSL_ENABLED", "true").lower() == "true"
    SSL_CERT_PATH: str = os.getenv("SSL_CERT_PATH", "/path/to/cert.pem")
    SSL_KEY_PATH: str = os.getenv("SSL_KEY_PATH", "/path/to/key.pem")

    # === Database Configuration ===
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "32"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "64"))
    DB_ECHO: bool = DEBUG
    DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))
    DB_POOL_PRE_PING: bool = True

    # === Redis Configuration ===
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_SSL: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    REDIS_SOCKET_TIMEOUT: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", "2.0"))
    REDIS_MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "200"))

    # === Model Configuration ===
    HUGGING_FACE_HUB_TOKEN: str = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
    
    # Configuration des modèles principaux
    MODEL_NAME: str = os.getenv("MODEL_NAME", "mistralai/Mixtral-8x7B-Instruct-v0.1")
    MODEL_NAME_SUMMARIZER: str = os.getenv("MODEL_NAME_SUMMARIZER", "csebuetnlp/mT5_multilingual_XLSum")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    MODEL_REVISION: str = os.getenv("MODEL_REVISION", "main")

    # === GPU & CUDA Configuration ===
    CUDA_VISIBLE_DEVICES: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    CUDA_MEMORY_FRACTION: float = float(os.getenv("CUDA_MEMORY_FRACTION", "0.95"))
    CUDNN_BENCHMARK: bool = os.getenv("CUDNN_BENCHMARK", "true").lower() == "true"
    
    # Configuration CUDA avancée
    CUDA_MODULE_LOADING: str = os.getenv("CUDA_MODULE_LOADING", "LAZY")
    PYTORCH_CUDA_ALLOC_CONF: str = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:4096,garbage_collection_threshold:0.9")
    PYTORCH_ENABLE_MEM_EFFICIENT_OFFLOAD: bool = os.getenv("PYTORCH_ENABLE_MEM_EFFICIENT_OFFLOAD", "true").lower() == "true"

    # Configuration mémoire # DEPRECIATED
    MAX_MEMORY: Dict[str, str] = { 
        "0": "22GiB",  # GPU principal
        "cpu": "24GB"  # CPU
    }
    OFFLOAD_FOLDER: str = os.getenv("OFFLOAD_FOLDER", "offload_folder") # DEPRECIATED

    # Configuration quantification
    USE_FP16: bool = os.getenv("USE_FP16", "true").lower() == "true"

    # Google Drive Configuration
    GOOGLE_DRIVE_FOLDER_ID: str = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
    GOOGLE_DRIVE_CREDENTIALS_PATH: str = os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH", "")
    GOOGLE_DRIVE_SYNC_INTERVAL: int = int(os.getenv("GOOGLE_DRIVE_SYNC_INTERVAL", "3600"))
    
    # === Document Processing === DEPRECIATED
    MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "8192"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "8192"))
    MAX_RELEVANT_DOCS: int = int(os.getenv("MAX_RELEVANT_DOCS", "6"))

    # === Server Settings ===
    WORKERS: int = int(os.getenv("WORKERS", "24"))
    TIMEOUT: int = int(os.getenv("TIMEOUT", "180"))
    KEEPALIVE: int = int(os.getenv("KEEPALIVE", "5"))
    MAX_REQUESTS: int = int(os.getenv("MAX_REQUESTS", "10000"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "64"))

    # === Performance Configuration ===
    MKL_NUM_THREADS: int = int(os.getenv("MKL_NUM_THREADS", "16"))
    NUMEXPR_NUM_THREADS: int = int(os.getenv("NUMEXPR_NUM_THREADS", "16"))
    OMP_NUM_THREADS: int = int(os.getenv("OMP_NUM_THREADS", "16"))
    OPENBLAS_NUM_THREADS: int = int(os.getenv("OPENBLAS_NUM_THREADS", "16"))
    PYTORCH_ENABLE_CUDA_KERNEL_AUTOTUNING: int = 1

    # === Paths Configuration ===
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    LOG_DIR: Path = BASE_DIR / "logs"
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_DIR: Path = BASE_DIR / "cache"
    MODELS_DIR: Path = BASE_DIR / "models"
    PDF_TEMP_DIR: str = "temp/pdf"

    # === Format Configuration ===
    SUPPORTED_FORMATS: List[str] = ['.pdf', '.docx', '.txt']

    # === Elasticsearch Configuration ===
    ELASTICSEARCH_HOST: str = os.getenv("ELASTICSEARCH_HOST", "")
    ELASTICSEARCH_USER: str = os.getenv("ELASTICSEARCH_USER", "elastic")
    ELASTICSEARCH_PASSWORD: str = os.getenv("ELASTICSEARCH_PASSWORD", "")
    ELASTICSEARCH_CA_CERT: Optional[str] = os.getenv("ELASTICSEARCH_CA_CERT")
    ELASTICSEARCH_CLIENT_CERT: Optional[str] = os.getenv("ELASTICSEARCH_CLIENT_CERT")
    ELASTICSEARCH_CLIENT_KEY: Optional[str] = os.getenv("ELASTICSEARCH_CLIENT_KEY")
    ELASTICSEARCH_VERIFY_CERTS: bool = os.getenv("ELASTICSEARCH_VERIFY_CERTS", "true").lower() == "true"
    ELASTICSEARCH_INDEX_PREFIX: str = os.getenv("ELASTICSEARCH_INDEX_PREFIX", "owai")
    ELASTICSEARCH_EMBEDDING_DIM: int = int(os.getenv("ELASTICSEARCH_EMBEDDING_DIM", "768"))

    # === Streaming Configuration ===
    STREAM_BATCH_SIZE: int = int(os.getenv("STREAM_BATCH_SIZE", "10"))
    STREAM_MAX_QUEUE_SIZE: int = int(os.getenv("STREAM_MAX_QUEUE_SIZE", "1000"))
    STREAM_TIMEOUT: float = float(os.getenv("STREAM_TIMEOUT", "5.0"))
    STREAM_CHUNK_SIZE: int = int(os.getenv("STREAM_CHUNK_SIZE", "8192"))
    STREAM_KEEP_ALIVE: int = int(os.getenv("STREAM_KEEP_ALIVE", "15"))
    STREAM_RETRY_TIMEOUT: int = int(os.getenv("STREAM_RETRY_TIMEOUT", "30"))
    
    # === Processeur de chat ===
    CONFIDENCE_THRESHOLD: float = 0.5
    MIN_RELEVANT_DOCS: int = 1
    MAX_THEMES: int = 3
    MAX_CLARIFICATION_ATTEMPTS: int = 3
    MIN_QUERY_LENGTH: int = 2
    MAX_CONTEXT_DOCS: int = 6

    # Paramètres de configuration utilisateur (chat)
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "5"))
    
    # DEPRECIATED
    SYSTEM_ROLES: Dict[str, str] = {
        "system": "Je suis {app_name}, votre assistant IA spécialisé.",
        "user": "En tant qu'utilisateur, je demande :",
        "assistant": "En tant qu'assistant technique, je vais vous aider :",
        "context": "Voici le contexte pertinent :",
    }

     # === Search Configuration ===
    DEFAULT_SEARCH_METHOD: str = os.getenv("DEFAULT_SEARCH_METHOD", "rag")
    SEARCH_MAX_DOCS: int = int(os.getenv("SEARCH_MAX_DOCS", "20")) # Limite max de documents dans la recherche (niveau le plus haut)
    SEARCH_MIN_SCORE: float = float(os.getenv("SEARCH_MIN_SCORE", "0.3"))
    
    # Configuration des stratégies de recherche
    SEARCH_CACHE_TTL: int = int(os.getenv("SEARCH_CACHE_TTL", "3600"))
    SEARCH_MAX_CONCURRENT: int = int(os.getenv("SEARCH_MAX_CONCURRENT", "10"))
    SEARCH_BATCH_SIZE: int = int(os.getenv("SEARCH_BATCH_SIZE", "32"))
    
    # Configuration RAG
    RAG_VECTOR_WEIGHT: float = float(os.getenv("RAG_VECTOR_WEIGHT", "0.7"))
    RAG_SEMANTIC_WEIGHT: float = float(os.getenv("RAG_SEMANTIC_WEIGHT", "0.3"))
    
    # Configuration Hybrid
    HYBRID_RERANK_TOP_K: int = int(os.getenv("HYBRID_RERANK_TOP_K", "10"))
    HYBRID_WINDOW_SIZE: int = int(os.getenv("HYBRID_WINDOW_SIZE", "3"))
    
    # Configuration Semantic
    SEMANTIC_MAX_CONCEPTS: int = int(os.getenv("SEMANTIC_MAX_CONCEPTS", "5"))
    SEMANTIC_BOOST_EXACT: bool = os.getenv("SEMANTIC_BOOST_EXACT", "true").lower() == "true"


    @validator('DEFAULT_SEARCH_METHOD')
    def validate_search_method(cls, v):
        """Valide que la méthode de recherche est supportée."""
        valid_methods = ["rag", "hybrid", "semantic", "disabled"]
        if v not in valid_methods:
            raise ValueError(f"Méthode de recherche invalide. Valeurs possibles : {valid_methods}")
        return v
    
    # Types de réponses DEPRECIATED
    RESPONSE_TYPES: Dict[str, Dict[str, Any]] = {
        "comprehensive": {
            "description": "Réponse détaillée et approfondie",
            "temperature": 0.6,
            "style": "Je vais vous donner une explication détaillée et complète."
        },
        "concise": {
            "description": "Réponse courte et directe",
            "temperature": 0.5,
            "style": "Voici une réponse brève et précise."
        },
        "technical": {
            "description": "Réponse technique avec détails d'implémentation",
            "temperature": 0.3,
            "style": "Je vais vous fournir les détails techniques."
        },
        "clarification": {
            "description": "Demande de clarification",
            "temperature": 0.4,
            "style": "Pour mieux vous répondre, j'ai besoin de quelques précisions."
        }
    }

    # === Validation Methods ===
    @validator('MAX_MEMORY')
    def validate_max_memory(cls, v: str) -> str:
        """Valide la configuration mémoire."""
        try:
            if isinstance(v, str):
                config = json.loads(v)
            else:
                config = v

            # Vérification des clés et valeurs
            valid_keys = set(['cpu', 'disk']) | {str(i) for i in range(8)}
            for key in config.keys():
                if key not in valid_keys:
                    raise ValueError(f"Clé invalide dans MAX_MEMORY: {key}")
                
                # Vérification du format des valeurs
                value = config[key]
                if not isinstance(value, str) or not any(value.endswith(unit) for unit in ['GB', 'GiB']):
                    raise ValueError(f"Format de mémoire invalide pour {key}: {value}")
            
            return config if isinstance(v, dict) else v

        except json.JSONDecodeError:
            raise ValueError("MAX_MEMORY doit être un JSON valide")
        except Exception as e:
            raise ValueError(f"Configuration mémoire invalide: {e}")

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

    # === Helper Methods ===
    def get_database_url(self, include_db: bool = True) -> str:
        """Retourne l'URL de la base de données."""
        url = self.DATABASE_URL
        if not include_db and url:
            url = url.rsplit('/', 1)[0]
        return url

    def get_model_path(self, model_name: str) -> Path:
        """Retourne le chemin du modèle."""
        return self.MODELS_DIR / model_name.replace('/', '_')

    def get_log_path(self, name: str) -> Path:
        """Retourne le chemin des logs."""
        return self.LOG_DIR / f"{name}.log"

settings = Settings()

# Création des répertoires requis
for path in [settings.LOG_DIR, settings.DATA_DIR, settings.CACHE_DIR, settings.MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)