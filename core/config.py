# core/config.py
from pydantic_settings import BaseSettings
from typing import Dict, Optional, List, Tuple, ClassVar, Union
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Base
    APP_NAME: str = os.getenv("APP_NAME", "AI Chat Assistant")
    VERSION: str = os.getenv("VERSION", "1.0.0") 
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENV: str = os.getenv("ENV", "production")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Security 
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    API_KEY: str = os.getenv("API_KEY", "")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    CORS_ORIGINS: List[str] = ["*"]

    # Database - Optimisé pour charge élevée
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "16"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "32"))
    DB_ECHO: bool = DEBUG

    # Redis - Cache haute performance
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_SSL: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: float = 2.0
    REDIS_RETRY_ON_TIMEOUT: bool = True

    # Elasticsearch - Configuration optimisée
    ELASTICSEARCH_HOST: str = os.getenv("ELASTICSEARCH_HOST")
    ELASTICSEARCH_USER: str = os.getenv("ELASTICSEARCH_USER")
    ELASTICSEARCH_PASSWORD: str = os.getenv("ELASTICSEARCH_PASSWORD")
    ELASTICSEARCH_CA_CERT: Optional[str] = os.getenv("ELASTICSEARCH_CA_CERT")
    ELASTICSEARCH_CLIENT_CERT: Optional[str] = os.getenv("ELASTICSEARCH_CLIENT_CERT")
    ELASTICSEARCH_CLIENT_KEY: Optional[str] = os.getenv("ELASTICSEARCH_CLIENT_KEY")
    ELASTICSEARCH_INDEX_PREFIX: str = "chat"
    ELASTICSEARCH_EMBEDDING_DIM: int = 384
    ELASTICSEARCH_REFRESH_INTERVAL: str = "5s"
    ELASTICSEARCH_NUMBER_OF_SHARDS: int = 3
    ELASTICSEARCH_NUMBER_OF_REPLICAS: int = 1
    ELASTICSEARCH_MAX_RESULT_WINDOW: int = 10000

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    LOG_DIR: Path = BASE_DIR / "logs"
    DATA_DIR: Path = BASE_DIR / "data" 
    CACHE_DIR: Path = BASE_DIR / "cache"
    MODEL_DIR: Path = BASE_DIR / "models"
    PDF_TEMP_DIR: str = "temp/pdf"

    # Model - Configuration pour Mixtral
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "mistralai/Mixtral-8x7B-Instruct-v0.1")
    MODEL_REVISION: str = "main"
    DEVICE: str = "cuda"
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    
    # GPU & Performance (RTX 3090) - Optimisé pour haute performance
    CUDA_VISIBLE_DEVICES: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    CUDA_LAUNCH_BLOCKING: int = int(os.getenv("CUDA_LAUNCH_BLOCKING", "0"))
    CUDA_DEVICE: int = int(os.getenv("CUDA_DEVICE", "0"))
    CUDA_MEMORY_FRACTION: float = float(os.getenv("CUDA_MEMORY_FRACTION", "0.9"))
    CUDA_ALLOW_GROWTH: bool = os.getenv("CUDA_ALLOW_GROWTH", "true").lower() == "true"
    TF_FORCE_GPU_ALLOW_GROWTH: bool = os.getenv("TF_FORCE_GPU_ALLOW_GROWTH", "true").lower() == "true"
    PYTORCH_CUDA_ALLOC_CONF: str = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:2048")
    USE_CPU_ONLY: bool = os.getenv("USE_CPU_ONLY", "false").lower() == "true"
    GPU_MEMORY_FRACTION: float = float(os.getenv("GPU_MEMORY_FRACTION", "0.9"))
    MAX_THREADS: int = int(os.getenv("MAX_THREADS", "16"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))
    USE_FP16: bool = os.getenv("USE_FP16", "true").lower() == "true"
    MEMORY_LIMIT: int = int(os.getenv("MEMORY_LIMIT", "51200"))  # 50GB
    MODEL_PARALLEL_SIZE: int = int(os.getenv("MODEL_PARALLEL_SIZE", "2"))
    MAX_PARALLEL_PROCESSES: int = int(os.getenv("MAX_PARALLEL_PROCESSES", "8"))
    MKL_NUM_THREADS: int = int(os.getenv("MKL_NUM_THREADS", "16"))
    NUMEXPR_NUM_THREADS: int = int(os.getenv("NUMEXPR_NUM_THREADS", "16"))
    OMP_NUM_THREADS: int = int(os.getenv("OMP_NUM_THREADS", "16"))
    PYTORCH_ENABLE_CUDA_KERNEL_AUTOTUNING: bool = True
    USE_4BIT: bool = os.getenv("USE_4BIT", "false").lower() == "true"

    # Model Limits - Augmenté pour Mixtral
    MAX_INPUT_LENGTH: int = 4096
    MAX_OUTPUT_LENGTH: int = 4096
    MAX_PROMPT_LENGTH: int = 2048
    MAX_CONTEXT_LENGTH: int = 8192

    # Generation - Ajusté pour qualité/performance
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    TOP_K: int = 50
    DO_SAMPLE: bool = True
    NUM_BEAMS: int = 1
    MIN_NEW_TOKENS: int = 30
    MAX_NEW_TOKENS: int = 1024
    REPETITION_PENALTY: float = 1.1
    LENGTH_PENALTY: float = 1.0
    NO_REPEAT_NGRAM_SIZE: int = 3

    # Chat Settings - Optimisé pour performance
    MAX_HISTORY_LENGTH: int = 20
    STREAM_CHUNK_SIZE: int = 40
    STREAM_DELAY: float = 0.02
    MAX_CONCURRENT_CHATS: int = 100

    # Document Processing - Optimisé pour RTX 3090
    PDF_IMAGE_QUALITY: int = 90
    PDF_MAX_IMAGE_SIZE: Tuple[int, int] = (1200, 1200)
    SUPPORTED_FORMATS: List[str] = ['.pdf', '.docx', '.txt']
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "2000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    MAX_RELEVANT_DOCS: int = int(os.getenv("MAX_RELEVANT_DOCS", "8"))
    MAX_CHUNKS_PER_DOC: int = int(os.getenv("MAX_CHUNKS_PER_DOC", "100"))
    CONTEXT_CONFIDENCE_THRESHOLD: float = 0.75

    # Google Drive
    GOOGLE_DRIVE_FOLDER_ID: str = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    GOOGLE_DRIVE_CREDENTIALS_PATH: str = os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH")
    GOOGLE_DRIVE_SYNC_INTERVAL: int = int(os.getenv("GOOGLE_DRIVE_SYNC_INTERVAL", "3600"))
    GOOGLE_DRIVE_BATCH_SIZE: int = 50
    GOOGLE_DRIVE_MAX_RETRIES: int = 3

    # Logging - Optimisé pour débogage
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_MAX_BYTES: int = 50 * 1024 * 1024  # 50MB
    LOG_FILE_BACKUP_COUNT: int = 10
    LOG_ROTATION_INTERVAL: int = 24  # heures

    # Metrics & Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9090"))
    HEALTH_CHECK_INTERVAL: int = 30
    METRICS_RETENTION_DAYS: int = 7
    ENABLE_TRACING: bool = True
    PROFILING_ENABLED: bool = False

    # Server Performance - Optimisé pour 16 coeurs
    WORKERS: int = int(os.getenv("WORKERS", "16"))
    TIMEOUT: int = int(os.getenv("TIMEOUT", "120"))
    KEEPALIVE: int = int(os.getenv("KEEPALIVE", "5"))
    MAX_REQUESTS: int = int(os.getenv("MAX_REQUESTS", "20000"))
    GRACEFUL_TIMEOUT: int = 30
    WORKER_CLASS: str = "uvicorn.workers.UvicornWorker"

    # Cache
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    CACHE_PREFIX: str = "chat:"
    CACHE_MAX_MEMORY: int = 4096  # MB
    CACHE_MIN_FRAGMENT: int = 64  # bytes

    # Auto Cleanup
    CLEANUP_ENABLED: bool = True
    CLEANUP_INTERVAL: int = 24 * 60 * 60  # 24 heures
    CLEANUP_THRESHOLD_DAYS: int = 30
    CLEANUP_BATCH_SIZE: int = 1000

    # Templates & Messages
    CHAT_TEMPLATE: str = """System: {system}\nQuestion: {query}\nContexte: {context}\n\nRéponse:"""
    SYSTEM_PROMPT: str = """Je suis un assistant IA français, conçu pour aider et répondre aux questions de manière claire et précise."""
    SYSTEM_MESSAGES: ClassVar[Dict[str, str]] = {
        "welcome": "Bienvenue ! Comment puis-je vous aider ?",
        "error": "Désolé, une erreur est survenue.",
        "rate_limit": "Vous avez atteint la limite de requêtes.", 
        "maintenance": "Le système est en maintenance."
    }

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

    def get_database_url(self, include_db: bool = True) -> str:
        url = self.DATABASE_URL
        if not include_db:
            url = url.rsplit('/', 1)[0]
        return url

    @property
    def model_path(self) -> Path:
        return self.MODEL_DIR / self.MODEL_NAME.replace('/', '_')

    def get_log_path(self, name: str) -> Path:
        return self.LOG_DIR / f"{name}.log"

    @property 
    def generation_config(self) -> Dict:
        return {
            "max_new_tokens": self.MAX_NEW_TOKENS,
            "min_new_tokens": self.MIN_NEW_TOKENS,
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "top_k": self.TOP_K,
            "do_sample": self.DO_SAMPLE,
            "num_beams": self.NUM_BEAMS,
            "repetition_penalty": self.REPETITION_PENALTY,
            "length_penalty": self.LENGTH_PENALTY,
            "no_repeat_ngram_size": self.NO_REPEAT_NGRAM_SIZE
        }

settings = Settings()

# Create required directories
for path in [settings.LOG_DIR, settings.DATA_DIR, settings.CACHE_DIR, settings.MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)
