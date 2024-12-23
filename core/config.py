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

   # Database
   DATABASE_URL: str = os.getenv("DATABASE_URL")
   DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
   DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
   DB_ECHO: bool = DEBUG

   # Redis
   REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
   REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
   REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD")
   REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
   REDIS_SSL: bool = os.getenv("REDIS_SSL", "false").lower() == "true"

   # Elasticsearch 
   ELASTICSEARCH_HOST: str = os.getenv("ELASTICSEARCH_HOST")
   ELASTICSEARCH_USER: str = os.getenv("ELASTICSEARCH_USER")
   ELASTICSEARCH_PASSWORD: str = os.getenv("ELASTICSEARCH_PASSWORD")
   ELASTICSEARCH_CA_CERT: Optional[str] = os.getenv("ELASTICSEARCH_CA_CERT")
   ELASTICSEARCH_CLIENT_CERT: Optional[str] = os.getenv("ELASTICSEARCH_CLIENT_CERT")
   ELASTICSEARCH_CLIENT_KEY: Optional[str] = os.getenv("ELASTICSEARCH_CLIENT_KEY")
   ELASTICSEARCH_INDEX_PREFIX: str = "chat"
   ELASTICSEARCH_EMBEDDING_DIM: int = 384
   ELASTICSEARCH_VERIFY_CERT: bool = True

   # Paths
   BASE_DIR: Path = Path(__file__).resolve().parent.parent
   LOG_DIR: Path = BASE_DIR / "logs"
   DATA_DIR: Path = BASE_DIR / "data" 
   CACHE_DIR: Path = BASE_DIR / "cache"
   MODEL_DIR: Path = BASE_DIR / "models"
   PDF_TEMP_DIR: str = "temp/pdf"

   # Model
   EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
   MODEL_NAME: str = os.getenv("MODEL_NAME")
   MODEL_REVISION: str = "main"
   DEVICE: str = "cuda"
   EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
   
   # GPU & Performance (RTX 3090)
   CUDA_VISIBLE_DEVICES: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
   CUDA_LAUNCH_BLOCKING: int = int(os.getenv("CUDA_LAUNCH_BLOCKING", "0"))
   TF_FORCE_GPU_ALLOW_GROWTH: bool = os.getenv("TF_FORCE_GPU_ALLOW_GROWTH", "true").lower() == "true"
   PYTORCH_CUDA_ALLOC_CONF: str = os.getenv("PYTORCH_CUDA_ALLOC_CONF")
   USE_CPU_ONLY: bool = os.getenv("USE_CPU_ONLY", "false").lower() == "true"
   GPU_MEMORY_FRACTION: float = float(os.getenv("GPU_MEMORY_FRACTION", "0.8"))
   MAX_THREADS: int = int(os.getenv("MAX_THREADS", "16"))
   BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
   USE_FP16: bool = os.getenv("USE_FP16", "true").lower() == "true"
   MEMORY_LIMIT: int = int(os.getenv("MEMORY_LIMIT", "20480"))
   MAX_PARALLEL_PROCESSES: int = int(os.getenv("MAX_PARALLEL_PROCESSES", "4"))
   MKL_NUM_THREADS: int = int(os.getenv("MKL_NUM_THREADS", "16"))
   NUMEXPR_NUM_THREADS: int = int(os.getenv("NUMEXPR_NUM_THREADS", "16")) 
   OMP_NUM_THREADS: int = int(os.getenv("OMP_NUM_THREADS", "16"))
   PYTORCH_ENABLE_CUDA_KERNEL_AUTOTUNING: bool = True
   USE_4BIT: bool = os.getenv("USE_4BIT", "false").lower() == "true"

   # Model Limits
   MAX_INPUT_LENGTH: int = 2048
   MAX_OUTPUT_LENGTH: int = 2048  
   MAX_PROMPT_LENGTH: int = 1024
   MAX_CONTEXT_LENGTH: int = 4096

   # Generation
   TEMPERATURE: float = 0.7
   TOP_P: float = 0.95
   TOP_K: int = 50
   DO_SAMPLE: bool = True
   NUM_BEAMS: int = 1
   MIN_NEW_TOKENS: int = 30
   MAX_NEW_TOKENS: int = 150

   # Chat Settings
   MAX_HISTORY_LENGTH: int = 10
   STREAM_CHUNK_SIZE: int = 20
   STREAM_DELAY: float = 0.05

   # Document Processing 
   PDF_IMAGE_QUALITY: int = 85
   PDF_MAX_IMAGE_SIZE: Tuple[int, int] = (800, 800)
   SUPPORTED_FORMATS: List[str] = ['.pdf', '.docx', '.txt']
   CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
   CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
   MAX_RELEVANT_DOCS: int = int(os.getenv("MAX_RELEVANT_DOCS", "5"))
   MAX_CHUNKS_PER_DOC: int = int(os.getenv("MAX_CHUNKS_PER_DOC", "50"))
   CONTEXT_CONFIDENCE_THRESHOLD: float = 0.7

   # Google Drive
   GOOGLE_DRIVE_FOLDER_ID: str = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
   GOOGLE_DRIVE_CREDENTIALS_PATH: str = os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH")
   GOOGLE_DRIVE_SYNC_INTERVAL: int = int(os.getenv("GOOGLE_DRIVE_SYNC_INTERVAL", "3600"))

   # Logging
   LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
   LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   LOG_FILE_MAX_BYTES: int = 10 * 1024 * 1024
   LOG_FILE_BACKUP_COUNT: int = 5

   # Metrics & Monitoring
   ENABLE_METRICS: bool = True
   METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9090"))
   HEALTH_CHECK_INTERVAL: int = 60

   # Server Performance
   WORKERS: int = int(os.getenv("WORKERS", "4"))
   TIMEOUT: int = int(os.getenv("TIMEOUT", "60"))
   KEEPALIVE: int = int(os.getenv("KEEPALIVE", "5"))
   MAX_REQUESTS: int = int(os.getenv("MAX_REQUESTS", "10000"))

   # Cache
   CACHE_TTL: int = 3600
   CACHE_PREFIX: str = "chat:"

   # Auto Cleanup
   CLEANUP_ENABLED: bool = True
   CLEANUP_INTERVAL: int = 24 * 60 * 60
   CLEANUP_THRESHOLD_DAYS: int = 30

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
           "num_beams": self.NUM_BEAMS
       }

settings = Settings()

# Create required directories
for path in [settings.LOG_DIR, settings.DATA_DIR, settings.CACHE_DIR, settings.MODEL_DIR]:
   path.mkdir(parents=True, exist_ok=True)
