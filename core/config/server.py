# core/config/server.py
import os
from pydantic import BaseModel

class ServerConfig(BaseModel):
    """Configuration du serveur web."""
    
    # Basic settings
    WORKERS: int = int(os.getenv("WORKERS", "24"))
    TIMEOUT: int = int(os.getenv("TIMEOUT", "180"))
    KEEPALIVE: int = int(os.getenv("KEEPALIVE", "5"))
    MAX_REQUESTS: int = int(os.getenv("MAX_REQUESTS", "10000"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "64"))
    
    # Streaming configuration
    STREAM_BATCH_SIZE: int = int(os.getenv("STREAM_BATCH_SIZE", "10"))
    STREAM_MAX_QUEUE_SIZE: int = int(os.getenv("STREAM_MAX_QUEUE_SIZE", "1000"))
    STREAM_TIMEOUT: float = float(os.getenv("STREAM_TIMEOUT", "5.0"))
    STREAM_CHUNK_SIZE: int = int(os.getenv("STREAM_CHUNK_SIZE", "8192"))
    STREAM_KEEP_ALIVE: int = int(os.getenv("STREAM_KEEP_ALIVE", "15"))
    STREAM_RETRY_TIMEOUT: int = int(os.getenv("STREAM_RETRY_TIMEOUT", "30"))