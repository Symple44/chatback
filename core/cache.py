# core/cache.py
import redis
import json
from typing import Optional, Any, Dict
import logging
import asyncio
import zlib
import pickle

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self):
        """Initialise la connexion Redis."""
        self._init_retries = 3
        self._init_retry_delay = 1.0
        self._connection = None
        self._initialized = False

    async def initialize(self):
        """Initialisation asynchrone de la connexion Redis."""
        if not self._initialized:
            await self._connect_with_retry()
            self._initialized = True

    def _connect_with_retry(self):
        """Établit la connexion Redis avec retry."""
        for attempt in range(self._init_retries):
            try:
                self._connection = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD,
                    db=settings.REDIS_DB,
                    ssl=settings.REDIS_SSL,
                    decode_responses=False,
                    socket_timeout=5.0,
                    socket_keepalive=True,
                    socket_connect_timeout=2.0,
                    retry_on_timeout=True,
                    health_check_interval=30,
                    max_connections=50,
                    encoding='utf-8'
                )
                
                # Configuration du client
                self._connection.config_set('maxmemory', '512mb')
                self._connection.config_set('maxmemory-policy', 'allkeys-lru')
                
                if self._connection.ping():
                    logger.info("Connexion Redis établie avec succès")
                    metrics.increment_counter("redis_init_success")
                    return
                    
            except redis.RedisError as e:
                if attempt == self._init_retries - 1:
                    logger.error(f"Échec de connexion à Redis après {self._init_retries} tentatives: {e}")
                    metrics.increment_counter("redis_init_error")
                    raise
                logger.warning(f"Tentative {attempt + 1} échouée, nouvelle tentative dans {self._init_retry_delay}s")
                asyncio.sleep(self._init_retry_delay)

    async def get(self, key: str, default: Any = None) -> Optional[Any]:
        """Récupère une valeur du cache."""
        try:
            if not self._connection:
                return default

            data = self._connection.get(key)
            if data:
                try:
                    decompressed = zlib.decompress(data)
                    return pickle.loads(decompressed)
                except (zlib.error, pickle.PickleError) as e:
                    logger.error(f"Erreur décompression/désérialisation: {e}")
                    return default
            return default
            
        except Exception as e:
            logger.error(f"Erreur Redis get: {e}")
            return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Définit une valeur dans le cache."""
        try:
            if not self._connection:
                return False

            serialized = pickle.dumps(value)
            compressed = zlib.compress(serialized)
            
            return bool(self._connection.set(
                key,
                compressed,
                ex=ttl
            ))
            
        except Exception as e:
            logger.error(f"Erreur Redis set: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Supprime une clé du cache."""
        try:
            if self._connection:
                return bool(self._connection.delete(key))
            return False
        except Exception as e:
            logger.error(f"Erreur Redis delete: {e}")
            return False

    async def check_connection(self) -> bool:
        """Vérifie la connexion Redis."""
        try:
            return bool(self._connection and self._connection.ping())
        except:
            return False

    async def cleanup(self):
        """Nettoie les ressources."""
        if self._connection:
            self._connection.close()
            self._initialized = False
            logger.info("Connexion Redis fermée")

# Instance du cache
cache_instance = RedisCache()
