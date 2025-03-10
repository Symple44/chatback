# core/cache/redis_cache.py
import redis
import json
from typing import Optional, Any, Dict
import logging
import asyncio
import zlib
import pickle
from core.config.config import settings

logger = logging.getLogger(__name__)

class RedisCache:
    """Cache basé sur Redis avec compression et sérialisation avancée."""
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

    async def _connect_with_retry(self):
        """Établit la connexion Redis avec retry."""
        for attempt in range(self._init_retries):
            try:
                self._connection = redis.Redis(
                    host=settings.cache.REDIS_HOST,
                    port=settings.cache.REDIS_PORT,
                    password=settings.cache.REDIS_PASSWORD,
                    db=settings.cache.REDIS_DB,
                    ssl=settings.cache.REDIS_SSL,
                    decode_responses=False,
                    socket_timeout=5.0,
                    socket_keepalive=True,
                    socket_connect_timeout=2.0,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Vérification de la connexion
                if self._connection.ping():
                    logger.info("Connexion Redis établie avec succès")
                    return
                    
            except redis.RedisError as e:
                logger.warning(f"Tentative {attempt + 1} échouée: {e}")
                if attempt == self._init_retries - 1:
                    raise
                await asyncio.sleep(self._init_retry_delay)

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
                    await self.delete(key)  # Supprime la valeur corrompue
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
                ex=ttl or settings.cache.CACHE_TTL  # Utilise le TTL par défaut des settings
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

    def get_cache_stats(self) -> Dict[str, any]:
        """Récupère les statistiques du cache Redis."""
        try:
            if not self._connection:
                return {}
                
            info = self._connection.info()
            return {
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
                "total_connections_received": info.get("total_connections_received"),
                "total_commands_processed": info.get("total_commands_processed")
            }
        except Exception as e:
            logger.error(f"Erreur récupération stats cache: {e}")
            return {}

    async def cleanup(self):
        """Nettoie les ressources."""
        if self._connection:
            self._connection.close()
            self._initialized = False
            logger.info("Connexion Redis fermée")