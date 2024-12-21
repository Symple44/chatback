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

    async def _connect_with_retry(self):
        """Établit la connexion Redis avec retry."""
        for attempt in range(self._init_retries):
            try:
                self._connection = redis.Redis(
                    host='localhost',  # Utilisez les paramètres de settings ici
                    port=6379,
                    password='',  # Ajoutez un mot de passe si nécessaire
                    db=0,
                    decode_responses=False,
                    socket_timeout=5.0,
                    socket_keepalive=True
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
