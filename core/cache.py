# core/cache.py
import redis
import json
import zlib
from typing import Optional, Any, Dict, List, Union
from datetime import datetime, timedelta
import logging
import asyncio
from contextlib import asynccontextmanager
import hashlib
import pickle
from functools import wraps

from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger("cache")

def cache_key_generator(*args, **kwargs) -> str:
    """Génère une clé de cache unique basée sur les arguments."""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    key_string = ":".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()

class CacheDecorator:
    """Décorateur pour la mise en cache des résultats de fonction."""
    
    def __init__(self, ttl: Optional[int] = None, prefix: Optional[str] = None):
        self.ttl = ttl
        self.prefix = prefix
    
    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{self.prefix}:{cache_key_generator(*args, **kwargs)}" if self.prefix else cache_key_generator(*args, **kwargs)
            
            # Tentative de récupération depuis le cache
            result = await cache_instance.get(cache_key)
            if result is not None:
                metrics.increment_counter("cache_decorator_hits")
                return result
            
            # Exécution de la fonction et mise en cache
            result = await func(*args, **kwargs)
            if result is not None:
                await cache_instance.set(cache_key, result, ttl=self.ttl)
                metrics.increment_counter("cache_decorator_misses")
            
            return result
        return wrapper

class RedisCache:
    def __init__(self):
        """Initialise la connexion Redis avec gestion des erreurs et retry."""
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
        """Récupère une valeur du cache avec gestion des erreurs."""
        with metrics.timer("redis_get"):
            try:
                data = self._connection.get(key)
                if data is not None:
                    try:
                        # Décompression et désérialisation
                        decompressed = zlib.decompress(data)
                        result = pickle.loads(decompressed)
                        metrics.increment_counter("cache_hits")
                        return result
                    except (zlib.error, pickle.PickleError) as e:
                        logger.error(f"Erreur décompression/désérialisation pour {key}: {e}")
                        await self.delete(key)  # Supprime la valeur corrompue
                        return default
                
                metrics.increment_counter("cache_misses")
                return default
                
            except redis.RedisError as e:
                logger.error(f"Erreur Redis pour get({key}): {e}")
                metrics.increment_counter("cache_errors")
                return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
        compress_level: int = 6,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Définit une valeur dans le cache avec options avancées.
        
        Args:
            key: Clé du cache
            value: Valeur à stocker
            ttl: Durée de vie en secondes
            nx: True pour définir uniquement si la clé n'existe pas
            xx: True pour définir uniquement si la clé existe
            compress_level: Niveau de compression (0-9)
            tags: Liste de tags pour le regroupement
        """
        with metrics.timer("redis_set"):
            try:
                # Sérialisation et compression
                serialized = pickle.dumps(value)
                compressed = zlib.compress(serialized, level=compress_level)
                
                # Options de configuration
                set_options = {
                    'nx': nx,
                    'xx': xx,
                    'ex': ttl or settings.CACHE_TTL
                }
                
                # Stockage principal
                success = self._connection.set(key, compressed, **set_options)
                
                if success and tags:
                    # Gestion des tags
                    pipeline = self._connection.pipeline()
                    for tag in tags:
                        pipeline.sadd(f"tag:{tag}", key)
                        pipeline.expire(f"tag:{tag}", ttl or settings.CACHE_TTL)
                    pipeline.execute()
                
                if success:
                    metrics.increment_counter("cache_sets_success")
                else:
                    metrics.increment_counter("cache_sets_failed")
                
                return bool(success)
                
            except redis.RedisError as e:
                logger.error(f"Erreur Redis pour set({key}): {e}")
                metrics.increment_counter("cache_errors")
                return False

    async def delete_by_tags(self, tags: List[str]) -> int:
        """Supprime toutes les clés associées aux tags spécifiés."""
        try:
            keys_to_delete = set()
            for tag in tags:
                tagged_keys = self._connection.smembers(f"tag:{tag}")
                keys_to_delete.update(tagged_keys)
                self._connection.delete(f"tag:{tag}")
            
            if keys_to_delete:
                return self._connection.delete(*keys_to_delete)
            return 0
            
        except redis.RedisError as e:
            logger.error(f"Erreur lors de la suppression par tags: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Récupère des statistiques détaillées du cache."""
        try:
            info = self._connection.info()
            return {
                "memory": {
                    "used_memory": info.get("used_memory_human"),
                    "peak_memory": info.get("used_memory_peak_human"),
                    "fragmentation_ratio": info.get("mem_fragmentation_ratio"),
                    "evicted_keys": info.get("evicted_keys")
                },
                "stats": {
                    "total_connections": info.get("total_connections_received"),
                    "connected_clients": info.get("connected_clients"),
                    "ops_per_sec": info.get("instantaneous_ops_per_sec"),
                    "hit_rate": self._calculate_hit_rate(info),
                    "keyspace_hits": info.get("keyspace_hits"),
                    "keyspace_misses": info.get("keyspace_misses")
                },
                "server": {
                    "version": info.get("redis_version"),
                    "uptime_days": info.get("uptime_in_days"),
                    "connected_slaves": info.get("connected_slaves"),
                    "role": info.get("role")
                }
            }
        except redis.RedisError as e:
            logger.error(f"Erreur lors de la récupération des stats: {e}")
            return {}

    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calcule le taux de succès du cache."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return round((hits / total * 100), 2) if total > 0 else 0

    @asynccontextmanager
    async def lock(self, key: str, ttl: int = 30, retry_delay: float = 0.1, max_retries: int = 50):
        """
        Gestionnaire de verrou distribué avec retry.
        
        Usage:
            async with cache.lock("my_key"):
                # Code protégé
        """
        lock_key = f"lock:{key}"
        try:
            for _ in range(max_retries):
                if await self.set(lock_key, 1, ttl=ttl, nx=True):
                    try:
                        yield
                        break
                    finally:
                        await self.delete(lock_key)
                await asyncio.sleep(retry_delay)
            else:
                raise TimeoutError(f"Impossible d'acquérir le verrou pour {key}")
        except Exception as e:
            logger.error(f"Erreur avec le verrou pour {key}: {e}")
            raise

    async def cleanup(self) -> None:
        """Nettoie les ressources du cache."""
        try:
            if self._connection:
                self._connection.close()
                logger.info("Connexion Redis fermée proprement")
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du cache: {e}")

    def __del__(self):
        """Destructeur pour nettoyer la connexion."""
        try:
            if self._connection:
                self._connection.close()
        except:
            pass

# Instance singleton du cache
cache_instance = RedisCache()
