# core/cache/__init__.py
from typing import Optional

from .redis_cache import RedisCache
from .memory_cache import MemoryCache
from .file_cache import FileCache
from .cache_manager import CacheManager

# Instances par défaut pour l'usage courant
redis_cache = RedisCache()
cache_manager = CacheManager(redis_cache=redis_cache)

__all__ = [
    'RedisCache',
    'MemoryCache',
    'FileCache',
    'CacheManager',
    'redis_cache',
    'cache_manager'
]