# core/cache/cache_manager.py
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar
import asyncio
import json
import time
import hashlib
import functools
from functools import wraps

from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings

from .redis_cache import RedisCache
from .memory_cache import MemoryCache
from .file_cache import FileCache

logger = get_logger("cache_manager")

T = TypeVar('T')

class CacheManager:
    """
    Gestionnaire de cache multi-niveaux (mémoire, Redis, fichier).
    
    Ce gestionnaire coordonne plusieurs niveaux de cache pour offrir:
    - Performances avec cache mémoire
    - Partage entre processus avec Redis
    - Persistance avec cache fichier
    """
    
    def __init__(self, 
                 redis_cache: RedisCache,
                 namespace: str = "default",
                 memory_max_entries: int = 1000,
                 enable_file_cache: bool = True):
        """
        Initialise le gestionnaire de cache.
        
        Args:
            redis_cache: Instance du cache Redis
            namespace: Espace de noms pour isoler les différents caches
            memory_max_entries: Nombre maximum d'entrées en mémoire
            enable_file_cache: Si True, active le cache fichier
        """
        self.namespace = namespace
        self.redis_cache = redis_cache
        self.memory_cache = MemoryCache(max_entries=memory_max_entries)
        self.enable_file_cache = enable_file_cache
        
        if enable_file_cache:
            self.file_cache = FileCache(namespace=namespace)
        
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialise tous les caches."""
        if self._initialized:
            return
            
        async with self._lock:
            if self._initialized:
                return
                
            # Initialiser Redis
            await self.redis_cache.initialize()
            
            # Initialiser le cache fichier si activé
            if self.enable_file_cache:
                await self.file_cache.initialize()
            
            self._initialized = True
            logger.info(f"Gestionnaire de cache {self.namespace} initialisé")
    
    def _get_namespaced_key(self, key: str) -> str:
        """Obtient la clé avec namespace."""
        return f"{self.namespace}:{key}"
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur du cache, en essayant les différents niveaux.
        
        Args:
            key: Clé à rechercher
            default: Valeur par défaut si la clé n'existe pas
            
        Returns:
            Valeur associée à la clé ou valeur par défaut
        """
        await self.initialize()
        
        # 1. Essayer d'abord le cache mémoire (le plus rapide)
        value = await self.memory_cache.get(key)
        if value is not None:
            metrics.track_cache_operation(hit=True)
            return value
        
        # 2. Ensuite essayer Redis
        namespaced_key = self._get_namespaced_key(key)
        redis_value = await self.redis_cache.get(namespaced_key)
        if redis_value is not None:
            # Mettre en cache mémoire pour les accès futurs
            await self.memory_cache.set(key, redis_value)
            metrics.track_cache_operation(hit=True)
            return redis_value
        
        # 3. Enfin, essayer le cache fichier si activé
        if self.enable_file_cache:
            file_value = await self.file_cache.get(key)
            if file_value is not None:
                # Mettre en cache mémoire pour les accès futurs
                await self.memory_cache.set(key, file_value)
                metrics.track_cache_operation(hit=True)
                return file_value
        
        metrics.track_cache_operation(hit=False)
        return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, persist: bool = False) -> bool:
        """
        Enregistre une valeur dans le cache.
        
        Args:
            key: Clé de l'entrée
            value: Valeur à stocker
            ttl: Durée de vie en secondes (None = pas d'expiration)
            persist: Si True, sauvegarde également dans le cache fichier
            
        Returns:
            True si réussi, False sinon
        """
        await self.initialize()
        
        # 1. Enregistrer en mémoire
        memory_success = await self.memory_cache.set(key, value, ttl)
        
        # 2. Enregistrer dans Redis
        namespaced_key = self._get_namespaced_key(key)
        redis_success = await self.redis_cache.set(namespaced_key, value, ttl)
        
        # 3. Enregistrer dans le cache fichier si demandé
        file_success = True
        if persist and self.enable_file_cache:
            file_success = await self.file_cache.set(key, value, ttl)
        
        return memory_success and redis_success and (not persist or file_success)
    
    async def delete(self, key: str) -> bool:
        """
        Supprime une entrée du cache à tous les niveaux.
        
        Args:
            key: Clé à supprimer
            
        Returns:
            True si la clé existait, False sinon
        """
        await self.initialize()
        
        # Supprimer de tous les niveaux
        memory_deleted = await self.memory_cache.delete(key)
        
        namespaced_key = self._get_namespaced_key(key)
        redis_deleted = await self.redis_cache.delete(namespaced_key)
        
        file_deleted = False
        if self.enable_file_cache:
            file_deleted = await self.file_cache.delete(key)
        
        # Si au moins un niveau avait la clé
        return memory_deleted or redis_deleted or file_deleted
    
    async def clear(self) -> None:
        """Vide tous les niveaux de cache."""
        await self.initialize()
        
        # Vider chaque niveau
        await self.memory_cache.clear()
        
        # Note: Redis ne sera pas entièrement vidé, seulement les clés du namespace
        
        if self.enable_file_cache:
            await self.file_cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de tous les niveaux de cache."""
        await self.initialize()
        
        stats = {
            "namespace": self.namespace,
            "memory": self.memory_cache.get_stats()
        }
        
        # Ajouter les statistiques Redis si disponible
        redis_stats = await self.redis_cache.get_cache_stats()
        if redis_stats:
            stats["redis"] = redis_stats
        
        # Ajouter les statistiques du cache fichier si activé
        if self.enable_file_cache:
            stats["file"] = self.file_cache.get_stats()
        
        return stats
    
    async def cleanup(self) -> None:
        """Nettoie les ressources."""
        # Nettoyer Redis
        await self.redis_cache.cleanup()

def cached(namespace: str = "default", ttl: Optional[int] = 3600, key_builder: Optional[Callable] = None):
    """
    Décorateur pour mettre en cache le résultat d'une fonction ou d'une coroutine.
    
    Args:
        namespace: Espace de noms du cache
        ttl: Durée de vie en secondes (None = pas d'expiration)
        key_builder: Fonction pour générer la clé de cache
        
    Returns:
        Décorateur pour fonction ou coroutine
    """
    def decorator(func):
        # Créer une instance du cache
        from .redis_cache import RedisCache
        
        redis_cache = RedisCache()
        cache = CacheManager(redis_cache, namespace)
        
        # Construire la clé par défaut
        def default_key_builder(*args, **kwargs):
            arg_str = str(args) + str(sorted(kwargs.items()))
            key = f"{func.__module__}.{func.__name__}:{hashlib.md5(arg_str.encode()).hexdigest()}"
            return key
        
        key_func = key_builder or default_key_builder
        
        # Décorateur pour fonction asynchrone
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs)
            result = await cache.get(key)
            if result is not None:
                return result
            result = await func(*args, **kwargs)
            await cache.set(key, result, ttl)
            return result
        
        # Décorateur pour fonction synchrone
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs)
            
            # Créer une boucle si nécessaire
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(cache.get(key))
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            loop.run_until_complete(cache.set(key, result, ttl))
            return result
        
        # Utiliser le wrapper approprié
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator