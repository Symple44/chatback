# core/cache/memory_cache.py
from typing import Dict, Any, Optional, TypeVar, Generic
import time
import asyncio

from core.utils.logger import get_logger

logger = get_logger("memory_cache")

T = TypeVar('T')

class CacheEntry(Generic[T]):
    """Entrée de cache avec métadonnées."""
    
    def __init__(self, key: str, value: T, expires_at: Optional[float] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.expires_at = expires_at
        self.access_count = 0
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Vérifie si l'entrée est expirée."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def access(self) -> None:
        """Enregistre un accès à cette entrée."""
        self.access_count += 1
        self.last_accessed = time.time()

class MemoryCache:
    """Cache en mémoire pour les accès rapides."""
    
    def __init__(self, max_entries: int = 1000):
        """
        Initialise le cache mémoire.
        
        Args:
            max_entries: Nombre maximum d'entrées
        """
        self.max_entries = max_entries
        self.cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé à rechercher
            default: Valeur par défaut si la clé n'existe pas
            
        Returns:
            Valeur associée à la clé ou valeur par défaut
        """
        entry = self.cache.get(key)
        if entry:
            if entry.is_expired():
                # Supprimer l'entrée expirée
                await self.delete(key)
                self.stats["misses"] += 1
                return default
            else:
                # Mise à jour des statistiques
                entry.access()
                self.stats["hits"] += 1
                return entry.value
        
        self.stats["misses"] += 1
        return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Enregistre une valeur dans le cache.
        
        Args:
            key: Clé de l'entrée
            value: Valeur à stocker
            ttl: Durée de vie en secondes (None = pas d'expiration)
            
        Returns:
            True si réussi, False sinon
        """
        # Calculer la date d'expiration si un TTL est spécifié
        expires_at = None if ttl is None else time.time() + ttl
        
        # Gérer la limite du cache mémoire
        await self._manage_size()
        
        # Créer l'entrée
        entry = CacheEntry(key, value, expires_at)
        
        # Enregistrer dans le cache
        async with self._lock:
            self.cache[key] = entry
            self.stats["sets"] += 1
        
        return True
    
    async def delete(self, key: str) -> bool:
        """
        Supprime une entrée du cache.
        
        Args:
            key: Clé à supprimer
            
        Returns:
            True si la clé existait, False sinon
        """
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                self.stats["deletes"] += 1
                return True
        return False
    
    async def clear(self) -> None:
        """Vide le cache."""
        async with self._lock:
            self.cache.clear()
    
    async def _manage_size(self) -> None:
        """Gère la taille du cache en supprimant les entrées les moins utilisées."""
        if len(self.cache) < self.max_entries:
            return
            
        async with self._lock:
            # Supprimer les entrées expirées
            now = time.time()
            expired_keys = [k for k, v in self.cache.items() 
                          if v.expires_at and v.expires_at <= now]
            
            for key in expired_keys:
                del self.cache[key]
            
            # Si toujours trop d'entrées, supprimer les moins utilisées
            if len(self.cache) > self.max_entries:
                # Trier par nombre d'accès et date du dernier accès
                sorted_entries = sorted(
                    self.cache.items(),
                    key=lambda x: (x[1].access_count, x[1].last_accessed)
                )
                
                # Supprimer 20% des entrées les moins utilisées
                to_remove = int(len(sorted_entries) * 0.2)
                for key, _ in sorted_entries[:to_remove]:
                    del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        total_ops = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_ops if total_ops > 0 else 0
        
        return {
            "entries": len(self.cache),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": round(hit_rate * 100, 2),
            "sets": self.stats["sets"],
            "deletes": self.stats["deletes"]
        }