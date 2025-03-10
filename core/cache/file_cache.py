# core/cache/file_cache.py
from typing import Dict, Any, Optional
import os
import json
import pickle
import hashlib
import time
from pathlib import Path
import aiofiles
import asyncio

from core.utils.logger import get_logger
from core.config.config import settings

logger = get_logger("file_cache")

class FileCache:
    """Cache fichier pour les données persistantes."""
    
    def __init__(self, namespace: str = "default"):
        """
        Initialise le cache fichier.
        
        Args:
            namespace: Espace de noms pour isoler différents caches
        """
        self.namespace = namespace
        self.cache_dir = Path(settings.CACHE_DIR) / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "index.json"
        self.index: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialise le cache fichier en chargeant l'index."""
        if not self._initialized:
            await self._load_index()
            self._initialized = True
    
    async def _load_index(self) -> None:
        """Charge l'index du cache fichier."""
        if not self.index_path.exists():
            self.index = {}
            return
            
        try:
            async with aiofiles.open(self.index_path, "r") as f:
                self.index = json.loads(await f.read())
                
            # Supprimer les entrées expirées
            now = time.time()
            keys_to_delete = []
            
            for key, metadata in self.index.items():
                if metadata.get("expires_at") and now > metadata["expires_at"]:
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                file_path = self._get_file_path(key)
                if file_path.exists():
                    os.remove(file_path)
                del self.index[key]
                
            # Sauvegarder l'index nettoyé
            await self._save_index()
                
        except Exception as e:
            logger.error(f"Erreur chargement index cache fichier: {e}")
            self.index = {}
    
    async def _save_index(self) -> None:
        """Sauvegarde l'index du cache fichier."""
        try:
            async with aiofiles.open(self.index_path, "w") as f:
                await f.write(json.dumps(self.index))
        except Exception as e:
            logger.error(f"Erreur sauvegarde index cache fichier: {e}")
    
    def _get_file_path(self, key: str) -> Path:
        """Obtient le chemin du fichier cache pour une clé."""
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed_key}.cache"
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé à rechercher
            default: Valeur par défaut si la clé n'existe pas
            
        Returns:
            Valeur associée à la clé ou valeur par défaut
        """
        await self.initialize()
        
        # Vérifier si la clé existe dans l'index
        metadata = self.index.get(key)
        if not metadata:
            self.stats["misses"] += 1
            return default
        
        # Vérifier si l'entrée est expirée
        if metadata.get("expires_at") and time.time() > metadata["expires_at"]:
            await self.delete(key)
            self.stats["misses"] += 1
            return default
        
        # Charger la valeur depuis le fichier
        file_path = self._get_file_path(key)
        if not file_path.exists():
            # Incohérence entre l'index et les fichiers
            async with self._lock:
                if key in self.index:
                    del self.index[key]
                    await self._save_index()
            self.stats["misses"] += 1
            return default
        
        try:
            async with aiofiles.open(file_path, "rb") as f:
                data = await f.read()
                value = pickle.loads(data)
                
            # Mettre à jour les statistiques d'accès
            async with self._lock:
                metadata["access_count"] = metadata.get("access_count", 0) + 1
                metadata["last_accessed"] = time.time()
                await self._save_index()
            
            self.stats["hits"] += 1
            return value
            
        except Exception as e:
            logger.error(f"Erreur lecture cache fichier pour {key}: {e}")
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
        await self.initialize()
        
        # Calculer la date d'expiration
        expires_at = None if ttl is None else time.time() + ttl
        
        try:
            # Sérialiser la valeur
            data = pickle.dumps(value)
            
            # Enregistrer dans le fichier
            file_path = self._get_file_path(key)
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(data)
            
            # Mettre à jour l'index
            async with self._lock:
                self.index[key] = {
                    "created_at": time.time(),
                    "expires_at": expires_at,
                    "access_count": 0,
                    "last_accessed": time.time()
                }
                await self._save_index()
            
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Erreur écriture cache fichier pour {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Supprime une entrée du cache.
        
        Args:
            key: Clé à supprimer
            
        Returns:
            True si la clé existait, False sinon
        """
        await self.initialize()
        
        existed = False
        
        # Supprimer de l'index
        async with self._lock:
            if key in self.index:
                del self.index[key]
                existed = True
                await self._save_index()
        
        # Supprimer le fichier
        file_path = self._get_file_path(key)
        if file_path.exists():
            try:
                os.remove(file_path)
                existed = True
            except Exception as e:
                logger.error(f"Erreur suppression fichier cache pour {key}: {e}")
        
        if existed:
            self.stats["deletes"] += 1
        
        return existed
    
    async def clear(self) -> None:
        """Vide le cache fichier."""
        await self.initialize()
        
        # Supprimer tous les fichiers
        for file_path in self.cache_dir.glob("*.cache"):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Erreur suppression fichier cache {file_path}: {e}")
        
        # Supprimer l'index
        if self.index_path.exists():
            try:
                os.remove(self.index_path)
            except Exception as e:
                logger.error(f"Erreur suppression index cache: {e}")
        
        # Réinitialiser l'index en mémoire
        self.index = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        total_ops = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_ops if total_ops > 0 else 0
        
        # Compter les fichiers et calculer la taille
        try:
            files = list(self.cache_dir.glob("*.cache"))
            total_size = sum(f.stat().st_size for f in files)
        except Exception:
            files = []
            total_size = 0
        
        return {
            "entries": len(self.index),
            "files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": round(hit_rate * 100, 2),
            "sets": self.stats["sets"],
            "deletes": self.stats["deletes"]
        }