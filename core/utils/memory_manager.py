# core/utils/memory_manager.py
import psutil
import gc
import os
import asyncio
import torch
from datetime import datetime, timedelta
from typing import Dict, Optional
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("memory_manager")

class MemoryManager:
    def __init__(self):
        """Initialise le gestionnaire de mémoire."""
        self.running = False
        self.last_usage = 0
        self.last_cleanup = datetime.now()
        self.warning_threshold = 75
        self.critical_threshold = 85
        self.cleanup_interval = 60  # secondes
        self.can_drop_caches = self._check_cache_access()
        self.metrics_history = []
        
    def _check_cache_access(self) -> bool:
        """Vérifie si l'accès au cache système est disponible."""
        try:
            with open('/proc/sys/vm/drop_caches', 'r') as f:
                f.read()
            return True
        except:
            logger.info("Accès au cache système non disponible - fonctionnalité désactivée")
            return False

    def _get_process_limits(self) -> Dict[str, float]:
        """Récupère les limites du processus."""
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            return {
                "soft_limit_gb": soft / (1024**3) if soft != resource.RLIM_INFINITY else float('inf'),
                "hard_limit_gb": hard / (1024**3) if hard != resource.RLIM_INFINITY else float('inf')
            }
        except Exception as e:
            logger.debug(f"Impossible de récupérer les limites du processus: {e}")
            return {"soft_limit_gb": float('inf'), "hard_limit_gb": float('inf')}

    def check_memory(self) -> dict:
        """Vérifie l'état détaillé de la mémoire."""
        try:
            mem = psutil.virtual_memory()
            process = psutil.Process()
            cpu_times = process.cpu_times()
            
            stats = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "ram_total_gb": mem.total / (1024**3),
                    "ram_used_percent": mem.percent,
                    "ram_available_gb": mem.available / (1024**3),
                    "ram_cached_gb": mem.cached / (1024**3) if hasattr(mem, 'cached') else 0,
                    "cpu_percent": psutil.cpu_percent(interval=0.1)
                },
                "process": {
                    "memory_gb": process.memory_info().rss / (1024**3),
                    "memory_percent": process.memory_percent(),
                    "cpu_percent": process.cpu_percent(),
                    "threads": len(process.threads()),
                    "user_time": cpu_times.user,
                    "system_time": cpu_times.system
                },
                "limits": self._get_process_limits()
            }
            
            # Ajout des métriques swap si disponibles
            try:
                swap = psutil.swap_memory()
                stats["system"]["swap_total_gb"] = swap.total / (1024**3)
                stats["system"]["swap_used_percent"] = swap.percent
            except:
                pass
            
            # Stockage historique
            self.metrics_history.append(stats)
            if len(self.metrics_history) > 100:  # Garde les 100 dernières mesures
                self.metrics_history.pop(0)
            
            return stats
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification mémoire: {e}")
            return self._get_default_stats()

    def _get_default_stats(self) -> dict:
        """Retourne des statistiques par défaut en cas d'erreur."""
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "ram_total_gb": 0,
                "ram_used_percent": 0,
                "ram_available_gb": 0,
                "ram_cached_gb": 0,
                "cpu_percent": 0
            },
            "process": {
                "memory_gb": 0,
                "memory_percent": 0,
                "cpu_percent": 0,
                "threads": 0,
                "user_time": 0,
                "system_time": 0
            },
            "limits": {
                "soft_limit_gb": float('inf'),
                "hard_limit_gb": float('inf')
            }
        }

    def optimize_system(self):
        """Optimise les paramètres système disponibles."""
        try:
            # Nettoyage Python
            gc.collect()
            
            # Optimisation torch si disponible
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Configuration des threads
            if hasattr(torch, 'set_num_threads'):
                torch.set_num_threads(settings.MAX_THREADS)
            
            # Tentative de libération du cache système si disponible
            if self.can_drop_caches:
                try:
                    os.system('sync')
                    with open('/proc/sys/vm/drop_caches', 'w') as f:
                        f.write('3')
                except Exception as e:
                    logger.debug(f"Impossible de libérer le cache système: {e}")
            
            logger.info("Optimisation système effectuée")
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'optimisation système: {e}")

    async def start_monitoring(self):
        """Démarre le monitoring de la mémoire."""
        self.running = True
        logger.info("Démarrage du monitoring mémoire")
        
        while self.running:
            try:
                stats = self.check_memory()
                
                # Log des métriques principales
                logger.info(
                    f"Mémoire - Système: {stats['system']['ram_used_percent']:.1f}% "
                    f"({stats['system']['ram_available_gb']:.1f}GB libre), "
                    f"Process: {stats['process']['memory_gb']:.1f}GB, "
                    f"CPU: {stats['system']['cpu_percent']}%"
                )
                
                # Vérification des seuils
                if stats['system']['ram_used_percent'] > self.critical_threshold:
                    logger.warning("Usage mémoire critique - nettoyage forcé")
                    await self.force_cleanup()
                elif stats['system']['ram_used_percent'] > self.warning_threshold:
                    logger.info("Usage mémoire élevé - nettoyage préventif")
                    await self.cleanup()
                
                # Nettoyage périodique
                if (datetime.now() - self.last_cleanup).seconds > self.cleanup_interval:
                    await self.cleanup()
                    self.last_cleanup = datetime.now()
                
            except Exception as e:
                logger.error(f"Erreur monitoring mémoire: {e}")
            
            await asyncio.sleep(settings.CLEANUP_INTERVAL)

    async def cleanup(self):
        """Nettoyage standard de la mémoire."""
        try:
            # Nettoyage Python
            gc.collect()
            
            # Nettoyage torch si disponible
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Nettoyage mémoire effectué")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage mémoire: {e}")

    async def force_cleanup(self):
        """Nettoyage forcé de la mémoire."""
        try:
            # Multiple passes de nettoyage Python
            for _ in range(3):
                gc.collect()
            
            # Optimisation système
            self.optimize_system()
            
            # Force le garbage collector
            gc.collect(generation=2)
            
            logger.info("Nettoyage forcé effectué")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage forcé: {e}")

    def get_metrics_history(self, minutes: Optional[int] = None) -> list:
        """Récupère l'historique des métriques."""
        if not minutes:
            return self.metrics_history
        
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            metric for metric in self.metrics_history
            if datetime.fromisoformat(metric['timestamp']) > cutoff
        ]

    def stop(self):
        """Arrête le monitoring."""
        self.running = False
        logger.info("Arrêt du monitoring mémoire")

    def __del__(self):
        """Nettoyage à la destruction de l'objet."""
        self.stop()