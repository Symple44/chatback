# core/utils/hardware_monitor.py
import psutil
import time
import asyncio
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

class HardwareMonitor:
    """Moniteur des ressources matérielles."""
    
    def __init__(self, interval: int = 30):
        """
        Initialise le moniteur matériel.
        
        Args:
            interval: Intervalle de surveillance en secondes
        """
        self.interval = interval
        self.running = False
        self.last_stats = {}
        self.monitoring_task = None
        self.monitor_lock = asyncio.Lock()
        
        self._hardware_info = {
            "cpu": {
                "model": os.environ.get("CPU_MODEL", "Non détecté"),
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True)
            },
            "ram": {
                "total": psutil.virtual_memory().total / (1024**3)
            },
            "gpu": {
                "model": os.environ.get("GPU_NAME", "Non détecté"),
                "available": CUDA_AVAILABLE
            }
        }
        
        # Compléter avec des informations GPU si disponibles
        if CUDA_AVAILABLE:
            try:
                device = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(device)
                self._hardware_info["gpu"].update({
                    "name": torch.cuda.get_device_name(device),
                    "vram_total": properties.total_memory / (1024**3),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "multi_processors": properties.multi_processor_count
                })
            except Exception as e:
                logger.warning(f"Impossible de récupérer les infos GPU: {e}")
    
    async def start_monitoring(self):
        """Démarre la surveillance en arrière-plan."""
        if self.running:
            return
            
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Surveillance matérielle démarrée")
    
    async def stop_monitoring(self):
        """Arrête la surveillance."""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        logger.info("Surveillance matérielle arrêtée")
    
    async def _monitoring_loop(self):
        """Boucle de surveillance des ressources."""
        while self.running:
            try:
                stats = await self.get_current_stats()
                async with self.monitor_lock:
                    self.last_stats = stats
                
                # Log des statistiques seulement si utilisation élevée
                if stats["cpu"]["percent"] > 80 or stats["ram"]["percent"] > 80:
                    logger.warning(f"Utilisation élevée des ressources: "
                                   f"CPU={stats['cpu']['percent']}%, "
                                   f"RAM={stats['ram']['percent']}%, "
                                   f"VRAM={stats.get('gpu', {}).get('vram_used_percent', 0)}%")
                    
            except Exception as e:
                logger.error(f"Erreur surveillance matérielle: {e}")
            
            await asyncio.sleep(self.interval)
    
    async def get_current_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques actuelles d'utilisation des ressources.
        
        Returns:
            Dict contenant les statistiques
        """
        try:
            # Statistiques CPU
            cpu_stats = {
                "percent": psutil.cpu_percent(interval=0.1),
                "per_core": psutil.cpu_percent(interval=0.1, percpu=True),
                "freq_current": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else None
            }
            
            # Statistiques RAM
            memory = psutil.virtual_memory()
            ram_stats = {
                "total": memory.total / (1024**3),
                "available": memory.available / (1024**3),
                "used": memory.used / (1024**3),
                "percent": memory.percent
            }
            
            # Statistiques GPU si disponible
            gpu_stats = {}
            if CUDA_AVAILABLE:
                try:
                    gpu_stats = {
                        "vram_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                        "vram_allocated": torch.cuda.memory_allocated() / (1024**3),
                        "vram_reserved": torch.cuda.memory_reserved() / (1024**3),
                        "vram_used_percent": torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
                    }
                except Exception as e:
                    logger.warning(f"Erreur récupération stats GPU: {e}")
            
            # Statistiques I/O
            io_stats = {
                "disk_read": sum(disk.read_bytes for disk in psutil.disk_io_counters(perdisk=True).values()) / (1024**3),
                "disk_write": sum(disk.write_bytes for disk in psutil.disk_io_counters(perdisk=True).values()) / (1024**3)
            }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu": cpu_stats,
                "ram": ram_stats,
                "gpu": gpu_stats,
                "io": io_stats
            }
            
        except Exception as e:
            logger.error(f"Erreur récupération statistiques matérielles: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Retourne les informations statiques sur le matériel.
        
        Returns:
            Dict contenant les informations matérielles
        """
        return self._hardware_info
    
    async def get_stats_history(self, minutes: int = 5) -> Dict[str, Any]:
        """
        Retourne l'historique des statistiques.
        
        Args:
            minutes: Nombre de minutes d'historique à récupérer
            
        Returns:
            Dict contenant l'historique des statistiques
        """
        # Cette méthode serait implémentée si vous souhaitez conserver un historique
        # Pour l'instant, retourne simplement les dernières statistiques
        async with self.monitor_lock:
            return self.last_stats

# Instance globale du moniteur
hardware_monitor = HardwareMonitor()