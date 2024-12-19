# core/utils/metrics.py
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time
from collections import defaultdict
import statistics
import logging
import json
import os
from pathlib import Path
import threading
from contextlib import contextmanager
import psutil
import asyncio

from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("metrics")

class Timer:
    """Gestionnaire de contexte pour chronométrer le code."""
    def __init__(self, metric_name: str, metrics_manager):
        self.metric_name = metric_name
        self.metrics_manager = metrics_manager
        self.start_time = None

    def __enter__(self):
        """Démarre le chronométrage."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Termine le chronométrage et enregistre la durée."""
        if exc_type is None:
            duration = time.perf_counter() - self.start_time
            self.metrics_manager.record_timing(self.metric_name, duration)
        else:
            self.metrics_manager.increment_counter(f"{self.metric_name}_errors")

class MetricsManager:
    """Gestionnaire de métriques de l'application."""

    def __init__(self):
        """Initialise le gestionnaire de métriques."""
        self.metrics_dir = Path(settings.BASE_DIR) / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.gauges = {}
        self.timers = {}
        
        self.start_time = datetime.utcnow()
        self.last_save = self.start_time
        self.last_cleanup = self.start_time
        
        self._lock = threading.Lock()
        self._save_interval = timedelta(minutes=5)
        self._cleanup_interval = timedelta(days=7)
        self._task = None

    async def initialize(self):
        """Initialise les tâches périodiques."""
        if self._task is None:
            self._task = asyncio.create_task(self._periodic_tasks())
            logger.info("Tâches périodiques des métriques initialisées")

    @contextmanager
    def timer(self, operation: str):
        """Mesure le temps d'exécution d'une opération."""
        timer = Timer(operation, self)
        with timer:
            yield

    def record_timing(self, metric_name: str, value: float):
        """Enregistre une mesure de temps."""
        with self._lock:
            self.metrics[f"{metric_name}_timing"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "value": value
            })
            self.timers[metric_name] = value

    def increment_counter(self, metric_name: str, value: int = 1):
        """Incrémente un compteur."""
        with self._lock:
            self.counters[metric_name] += value

    def set_gauge(self, metric_name: str, value: float):
        """Définit la valeur d'une jauge."""
        with self._lock:
            self.gauges[metric_name] = {
                "timestamp": datetime.utcnow().isoformat(),
                "value": value
            }

    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Calcule les statistiques pour une métrique."""
        values = [m["value"] for m in self.metrics.get(f"{metric_name}_timing", [])]
        if not values:
            return {}

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "percentile_95": statistics.quantiles(values, n=20)[-1] if len(values) >= 20 else max(values)
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques système."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "disk": {
                    "total": disk.total,
                    "free": disk.free,
                    "percent": disk.percent
                },
                "network": self._get_network_stats(),
                "process": self._get_process_stats()
            }
        except Exception as e:
            logger.error(f"Erreur récupération métriques système: {e}")
            return {}

    def _get_network_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques réseau."""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "error_in": net_io.errin,
                "error_out": net_io.errout
            }
        except:
            return {}

    def _get_process_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques du processus."""
        try:
            process = psutil.Process()
            return {
                "memory_percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent(),
                "threads": len(process.threads()),
                "open_files": len(process.open_files()),
                "connections": len(process.connections())
            }
        except:
            return {}

    async def _periodic_tasks(self):
        """Exécute les tâches périodiques."""
        while True:
            try:
                now = datetime.utcnow()
                
                # Sauvegarde périodique
                if now - self.last_save >= self._save_interval:
                    await self.save_metrics()
                    self.last_save = now
                
                # Nettoyage périodique
                if now - self.last_cleanup >= self._cleanup_interval:
                    await self.cleanup_old_metrics()
                    self.last_cleanup = now
                
                await asyncio.sleep(60)  # Vérifie toutes les minutes
                
            except Exception as e:
                logger.error(f"Erreur dans les tâches périodiques: {e}")
                await asyncio.sleep(60)

    async def save_metrics(self):
        """Sauvegarde les métriques dans un fichier."""
        try:
            metrics_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
                "counters": dict(self.counters),
                "gauges": self.gauges,
                "timers": self.timers,
                "system_metrics": self.get_system_metrics()
            }
            
            filename = self.metrics_dir / f"metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"Métriques sauvegardées dans {filename}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde métriques: {e}")

    async def cleanup_old_metrics(self):
        """Nettoie les anciennes métriques."""
        try:
            cutoff = datetime.utcnow() - self._cleanup_interval
            
            # Nettoyage des métriques en mémoire
            with self._lock:
                for metric_name in list(self.metrics.keys()):
                    self.metrics[metric_name] = [
                        m for m in self.metrics[metric_name]
                        if datetime.fromisoformat(m["timestamp"]) > cutoff
                    ]
            
            # Nettoyage des fichiers
            for file in self.metrics_dir.glob("metrics_*.json"):
                file_date = datetime.strptime(
                    file.stem[8:],
                    '%Y%m%d_%H%M%S'
                )
                if file_date < cutoff:
                    file.unlink()
            
            logger.info("Nettoyage des anciennes métriques terminé")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage métriques: {e}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Retourne un instantané des métriques actuelles."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_hours": (datetime.utcnow() - self.start_time).total_seconds() / 3600,
            "counters": dict(self.counters),
            "gauges": self.gauges,
            "timers": self.timers,
            "system": self.get_system_metrics()
        }

# Instance unique des métriques
metrics = MetricsManager()
