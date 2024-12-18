from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time
from collections import defaultdict
import statistics
import logging
import json
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)
class Timer:
    """Context manager for timing code execution."""
    def __init__(self, metric_name: str, metrics_manager):
        self.metric_name = metric_name
        self.metrics_manager = metrics_manager
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        self.metrics_manager.timers[self.metric_name] = elapsed_time
        logger.info(f"Metric '{self.metric_name}' took {elapsed_time:.2f} seconds.")
        
class MetricsManager:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.gauges = {}
        self.timers = {}
        self.start_time = datetime.utcnow()
        self.last_save = self.start_time
        self._setup_storage()

    def _setup_storage(self):
        """Configure le stockage des métriques."""
        self.metrics_dir = "metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)

    @contextmanager
    def timer(self, operation: str):
        """
        Contextmanager pour mesurer le temps d'exécution d'une opération.
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timing(operation, duration)
            self.timers[operation] = duration

    def record_timing(self, metric_name: str, value: float):
        """
        Enregistre une mesure de temps.
        """
        self.metrics[f"{metric_name}_timing"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "value": value
        })
        
    def get_latest_timings(self) -> Dict[str, float]:
        """
        Récupère les derniers timings pour toutes les opérations.
        """
        return self.timers.copy()

    def increment_counter(self, metric_name: str, value: int = 1):
        """
        Incrémente un compteur.
        
        Args:
            metric_name: Nom du compteur
            value: Valeur d'incrémentation
        """
        self.counters[metric_name] += value

    def set_gauge(self, metric_name: str, value: float):
        """
        Définit la valeur d'une jauge.
        
        Args:
            metric_name: Nom de la jauge
            value: Valeur à définir
        """
        self.gauges[metric_name] = {
            "timestamp": datetime.utcnow().isoformat(),
            "value": value
        }

    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """
        Calcule les statistiques pour une métrique donnée.
        
        Args:
            metric_name: Nom de la métrique
            
        Returns:
            Dictionnaire des statistiques
        """
        values = [m["value"] for m in self.metrics.get(f"{metric_name}_timing", [])]
        if not values:
            return {}

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Génère un résumé des métriques.
        
        Returns:
            Dictionnaire contenant le résumé des métriques
        """
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "counters": dict(self.counters),
            "gauges": self.gauges,
            "timings": {}
        }

        for metric_name in self.metrics:
            if metric_name.endswith("_timing"):
                base_name = metric_name[:-7]
                summary["timings"][base_name] = self.get_statistics(base_name)

        return summary

    def get_timer_value(self, operation: str) -> float:
        """
        Récupère la dernière valeur d'un timer.
        
        Args:
            operation: Nom de l'opération
            
        Returns:
            Durée de la dernière exécution ou 0 si non trouvée
        """
        return self.timers.get(operation, 0.0)
            
    def save_metrics(self, force: bool = False):
        """
        Sauvegarde les métriques dans un fichier.
        
        Args:
            force: Force la sauvegarde même si l'intervalle n'est pas atteint
        """
        now = datetime.utcnow()
        if not force and (now - self.last_save) < timedelta(minutes=5):
            return

        try:
            filename = os.path.join(
                self.metrics_dir,
                f"metrics_{now.strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(filename, 'w') as f:
                json.dump(self.get_metrics_summary(), f, indent=2)
            
            self.last_save = now
            logger.info(f"Métriques sauvegardées dans {filename}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des métriques: {e}")

    def cleanup_old_metrics(self, days: int = 7):
        """
        Nettoie les anciennes métriques.
        
        Args:
            days: Nombre de jours de rétention
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            for metric_name in list(self.metrics.keys()):
                self.metrics[metric_name] = [
                    m for m in self.metrics[metric_name]
                    if datetime.fromisoformat(m["timestamp"]) > cutoff
                ]
            
            # Nettoyage des fichiers
            for filename in os.listdir(self.metrics_dir):
                if filename.startswith("metrics_"):
                    file_path = os.path.join(self.metrics_dir, filename)
                    file_time = datetime.strptime(
                        filename[8:23],
                        '%Y%m%d_%H%M%S'
                    )
                    
                    if file_time < cutoff:
                        os.remove(file_path)
                        
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des métriques: {e}")

# Création d'une instance globale
metrics = MetricsManager()

# Exemple d'utilisation:
# with metrics.timer("process_query"):
#     result = process_query()
# metrics.increment_counter("queries_processed")
# metrics.set_gauge("memory_usage", get_memory_usage())