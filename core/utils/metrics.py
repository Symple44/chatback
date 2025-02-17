# core/utils/metrics.py
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
import time
import threading
import psutil
from collections import defaultdict

class Metrics:
    def __init__(self):
        # Métriques de base existantes
        self.counters: Dict[str, int] = {}
        self.timers: Dict[str, float] = {}
        self.gauges: Dict[str, Any] = {}
        
        # Nouvelles métriques pour la recherche
        self.search_metrics = defaultdict(lambda: defaultdict(float))
        self.cache_stats = defaultdict(int)
        self.request_tracking = {}
        
        # Protection thread
        self._lock = threading.Lock()

    async def initialize(self):
        """Initialise le système de métriques."""
        self.reset_all()

    def increment_counter(self, name: str, value: int = 1):
        """Incrémente un compteur de manière thread-safe."""
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + value

    def record_time(self, name: str, value: float):
        """Enregistre une durée de manière thread-safe."""
        with self._lock:
            self.timers[name] = value

    def set_gauge(self, name: str, value: Any):
        """Définit une valeur de gauge de manière thread-safe."""
        with self._lock:
            self.gauges[name] = value

    def get_metrics(self) -> Dict[str, Any]:
        """Retourne toutes les métriques de manière thread-safe."""
        with self._lock:
            return {
                'counters': self.counters.copy(),
                'timers': self.timers.copy(),
                'gauges': self.gauges.copy(),
                'search_metrics': self._get_search_metrics(),
                'system_metrics': self._get_system_metrics(),
                'timestamp': datetime.utcnow().isoformat()
            }

    def get_latest_timings(self) -> Dict[str, float]:
        """Retourne les derniers timings enregistrés."""
        with self._lock:
            return self.timers.copy()
    
    def get_timer_value(self, name: str) -> float:
        """Retourne la valeur d'un timer spécifique."""
        with self._lock:
            return self.timers.get(name, 0.0)

    @contextmanager
    def timer(self, name: str):
        """Context manager pour mesurer le temps d'exécution."""
        start = time.perf_counter()
        try:
            yield
        finally:
            with self._lock:
                self.timers[name] = time.perf_counter() - start

    # Nouvelles méthodes pour le tracking de recherche
    def start_request_tracking(self, request_id: str):
        """Démarre le suivi d'une requête."""
        with self._lock:
            self.request_tracking[request_id] = {
                "start_time": time.time(),
                "operations": []
            }

    def finish_request_tracking(self, request_id: str):
        """Termine le suivi d'une requête et enregistre les métriques."""
        with self._lock:
            if request_id in self.request_tracking:
                start_time = self.request_tracking[request_id]["start_time"]
                total_time = time.time() - start_time
                
                self.search_metrics["requests"]["total"] += 1
                self.search_metrics["requests"]["total_time"] += total_time
                self.search_metrics["requests"]["average_time"] = (
                    self.search_metrics["requests"]["total_time"] / 
                    self.search_metrics["requests"]["total"]
                )

                del self.request_tracking[request_id]

    def track_search_operation(
        self,
        method: str,
        success: bool,
        processing_time: float,
        results_count: int
    ):
        """Enregistre les métriques d'une opération de recherche."""
        with self._lock:
            self.search_metrics["methods"][method]["total"] += 1
            self.search_metrics["methods"][method]["successes"] += int(success)
            self.search_metrics["methods"][method]["total_time"] += processing_time
            self.search_metrics["methods"][method]["results_count"] += results_count
            
            total = self.search_metrics["methods"][method]["total"]
            if total > 0:
                self.search_metrics["methods"][method]["average_time"] = (
                    self.search_metrics["methods"][method]["total_time"] / total
                )
                self.search_metrics["methods"][method]["success_rate"] = (
                    self.search_metrics["methods"][method]["successes"] / total
                )

    def track_cache_operation(self, hit: bool):
        """Enregistre une opération de cache."""
        with self._lock:
            self.cache_stats["total"] += 1
            if hit:
                self.cache_stats["hits"] += 1

    def _get_search_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de recherche agrégées."""
        total_searches = sum(
            self.search_metrics["methods"][method]["total"]
            for method in self.search_metrics["methods"]
        )
        
        if total_searches == 0:
            return {
                "total_searches": 0,
                "success_rate": 0,
                "average_time": 0,
                "cache_hit_rate": 0,
                "methods": {}
            }

        total_successes = sum(
            self.search_metrics["methods"][method]["successes"]
            for method in self.search_metrics["methods"]
        )

        cache_hit_rate = (
            self.cache_stats["hits"] / self.cache_stats["total"]
            if self.cache_stats["total"] > 0 else 0
        )

        return {
            "total_searches": total_searches,
            "success_rate": total_successes / total_searches,
            "average_time": sum(
                self.search_metrics["methods"][method]["total_time"]
                for method in self.search_metrics["methods"]
            ) / total_searches,
            "cache_hit_rate": cache_hit_rate,
            "methods": {
                method: {
                    "total": stats["total"],
                    "success_rate": stats.get("success_rate", 0),
                    "average_time": stats.get("average_time", 0),
                    "results_per_search": stats["results_count"] / stats["total"]
                        if stats["total"] > 0 else 0
                }
                for method, stats in self.search_metrics["methods"].items()
            }
        }

    def _get_system_metrics(self) -> Dict[str, float]:
        """Retourne les métriques système."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "memory": {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            },
            "cpu_percent": process.cpu_percent()
        }

    def reset_all(self):
        """Réinitialise toutes les métriques."""
        with self._lock:
            self.counters.clear()
            self.timers.clear()
            self.gauges.clear()
            self.search_metrics.clear()
            self.cache_stats.clear()
            self.request_tracking.clear()

# Instance unique des métriques
metrics = Metrics()