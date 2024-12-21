# core/utils/metrics.py
from typing import Dict, Any
from datetime import datetime
from contextlib import contextmanager
import time

class Metrics:
    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.timers: Dict[str, float] = {}
        self.gauges: Dict[str, Any] = {}

    async def initialize(self):
        """Initialise le système de métriques."""
        pass

    def increment_counter(self, name: str, value: int = 1):
        """Incrémente un compteur."""
        self.counters[name] = self.counters.get(name, 0) + value

    def record_time(self, name: str, value: float):
        """Enregistre une durée."""
        self.timers[name] = value

    def set_gauge(self, name: str, value: Any):
        """Définit une valeur de gauge."""
        self.gauges[name] = value

    def get_metrics(self) -> Dict[str, Any]:
        """Retourne toutes les métriques."""
        return {
            'counters': self.counters,
            'timers': self.timers,
            'gauges': self.gauges,
            'timestamp': datetime.utcnow().isoformat()
        }
    @contextmanager
    def timer(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timers[name] = time.perf_counter() - start

# Instance unique des métriques
metrics = Metrics()
