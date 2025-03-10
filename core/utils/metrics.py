# core/utils/metrics.py

from typing import Dict, Any, Optional, List, Union
import time
import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue

logger = logging.getLogger("metrics")

class Metrics:
    """
    Classe pour le suivi des métriques de l'application.
    Effectue le suivi des requêtes, des temps d'exécution, et des statistiques
    pour les différents composants de l'application.
    """
    def __init__(self):
        """Initialise le système de métriques."""
        # Métriques générales
        self.counters = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "pdf_table_extractions": 0,
            "pdf_extraction_errors": 0,
            "tables_detected": 0,
            "table_detection_errors": 0
        }
        
        # Métriques de recherche
        self.search_metrics = {
            "methods": {
                "rag": {"total": 0, "success": 0, "avg_time": 0.0, "avg_results": 0},
                "hybrid": {"total": 0, "success": 0, "avg_time": 0.0, "avg_results": 0},
                "semantic": {"total": 0, "success": 0, "avg_time": 0.0, "avg_results": 0},
                "ocr": {"total": 0, "success": 0, "avg_time": 0.0, "avg_results": 0},
                "camelot": {"total": 0, "success": 0, "avg_time": 0.0, "avg_results": 0},
                "tabula": {"total": 0, "success": 0, "avg_time": 0.0, "avg_results": 0},
                "pdfplumber": {"total": 0, "success": 0, "avg_time": 0.0, "avg_results": 0},
                "ai": {"total": 0, "success": 0, "avg_time": 0.0, "avg_results": 0},
                "auto": {"total": 0, "success": 0, "avg_time": 0.0, "avg_results": 0}
            },
            "total_searches": 0,
            "total_time": 0.0,
            "total_results": 0
        }
        
        # Métriques de modèles
        self.model_metrics = {
            "inference_count": 0,
            "total_inference_time": 0.0,
            "avg_inference_time": 0.0,
            "total_tokens_generated": 0,
            "avg_tokens_per_request": 0
        }
        
        # Suivi des requêtes actives (ID de requête -> infos)
        self.active_requests = {}
        
        # Métriques temporelles
        self.timers = {}
        
        # Verrou pour les opérations thread-safe
        self._lock = threading.RLock()
        
        # File pour les métriques à enregistrer
        self._metrics_queue = queue.Queue()
        self._metrics_worker = None
        self._stop_worker = threading.Event()
        
        # Chemin pour les fichiers de métriques
        self.metrics_dir = Path("logs/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Période d'agrégation en secondes
        self.aggregation_period = 300  # 5 minutes
        
        # Métriques agrégées
        self.aggregated_metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "period_seconds": self.aggregation_period,
            "counters": {},
            "search": {},
            "model": {},
            "timers": {}
        }
        
        # Horodatage de la dernière agrégation
        self.last_aggregation = datetime.utcnow()
    
    async def initialize(self):
        """Initialise le système de métriques de manière asynchrone."""
        logger.info("Initialisation du système de métriques")
        
        # Démarrer le worker de métriques en arrière-plan
        self._start_metrics_worker()
        
        # Planifier la première agrégation des métriques
        asyncio.create_task(self._schedule_metrics_aggregation())
        
        logger.info("Système de métriques initialisé")
    
    def _start_metrics_worker(self):
        """Démarre un thread worker pour traiter les métriques en arrière-plan."""
        if self._metrics_worker is None or not self._metrics_worker.is_alive():
            self._stop_worker.clear()
            self._metrics_worker = threading.Thread(target=self._process_metrics_queue)
            self._metrics_worker.daemon = True
            self._metrics_worker.start()
            logger.debug("Worker de métriques démarré")
    
    def _process_metrics_queue(self):
        """Traite la file d'attente des métriques en arrière-plan."""
        while not self._stop_worker.is_set():
            try:
                # Attendre un élément dans la file
                metric_item = self._metrics_queue.get(timeout=1.0)
                
                # Traiter l'élément
                metric_type = metric_item.get("type")
                if metric_type == "counter":
                    self._update_counter(metric_item["name"], metric_item["value"])
                elif metric_type == "timer":
                    self._record_timer(metric_item["name"], metric_item["value"])
                
                # Marquer comme traité
                self._metrics_queue.task_done()
            except queue.Empty:
                # Aucun élément dans la file, continuer
                pass
            except Exception as e:
                logger.error(f"Erreur dans le worker de métriques: {e}")
    
    def _update_counter(self, name: str, value: int = 1):
        """Met à jour un compteur de métriques."""
        with self._lock:
            if name in self.counters:
                self.counters[name] += value
            else:
                self.counters[name] = value
    
    def _record_timer(self, name: str, value: float):
        """Enregistre une mesure de temps."""
        with self._lock:
            if name not in self.timers:
                self.timers[name] = {"count": 0, "total_time": 0.0, "avg_time": 0.0, "min": float("inf"), "max": 0.0}
            
            metrics = self.timers[name]
            metrics["count"] += 1
            metrics["total_time"] += value
            metrics["avg_time"] = metrics["total_time"] / metrics["count"]
            metrics["min"] = min(metrics["min"], value)
            metrics["max"] = max(metrics["max"], value)
    
    async def _schedule_metrics_aggregation(self):
        """Planifie l'agrégation périodique des métriques."""
        while True:
            try:
                # Attendre la période d'agrégation
                await asyncio.sleep(self.aggregation_period)
                
                # Exécuter l'agrégation
                await self.aggregate_metrics()
            except Exception as e:
                logger.error(f"Erreur lors de l'agrégation des métriques: {e}")
                await asyncio.sleep(60)  # Attendre un peu avant de réessayer
    
    async def aggregate_metrics(self):
        """Agrège les métriques et les enregistre dans un fichier."""
        now = datetime.utcnow()
        
        with self._lock:
            # Calculer la période réelle
            period_seconds = (now - self.last_aggregation).total_seconds()
            
            # Créer une copie des métriques actuelles
            self.aggregated_metrics = {
                "timestamp": now.isoformat(),
                "period_seconds": period_seconds,
                "counters": self.counters.copy(),
                "search": {
                    "methods": {k: v.copy() for k, v in self.search_metrics["methods"].items()},
                    "total_searches": self.search_metrics["total_searches"],
                    "total_time": self.search_metrics["total_time"],
                    "total_results": self.search_metrics["total_results"]
                },
                "model": self.model_metrics.copy(),
                "timers": {k: v.copy() for k, v in self.timers.items()}
            }
            
            # Mettre à jour l'horodatage
            self.last_aggregation = now
        
        # Enregistrer les métriques dans un fichier
        try:
            filename = now.strftime("%Y%m%d_%H%M%S") + "_metrics.json"
            filepath = self.metrics_dir / filename
            
            with open(filepath, "w") as f:
                json.dump(self.aggregated_metrics, f, indent=2)
            
            logger.debug(f"Métriques agrégées enregistrées dans {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement des métriques: {e}")
    
    def increment_counter(self, name: str, value: int = 1):
        """
        Incrémente un compteur de métriques.
        
        Args:
            name: Nom du compteur
            value: Valeur d'incrémentation (par défaut: 1)
        """
        # Ajouter à la file d'attente pour traitement en arrière-plan
        self._metrics_queue.put({
            "type": "counter",
            "name": name,
            "value": value
        })
    
    def start_request_tracking(self, request_id: str):
        """
        Commence à suivre une requête.
        
        Args:
            request_id: Identifiant de la requête
        """
        with self._lock:
            self.active_requests[request_id] = {
                "start_time": time.time(),
                "status": "processing"
            }
            self.counters["total_requests"] += 1
    
    def finish_request_tracking(self, request_id: str, success: bool = True):
        """
        Termine le suivi d'une requête.
        
        Args:
            request_id: Identifiant de la requête
            success: Si la requête a réussi ou échoué
        """
        with self._lock:
            if request_id in self.active_requests:
                request_data = self.active_requests[request_id]
                elapsed_time = time.time() - request_data["start_time"]
                
                # Mettre à jour les compteurs
                if success:
                    self.counters["successful_requests"] += 1
                else:
                    self.counters["failed_requests"] += 1
                
                # Enregistrer le temps d'exécution
                self._record_timer("request_time", elapsed_time)
                
                # Supprimer la requête active
                del self.active_requests[request_id]
    
    def track_cache_operation(self, hit: bool = True):
        """
        Suit une opération de cache.
        
        Args:
            hit: Si l'opération est un hit (True) ou un miss (False)
        """
        with self._lock:
            if hit:
                self.counters["cache_hits"] += 1
            else:
                self.counters["cache_misses"] += 1
    
    def track_model_inference(self, inference_time: float, tokens_generated: int):
        """
        Suit une inférence de modèle.
        
        Args:
            inference_time: Temps d'inférence en secondes
            tokens_generated: Nombre de tokens générés
        """
        with self._lock:
            self.model_metrics["inference_count"] += 1
            self.model_metrics["total_inference_time"] += inference_time
            self.model_metrics["total_tokens_generated"] += tokens_generated
            
            # Mettre à jour les moyennes
            self.model_metrics["avg_inference_time"] = self.model_metrics["total_inference_time"] / self.model_metrics["inference_count"]
            self.model_metrics["avg_tokens_per_request"] = self.model_metrics["total_tokens_generated"] / self.model_metrics["inference_count"]
    
    def track_search_operation(
        self, 
        method: str, 
        success: bool, 
        processing_time: float, 
        results_count: int
    ):
        """
        Suit une opération de recherche.
        
        Args:
            method: Méthode de recherche utilisée
            success: Si la recherche a réussi
            processing_time: Temps de traitement en secondes
            results_count: Nombre de résultats trouvés
        """
        with self._lock:
            # S'assurer que la méthode est une chaîne de caractères
            method_str = str(method).lower() if isinstance(method, (str, float, int)) else "auto"
            
            # S'assurer que la méthode existe dans le dictionnaire
            if method_str not in self.search_metrics["methods"]:
                self.search_metrics["methods"][method_str] = {
                    "total": 0, 
                    "success": 0, 
                    "avg_time": 0.0, 
                    "avg_results": 0
                }
            
            # Mettre à jour les métriques de la méthode
            method_metrics = self.search_metrics["methods"][method_str]
            method_metrics["total"] += 1
            
            if success:
                method_metrics["success"] += 1
            
            # Mettre à jour les moyennes
            current_avg_time = method_metrics["avg_time"]
            current_count = method_metrics["total"]
            method_metrics["avg_time"] = (current_avg_time * (current_count - 1) + processing_time) / current_count
            
            current_avg_results = method_metrics["avg_results"]
            method_metrics["avg_results"] = (current_avg_results * (current_count - 1) + results_count) / current_count
            
            # Mettre à jour les métriques globales
            self.search_metrics["total_searches"] += 1
            self.search_metrics["total_time"] += processing_time
            self.search_metrics["total_results"] += results_count
    
    @contextlib.contextmanager
    def timer(self, name: str):
        """
        Mesure le temps d'exécution d'un bloc de code.
        
        Args:
            name: Nom du timer
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            # Ajouter à la file d'attente pour traitement en arrière-plan
            self._metrics_queue.put({
                "type": "timer",
                "name": name,
                "value": elapsed_time
            })
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Obtient un résumé des métriques actuelles.
        
        Returns:
            Dictionnaire des métriques résumées
        """
        with self._lock:
            return {
                "counters": self.counters.copy(),
                "search": {
                    "methods": {k: v.copy() for k, v in self.search_metrics["methods"].items()},
                    "total_searches": self.search_metrics["total_searches"],
                    "avg_time": self.search_metrics["total_time"] / max(1, self.search_metrics["total_searches"]),
                    "avg_results": self.search_metrics["total_results"] / max(1, self.search_metrics["total_searches"])
                },
                "model": self.model_metrics.copy(),
                "timers": {k: v.copy() for k, v in self.timers.items()},
                "active_requests": len(self.active_requests)
            }
    
    async def cleanup(self):
        """Nettoie les ressources utilisées par le système de métriques."""
        logger.info("Nettoyage du système de métriques")
        
        # Arrêter le worker de métriques
        self._stop_worker.set()
        if self._metrics_worker and self._metrics_worker.is_alive():
            self._metrics_worker.join(timeout=2.0)
        
        # Sauvegarder les métriques finales
        await self.aggregate_metrics()
        
        logger.info("Système de métriques nettoyé")

# Singleton pour les métriques
metrics = Metrics()

# Contexte pour la mesure du temps
import contextlib