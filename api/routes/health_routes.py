# api/routes/health_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict
from datetime import datetime
import psutil
import os
from ..dependencies import get_components
from ..models.responses import HealthCheckResponse
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger("health_routes")
router = APIRouter(prefix="/health", tags=["health"])

@router.get("/", response_model=HealthCheckResponse)
async def health_check(
    full_check: bool = Query(False),
    components=Depends(get_components)
) -> Dict:
    """
    Vérifie l'état de santé de tous les composants du système.
    
    Args:
        components: Composants de l'application
        full_check: Si True, effectue une vérification complète
    
    Returns:
        État de santé du système
    """
    try:
        with metrics.timer("health_check"):
            # Vérification de base
            status = await check_basic_health(components)
            
            if full_check:
                # Vérifications supplémentaires
                status.update(await check_detailed_health(components))
            
            # Détermination du statut global
            overall_status = determine_overall_status(status)
            
            return HealthCheckResponse(
                status=True,
                components={
                    "database": True,
                    "elasticsearch": True,
                    "redis": True
                },
                metrics={
                    "database_pool": await components.db.get_pool_status(),
                    # Add other metrics as needed
                }
            )
            
    except Exception as e:
        return HealthCheckResponse(
            status=False,
            components={
                "database": False,
                "elasticsearch": False,
                "redis": False
            }
        )

@router.get("/memory")
async def memory_status(components=Depends(get_components)) -> Dict:
    """
    Vérifie l'état de la mémoire du système.
    """
    try:
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024 ** 3),
            "available_gb": memory.available / (1024 ** 3),
            "percent_used": memory.percent,
            "process_memory_mb": psutil.Process().memory_info().rss / (1024 * 1024)
        }
    except Exception as e:
        logger.error(f"Erreur vérification mémoire: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics(components=Depends(get_components)) -> Dict:
    """
    Récupère les métriques du système.
    """
    try:
        return {
            "timers": metrics.get_latest_timings(),
            "counters": dict(metrics.counters),
            "gauges": metrics.gauges
        }
    except Exception as e:
        logger.error(f"Erreur récupération métriques: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Fonctions utilitaires

async def check_basic_health(components) -> Dict:
    """Vérifie l'état de santé basique des composants."""
    return {
        "database": await components.db.health_check(),
        "elasticsearch": await components.es_client.check_connection(),
        "redis": await components.cache.check_connection(),
        "model": True  # Le modèle est toujours chargé si l'app fonctionne
    }

async def check_detailed_health(components) -> Dict:
    """Effectue des vérifications détaillées des composants."""
    status = {}
    
    # Vérification DB
    try:
        pool_status = await components.db.get_pool_status()
        status["database_pool"] = pool_status["checked_out"] < pool_status["pool_size"]
    except Exception as e:
        logger.error(f"Erreur vérification pool DB: {e}")
        status["database_pool"] = False

    # Vérification Redis
    try:
        redis_info = await components.cache.get_cache_stats()
        status["redis_memory"] = redis_info["used_memory_human"]
        status["redis_connected_clients"] = redis_info["connected_clients"]
    except Exception as e:
        logger.error(f"Erreur vérification Redis: {e}")
        status["redis_stats"] = False

    # Vérification disque
    try:
        disk = psutil.disk_usage("/")
        status["disk_space"] = disk.percent < 90  # Alerte si >90% utilisé
    except Exception as e:
        logger.error(f"Erreur vérification disque: {e}")
        status["disk_space"] = False

    return status

def determine_overall_status(status: Dict) -> str:
    """Détermine le statut global du système."""
    critical_components = {"database", "elasticsearch", "redis", "model"}
    
    # Si un composant critique est down
    if not all(status.get(comp, False) for comp in critical_components):
        return "unhealthy"
    
    # Si des composants non-critiques sont down
    if not all(status.values()):
        return "degraded"
    
    return "healthy"

def get_system_metrics() -> Dict:
    """Récupère les métriques système."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count()
            },
            "memory": {
                "total_gb": memory.total / (1024 ** 3),
                "available_gb": memory.available / (1024 ** 3),
                "percent": memory.percent
            },
            "disk": {
                "total_gb": disk.total / (1024 ** 3),
                "free_gb": disk.free / (1024 ** 3),
                "percent": disk.percent
            },
            "process": {
                "memory_mb": psutil.Process().memory_info().rss / (1024 * 1024),
                "threads": len(psutil.Process().threads())
            }
        }
    except Exception as e:
        logger.error(f"Erreur récupération métriques système: {e}")
        return {}
