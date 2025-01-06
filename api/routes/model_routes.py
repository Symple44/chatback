# api/routes/model_routes.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import asyncio
from datetime import datetime

from core.utils.logger import get_logger
from core.utils.metrics import metrics
from ..dependencies import get_components
from ..models.responses import ErrorResponse

logger = get_logger("model_routes")
router = APIRouter(prefix="/models", tags=["models"])

class ModelInfo(BaseModel):
    """Information sur un modèle."""
    name: str = Field(..., description="Nom du modèle")
    display_name: str = Field(..., description="Nom d'affichage")
    type: str = Field(..., description="Type de modèle (chat, embedding, etc.)")
    language: List[str] = Field(..., description="Langues supportées")
    context_length: int = Field(..., description="Longueur maximale du contexte")
    quantization: Optional[str] = Field(None, description="Type de quantization")
    capabilities: List[str] = Field(..., description="Capacités du modèle")
    gpu_requirements: Dict = Field(..., description="Prérequis GPU")
    is_active: bool = Field(True, description="Si le modèle est actif")

@router.get("/available", response_model=List[ModelInfo])
async def list_available_models(
    components=Depends(get_components)
) -> List[Dict]:
    """Liste tous les modèles disponibles."""
    try:
        return await components.model_manager.get_available_models()
    except Exception as e:
        logger.error(f"Erreur listing modèles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/current")
async def get_current_model(
    components=Depends(get_components)
) -> Dict:
    """Obtient les informations sur le modèle actuel."""
    try:
        return components.model_manager.get_model_info()
    except Exception as e:
        logger.error(f"Erreur récupération modèle actuel: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/change/{model_name}")
async def change_model(
    model_name: str,
    background_tasks: BackgroundTasks,
    components=Depends(get_components)
) -> Dict:
    """Change le modèle principal."""
    try:
        # Vérification du modèle
        if not await components.model_manager.is_model_available(model_name):
            raise HTTPException(
                status_code=400,
                detail=f"Modèle {model_name} non disponible"
            )

        # Vérification des ressources
        if not await components.model_manager._check_system_resources():
            raise HTTPException(
                status_code=503,
                detail="Ressources système insuffisantes"
            )

        # Changement du modèle
        await components.model_manager.change_model(model_name)
        
        # Nettoyage en arrière-plan
        background_tasks.add_task(components.memory_manager.cleanup_memory)

        return {
            "status": "success",
            "message": f"Modèle changé pour {model_name}",
            "model_info": components.model_manager.get_model_info(model_name)
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur changement modèle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/{model_name}")
async def get_model_stats(
    model_name: str,
    components=Depends(get_components)
) -> Dict:
    """Obtient les statistiques d'un modèle."""
    try:
        return components.model_manager.get_model_info(model_name)
    except Exception as e:
        logger.error(f"Erreur récupération stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reload/{model_name}")
async def reload_model(
    model_name: str,
    force: bool = False,
    components=Depends(get_components)
) -> Dict:
    """Recharge un modèle."""
    try:
        await components.model_manager.load_model(model_name, force_reload=force)
        return {
            "status": "success",
            "message": f"Modèle {model_name} rechargé"
        }
    except Exception as e:
        logger.error(f"Erreur rechargement modèle: {e}")
        raise HTTPException(status_code=500, detail=str(e))