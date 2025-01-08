# api/routes/model_routes.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import asyncio
from datetime import datetime

from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.llm.model_loader import ModelType
from ..dependencies import get_components
from ..models.responses import ErrorResponse

logger = get_logger("model_routes")
router = APIRouter(prefix="/models", tags=["models"])

class ModelInfo(BaseModel):
    """Information sur un modèle."""
    name: str = Field(..., description="Nom du modèle")
    display_name: str = Field(..., description="Nom d'affichage")
    type: str = Field(..., description="Type de modèle (chat, embedding, etc.)")
    languages: List[str] = Field(..., description="Langues supportées")
    context_length: int = Field(..., description="Longueur maximale du contexte")
    quantization: Optional[str] = Field(None, description="Type de quantization")
    capabilities: List[str] = Field(..., description="Capacités du modèle")
    gpu_requirements: Dict = Field(..., description="Prérequis GPU")
    is_active: bool = Field(True, description="Si le modèle est actif")

@router.get("/available", response_model=List[ModelInfo])
async def list_available_models(components=Depends(get_components)) -> List[Dict]:
    """Liste tous les modèles disponibles."""
    try:
        models_info = []
        for model_type in ModelType:
            model_configs = components.model_manager._get_available_models_by_type(model_type)
            for name, config in model_configs.items():
                models_info.append(
                    ModelInfo(
                        name=name,
                        display_name=config.get("display_name", name),
                        type=model_type.value,
                        languages=config.get("languages", []),
                        context_length=config.get("context_length", 2048),
                        quantization=config.get("quantization"),
                        capabilities=config.get("capabilities", []),
                        gpu_requirements=config.get("gpu_requirements", {}),
                        is_active=components.model_manager.get_current_model(model_type) == name
                    )
                )
        return models_info
    except Exception as e:
        logger.error(f"Erreur listing modèles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/current")
async def get_current_model(components=Depends(get_components)) -> Dict:
    """Obtient les informations sur les modèles actuels."""
    try:
        current_models = {}
        for model_type in ModelType:
            model = components.model_manager.current_models.get(model_type)
            if model:
                current_models[model_type.value] = {
                    "name": model.model_name,
                    "device": str(model.device),
                    "info": components.model_manager.get_model_info(model_type)
                }
        return {
            "current_models": current_models,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Erreur récupération modèles actuels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/change/{model_type}/{model_name}")
async def change_model(
    model_type: str,
    model_name: str,
    background_tasks: BackgroundTasks,
    keep_old: bool = False,
    components=Depends(get_components)
) -> Dict:
    """Change un modèle spécifique."""
    try:
        # Conversion du type de modèle
        try:
            model_type_enum = ModelType[model_type.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Type de modèle invalide. Types valides: {[t.value for t in ModelType]}"
            )

        # Vérification de la disponibilité du modèle
        if not components.model_manager.is_model_available(model_name, model_type_enum):
            raise HTTPException(
                status_code=400,
                detail=f"Modèle {model_name} non disponible pour le type {model_type}"
            )

        # Changement du modèle
        model = await components.model_manager.change_model(
            model_name=model_name,
            model_type=model_type_enum,
            keep_old=keep_old
        )

        # Nettoyage en arrière-plan si nécessaire
        if not keep_old:
            background_tasks.add_task(components.cuda_manager.cleanup_memory)

        return {
            "status": "success",
            "message": f"Modèle changé pour {model_name}",
            "model_info": components.model_manager.get_model_info(model_type_enum),
            "device": str(model.device),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Erreur changement modèle: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du changement de modèle: {str(e)}"
        )

@router.get("/stats/{model_type}")
async def get_model_stats(
    model_type: str,
    components=Depends(get_components)
) -> Dict:
    try:
        model_type_enum = ModelType[model_type.upper()]
        model = components.model_manager.current_models.get(model_type_enum)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Aucun modèle actif pour le type {model_type}"
            )

        # Mise à jour des stats mémoire
        components.cuda_manager.update_memory_stats()
        
        # Test de génération simple pour vérifier la santé
        test_result = await components.model_manager.test_model_health(
            model_type_enum,
            "Test de santé du modèle."
        )

        # Récupération des infos GPU plus détaillées
        gpu_info = {}
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gpu_info = {
                "name": torch.cuda.get_device_name(device),
                "total_memory": torch.cuda.get_device_properties(device).total_memory / (1024**3),
                "allocated_memory": torch.cuda.memory_allocated(device) / (1024**3),
                "reserved_memory": torch.cuda.memory_reserved(device) / (1024**3),
                "max_allocated": torch.cuda.max_memory_allocated(device) / (1024**3)
            }

        return {
            "model_name": model.model_name,
            "device": str(model.device),
            "info": components.model_manager.get_model_info(model_type_enum),
            "memory_stats": {
                "cuda": gpu_info,
                "ram": components.cuda_manager.memory_stats["ram"]
            },
            "health_check": {
                "status": "healthy" if test_result else "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "test_result": test_result
            }
        }

    except Exception as e:
        logger.error(f"Erreur récupération stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
@router.post("/reload/{model_type}")
async def reload_model(
    model_type: str,
    force: bool = False,
    components=Depends(get_components)
) -> Dict:
    """Recharge un modèle spécifique."""
    try:
        # Conversion du type de modèle
        try:
            model_type_enum = ModelType[model_type.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Type de modèle invalide. Types valides: {[t.value for t in ModelType]}"
            )

        # Récupération du modèle actuel
        current_model = components.model_manager.current_models.get(model_type_enum)
        if not current_model:
            raise HTTPException(
                status_code=404,
                detail=f"Aucun modèle actif pour le type {model_type}"
            )

        # Rechargement du modèle
        model = await components.model_manager.change_model(
            model_name=current_model.model_name,
            model_type=model_type_enum,
            force_reload=True
        )

        return {
            "status": "success",
            "message": f"Modèle {current_model.model_name} rechargé",
            "model_info": components.model_manager.get_model_info(model_type_enum),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Erreur rechargement modèle: {e}")
        raise HTTPException(status_code=500, detail=str(e))