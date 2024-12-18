# api/routes/user_routes.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
from pydantic import BaseModel, EmailStr
import uuid
from datetime import datetime
from ..dependencies import get_components
from core.utils.logger import get_logger

logger = get_logger("user_routes")
router = APIRouter(prefix="/users", tags=["users"])

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    full_name: str
    preferences: Dict = {}

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: str
    created_at: datetime
    preferences: Dict

@router.post("/", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    components=Depends(get_components)
) -> Dict:
    """
    Crée un nouvel utilisateur.
    """
    try:
        # Vérifier si l'email existe déjà
        existing_user = await components.db_manager.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Un utilisateur avec cet email existe déjà"
            )

        # Création de l'utilisateur
        user_id = str(uuid.uuid4())
        user = {
            "id": user_id,
            "email": user_data.email,
            "username": user_data.username,
            "full_name": user_data.full_name,
            "created_at": datetime.utcnow(),
            "preferences": user_data.preferences
        }
        
        await components.db_manager.create_user(user)
        logger.info(f"Nouvel utilisateur créé: {user_id}")
        return user
        
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'utilisateur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{user_id}")
async def get_user(
    user_id: str,
    components=Depends(get_components)
) -> Dict:
    """
    Récupère les informations d'un utilisateur.
    """
    try:
        user = await components.db_manager.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
        return user
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'utilisateur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    components=Depends(get_components)
) -> Dict:
    """
    Supprime un utilisateur et toutes ses données associées.
    """
    try:
        # Vérifier si l'utilisateur existe
        user = await components.db_manager.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
        
        # Supprimer les sessions de l'utilisateur
        await components.session_manager.delete_user_sessions(user_id)
        
        # Supprimer l'historique de chat
        await components.db_manager.delete_user_history(user_id)
        
        # Supprimer l'utilisateur
        await components.db_manager.delete_user(user_id)
        
        logger.info(f"Utilisateur supprimé: {user_id}")
        return {"message": "Utilisateur supprimé avec succès"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de l'utilisateur: {e}")
        raise HTTPException(status_code=500, detail=str(e))
