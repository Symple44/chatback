# api/routes/user_routes.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
from pydantic import BaseModel, EmailStr
import uuid
from datetime import datetime
from ..dependencies import get_components
from core.utils.logger import get_logger
from core.database.models import User

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

@router.post("/new", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    components=Depends(get_components)
) -> Dict:
    """
    Crée un nouvel utilisateur.
    """
    try:
        async with components.db.session_factory() as session:
            # Vérifier si l'email existe déjà
            result = await session.execute(
                select(User).where(User.email == user_data.email)
            )
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                raise HTTPException(
                    status_code=400,
                    detail="Un utilisateur avec cet email existe déjà"
                )

            # Création de l'utilisateur
            user = User(
                id=uuid.uuid4(),
                email=user_data.email,
                username=user_data.username,
                full_name=user_data.full_name,
                preferences=user_data.preferences
            )
            
            session.add(user)
            await session.commit()
            await session.refresh(user)
            
            logger.info(f"Nouvel utilisateur créé: {user.id}")
            return user.to_dict()
            
    except HTTPException as he:
        raise he
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
        user = await components.db.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
        return user
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'utilisateur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/del/{user_id}")
async def delete_user(
    user_id: str,
    components=Depends(get_components)
) -> Dict:
    """
    Supprime un utilisateur et toutes ses données associées.
    """
    try:
        # Vérifier si l'utilisateur existe
        user = await components.db.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
        
        # Supprimer les sessions de l'utilisateur
        await components.session_manager.delete_user_sessions(user_id)
        
        # Supprimer l'historique de chat
        await components.db.delete_user_history(user_id)
        
        # Supprimer l'utilisateur
        await components.db.delete_user(user_id)
        
        logger.info(f"Utilisateur supprimé: {user_id}")
        return {"message": "Utilisateur supprimé avec succès"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de l'utilisateur: {e}")
        raise HTTPException(status_code=500, detail=str(e))
