# api/routes/user_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, and_, func
import uuid
import os

from ..models.requests import UserCreate
from ..models.responses import UserResponse, ErrorResponse
from ..dependencies import get_components
from core.database.models import User, ChatSession, ChatHistory
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.database.base import get_session_manager

logger = get_logger("user_routes")
router = APIRouter(prefix="/users", tags=["users"])

@router.post("/new", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    background_tasks: BackgroundTasks,
    components=Depends(get_components)
) -> Dict:
    try:
        # Création de l'utilisateur
        new_user = await components.db_manager.create_user(user_data)
        if not new_user:
            raise HTTPException(status_code=500, detail="Erreur lors de la création de l'utilisateur")

        # Initialisation des ressources en arrière-plan
        background_tasks.add_task(
            components.db_manager.initialize_user_resources,
            str(new_user["id"])
        )

        return new_user

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'utilisateur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    include_stats: bool = Query(False),
    components=Depends(get_components)
) -> Dict:
    """
    Récupère les informations d'un utilisateur.
    """
    try:
        async with components.db.session_factory() as session:
            # Récupération utilisateur avec stats si demandé
            query = select(User).where(User.id == user_id)
            if include_stats:
                query = query.options(
                    selectinload(User.sessions),
                    selectinload(User.chat_history)
                )
            
            result = await session.execute(query)
            user = result.scalar_one_or_none()
            
            if not user:
                raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

            response = user.to_dict()
            
            if include_stats:
                response["stats"] = await calculate_user_statistics(session, user)
            
            return response
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'utilisateur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{user_id}")
async def update_user(
    user_id: str,
    user_data: Dict,
    components=Depends(get_components)
) -> Dict:
    """
    Met à jour les informations d'un utilisateur.
    """
    try:
        async with components.db.session_factory() as session:
            user = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = user.scalar_one_or_none()
            
            if not user:
                raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

            # Champs modifiables
            updatable_fields = {
                "username", "full_name", "metadata", "is_active"
            }

            for field, value in user_data.items():
                if field in updatable_fields:
                    # Vérification spéciale pour username
                    if field == "username" and value != user.username:
                        result = await session.execute(
                            select(User).where(User.username == value)
                        )
                        if result.scalar_one_or_none():
                            raise HTTPException(
                                status_code=400,
                                detail="Ce nom d'utilisateur est déjà pris"
                            )
                    
                    setattr(user, field, value)

            # Mise à jour des métadonnées
            user.metadata["last_updated"] = datetime.utcnow().isoformat()
            
            await session.commit()
            await session.refresh(user)
            
            logger.info(f"Utilisateur mis à jour: {user_id}")
            return user.to_dict()
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de l'utilisateur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    permanent: bool = Query(False),
    components=Depends(get_components)
) -> Dict:
    """Supprime ou désactive un utilisateur."""
    try:
        async with DatabaseSession() as session:
            user = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = user.scalar_one_or_none()
            
            if not user:
                raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

            if permanent:
                # Suppression permanente
                success = await components.db_manager.cleanup_user_data(user_id)
                if not success:
                    raise HTTPException(status_code=500, detail="Erreur lors de la suppression des données")
                await session.delete(user)
                message = "Utilisateur supprimé définitivement"
            else:
                # Désactivation
                user.is_active = False
                user.metadata["deactivated_at"] = datetime.utcnow().isoformat()
                message = "Utilisateur désactivé"

            await session.commit()
            logger.info(f"Utilisateur {user_id} {'supprimé' if permanent else 'désactivé'}")
            return {"message": message}
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de l'utilisateur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{user_id}/stats")
async def get_user_statistics(
    user_id: str,
    components=Depends(get_components)
) -> Dict:
    """Récupère les statistiques détaillées d'un utilisateur."""
    try:
        stats = await components.db_manager.calculate_user_statistics(user_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
        return stats
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des statistiques: {e}")
        raise HTTPException(status_code=500, detail=str(e))