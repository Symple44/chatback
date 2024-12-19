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
    """
    Crée un nouvel utilisateur.
    """
    try:
        metrics.increment_counter("user_creation_attempts")
        
        async with components.db.session_factory() as session:
            # Vérification de l'unicité de l'email
            result = await session.execute(
                select(User).where(User.email == user_data.email)
            )
            if result.scalar_one_or_none():
                raise HTTPException(
                    status_code=400,
                    detail="Un utilisateur avec cet email existe déjà"
                )

            # Vérification de l'unicité du username
            result = await session.execute(
                select(User).where(User.username == user_data.username)
            )
            if result.scalar_one_or_none():
                raise HTTPException(
                    status_code=400,
                    detail="Ce nom d'utilisateur est déjà pris"
                )

            # Création de l'utilisateur
            user = User(
                id=uuid.uuid4(),
                email=user_data.email,
                username=user_data.username,
                full_name=user_data.full_name,
                metadata={
                    **user_data.metadata,
                    "created_from": "api",
                    "created_at": datetime.utcnow().isoformat()
                },
                is_active=True
            )
            
            session.add(user)
            await session.commit()
            await session.refresh(user)
            
            # Initialisation des ressources en arrière-plan
            background_tasks.add_task(
                initialize_user_resources,
                components,
                str(user.id)
            )
            
            metrics.increment_counter("user_creations_success")
            logger.info(f"Nouvel utilisateur créé: {user.id}")
            
            return user.to_dict()
            
    except HTTPException as he:
        metrics.increment_counter("user_creation_validation_errors")
        raise he
    except Exception as e:
        metrics.increment_counter("user_creation_errors")
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
    """
    Supprime ou désactive un utilisateur.
    """
    try:
        async with components.db.session_factory() as session:
            user = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = user.scalar_one_or_none()
            
            if not user:
                raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

            if permanent:
                # Suppression permanente
                await cleanup_user_data(session, user_id)
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
            
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de l'utilisateur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{user_id}/stats")
async def get_user_statistics(
    user_id: str,
    components=Depends(get_components)
) -> Dict:
    """
    Récupère les statistiques détaillées d'un utilisateur.
    """
    try:
        async with components.db.session_factory() as session:
            user = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = user.scalar_one_or_none()
            
            if not user:
                raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

            return await calculate_user_statistics(session, user)
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des statistiques: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Fonctions utilitaires

async def initialize_user_resources(components, user_id: str):
    """Initialise les ressources pour un nouvel utilisateur."""
    try:
        # Création du répertoire utilisateur
        user_dir = f"data/users/{user_id}"
        os.makedirs(user_dir, exist_ok=True)
        
        # Configuration des préférences par défaut
        async with components.db.session_factory() as session:
            user = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = user.scalar_one_or_none()
            if user:
                user.metadata.update({
                    "preferences": {
                        "theme": "light",
                        "language": "fr",
                        "notifications": True
                    },
                    "resources": {
                        "user_dir": user_dir
                    }
                })
                await session.commit()
                
        logger.info(f"Ressources initialisées pour l'utilisateur {user_id}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des ressources: {e}")

async def calculate_user_statistics(session, user) -> Dict:
    """Calcule les statistiques complètes d'un utilisateur."""
    try:
        # Statistiques des sessions
        session_stats = await session.execute(
            select(
                func.count(ChatSession.id).label("total_sessions"),
                func.count(
                    ChatSession.id
                ).filter(ChatSession.is_active == True).label("active_sessions")
            ).where(ChatSession.user_id == user.id)
        )
        session_stats = session_stats.first()

        # Statistiques des messages
        message_stats = await session.execute(
            select(
                func.count(ChatHistory.id).label("total_messages"),
                func.avg(ChatHistory.processing_time).label("avg_processing_time"),
                func.avg(ChatHistory.confidence_score).label("avg_confidence")
            ).where(ChatHistory.user_id == user.id)
        )
        message_stats = message_stats.first()

        # Calcul des statistiques
        return {
            "sessions": {
                "total": session_stats.total_sessions,
                "active": session_stats.active_sessions
            },
            "messages": {
                "total": message_stats.total_messages,
                "avg_processing_time": round(message_stats.avg_processing_time, 3) if message_stats.avg_processing_time else 0,
                "avg_confidence": round(message_stats.avg_confidence, 3) if message_stats.avg_confidence else 0
            },
            "account": {
                "age_days": (datetime.utcnow() - user.created_at).days,
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "is_active": user.is_active
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul des statistiques: {e}")
        return {}

async def cleanup_user_data(session, user_id: str):
    """Nettoie toutes les données associées à un utilisateur."""
    try:
        # Suppression des sessions
        await session.execute(
            select(ChatSession).where(ChatSession.user_id == user_id)
        )
        
        # Suppression de l'historique
        await session.execute(
            select(ChatHistory).where(ChatHistory.user_id == user_id)
        )
        
        # Suppression du répertoire utilisateur
        user_dir = f"data/users/{user_id}"
        if os.path.exists(user_dir):
            import shutil
            shutil.rmtree(user_dir)
            
        logger.info(f"Données utilisateur nettoyées: {user_id}")
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des données: {e}")
        raise
