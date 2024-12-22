from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import and_, select, func
from sqlalchemy.orm import selectinload
import uuid
import json

from ..models.requests import SessionCreate
from ..models.responses import SessionResponse, ErrorResponse
from ..dependencies import get_components
from core.database.models import ChatSession, User, ChatHistory
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.database.base import get_session_manager

logger = get_logger("session_routes")
router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.post("/new")
async def create_session(
    data: SessionCreate,
    background_tasks: BackgroundTasks,
    components=Depends(get_components)
) -> Dict:
    """Crée une nouvelle session."""
    try:
        # Créer la session
        new_session = await components.session_manager.get_or_create_session(None, data.user_id, data.metadata)
        
        # Ajouter le nettoyage en background
        background_tasks.add_task(
            components.session_manager.cleanup_old_sessions,
            days=7,
            user_id=data.user_id
        )
        
        return new_session

    except Exception as e:
        logger.error(f"Erreur création session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}", response_model=List[SessionResponse])
async def get_user_sessions(
    user_id: str,
    active_only: bool = Query(True),
    limit: int = Query(10, ge=1, le=50),
    components=Depends(get_components)
) -> List[Dict]:
    """
    Récupère toutes les sessions d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        active_only: Filtrer uniquement les sessions actives
        limit: Nombre maximum de sessions à retourner
        components: Composants de l'application
    
    Returns:
        Liste des sessions de l'utilisateur
    """
    try:
        async with components.db.session_factory() as session:
            query = select(ChatSession).where(ChatSession.user_id == user_id)
            
            if active_only:
                query = query.where(ChatSession.is_active == True)
            
            query = (
                query.options(selectinload(ChatSession.chat_history))
                .order_by(ChatSession.updated_at.desc())
                .limit(limit)
            )
            
            result = await session.execute(query)
            sessions = result.scalars().all()
            
            return [s.to_dict() for s in sessions]

    except Exception as e:
        logger.error(f"Erreur récupération sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/del/{session_id}")
async def delete_session(
    session_id: str,
    purge: bool = Query(False),
    components=Depends(get_components)
) -> Dict:
    """
    Supprime ou désactive une session.
    
    Args:
        session_id: ID de la session
        purge: Si True, supprime définitivement la session
        components: Composants de l'application
    
    Returns:
        Message de confirmation
    """
    try:
        async with components.db.session_factory() as session:
            chat_session = await session.execute(
                select(ChatSession).where(ChatSession.session_id == session_id)
            )
            chat_session = chat_session.scalar_one_or_none()
            
            if not chat_session:
                raise HTTPException(status_code=404, detail="Session non trouvée")

            if purge:
                # Suppression complète
                await session.delete(chat_session)
                metrics.increment_counter("session_deletions")
                message = "Session supprimée définitivement"
            else:
                # Désactivation
                chat_session.is_active = False
                chat_session.metadata["deactivated_at"] = datetime.utcnow().isoformat()
                metrics.increment_counter("session_deactivations")
                message = "Session désactivée"

            await session.commit()
            logger.info(f"Session {session_id} {'supprimée' if purge else 'désactivée'}")
            
            return {"message": message}

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur suppression session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}")
async def get_session_info(
    session_id: str,
    include_history: bool = Query(False),
    components=Depends(get_components)
) -> Dict:
    """
    Récupère les informations d'une session spécifique.
    """
    try:
        # Récupérer la session
        session_data = await components.session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session non trouvée")

        # Ajouter les stats si demandées
        if include_history:
            session_data["stats"] = await components.session_manager.get_session_stats(session_id)

        return session_data

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur récupération session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{session_id}/update")
async def update_session(
    session_id: str,
    context_update: Dict,
    components=Depends(get_components)
) -> Dict:
    """
    Met à jour le contexte d'une session.
    
    Args:
        session_id: ID de la session
        context_update: Mise à jour du contexte
        components: Composants de l'application
    
    Returns:
        Session mise à jour
    """
    try:
        async with components.db.session_factory() as session:
            chat_session = await session.execute(
                select(ChatSession).where(ChatSession.session_id == session_id)
            )
            chat_session = chat_session.scalar_one_or_none()
            
            if not chat_session:
                raise HTTPException(status_code=404, detail="Session non trouvée")

            # Mise à jour du contexte
            current_context = chat_session.session_context
            current_context.update(context_update)
            
            chat_session.session_context = current_context
            chat_session.updated_at = datetime.utcnow()
            
            await session.commit()
            await session.refresh(chat_session)
            
            logger.info(f"Contexte de session mis à jour: {session_id}")
            return chat_session.to_dict()

    except Exception as e:
        logger.error(f"Erreur mise à jour session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
