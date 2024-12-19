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

@router.post("/new", response_model=SessionResponse)
async def create_session(
    data: SessionCreate,
    background_tasks: BackgroundTasks,
    components=Depends(get_components)
) -> Dict:
    """
    Crée une nouvelle session pour l'utilisateur.
    
    Args:
        data: Données de création de session
        background_tasks: Tâches d'arrière-plan
        components: Composants de l'application
    
    Returns:
        Nouvelle session créée
    """
    try:
        metrics.increment_counter("session_creation_attempts")
        
        async with components.db.session_factory() as session:
            # Vérification de l'utilisateur
            user = await session.execute(
                select(User).where(User.id == data.user_id)
            )
            user = user.scalar_one_or_none()
            if not user:
                raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

            # Vérification des sessions actives existantes
            active_sessions = await session.execute(
                select(ChatSession).where(
                    and_(
                        ChatSession.user_id == data.user_id,
                        ChatSession.is_active == True,
                        ChatSession.updated_at >= datetime.utcnow() - timedelta(hours=24)
                    )
                )
            )
            active_sessions = active_sessions.scalars().all()

            # Gestion des sessions existantes
            if len(active_sessions) >= 5:  # Limite de sessions actives
                oldest_session = min(active_sessions, key=lambda s: s.updated_at)
                oldest_session.is_active = False
                await session.flush()
                metrics.increment_counter("session_auto_deactivations")

            # Création de la nouvelle session
            new_session = ChatSession(
                session_id=str(uuid.uuid4()),
                user_id=str(data.user_id),
                session_context={
                    "created_at": datetime.utcnow().isoformat(),
                    "source": data.metadata.get("source", "unknown"),
                    "history": [],
                    "preferences": data.metadata.get("preferences", {}),
                    "initial_context": data.initial_context or {}
                },
                metadata=data.metadata
            )
            
            session.add(new_session)
            await session.commit()
            await session.refresh(new_session)
            
            # Tâches d'arrière-plan
            background_tasks.add_task(
                cleanup_old_sessions,
                components,
                data.user_id
            )
            
            metrics.increment_counter("session_creations_success")
            logger.info(f"Nouvelle session créée: {new_session.session_id}")
            
            return new_session.to_dict()

    except HTTPException as he:
        raise he
    except Exception as e:
        metrics.increment_counter("session_creation_errors")
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
    
    Args:
        session_id: ID de la session
        include_history: Inclure l'historique des messages
        components: Composants de l'application
    
    Returns:
        Informations de la session
    """
    try:
        async with components.db.session_factory() as session:
            query = select(ChatSession).where(ChatSession.session_id == session_id)
            
            if include_history:
                query = query.options(selectinload(ChatSession.chat_history))
            
            result = await session.execute(query)
            chat_session = result.scalar_one_or_none()
            
            if not chat_session:
                raise HTTPException(status_code=404, detail="Session non trouvée")
            
            session_data = chat_session.to_dict()
            
            # Ajout des statistiques
            if include_history:
                session_data["stats"] = await get_session_stats(session, chat_session)
            
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

# Fonctions utilitaires

async def cleanup_old_sessions(components, user_id: str):
    """Nettoie les anciennes sessions inactives."""
    try:
        async with components.db.session_factory() as session:
            # Désactivation des sessions inactives
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            result = await session.execute(
                select(ChatSession).where(
                    and_(
                        ChatSession.user_id == user_id,
                        ChatSession.updated_at < cutoff_date,
                        ChatSession.is_active == True
                    )
                )
            )
            old_sessions = result.scalars().all()
            
            for old_session in old_sessions:
                old_session.is_active = False
                old_session.metadata["deactivated_at"] = datetime.utcnow().isoformat()
                old_session.metadata["deactivation_reason"] = "automatic_cleanup"
            
            await session.commit()
            
            if old_sessions:
                logger.info(f"Nettoyage: {len(old_sessions)} sessions désactivées pour l'utilisateur {user_id}")
                metrics.increment_counter("session_cleanups")
                
    except Exception as e:
        logger.error(f"Erreur nettoyage sessions: {e}")

async def get_session_stats(session, chat_session: ChatSession) -> Dict:
    """Calcule les statistiques d'une session."""
    try:
        # Statistiques des messages
        message_stats = await session.execute(
            select(
                func.count(ChatHistory.id).label('total_messages'),
                func.avg(ChatHistory.processing_time).label('avg_processing_time'),
                func.avg(ChatHistory.confidence_score).label('avg_confidence')
            ).where(ChatHistory.session_id == chat_session.session_id)
        )
        stats = message_stats.first()._asdict()
        
        # Durée de la session
        duration = datetime.utcnow() - chat_session.created_at
        stats['duration_hours'] = duration.total_seconds() / 3600
        
        # Autres métriques
        stats['is_active'] = chat_session.is_active
        stats['last_activity'] = chat_session.updated_at.isoformat()
        
        return stats
        
    except Exception as e:
        logger.error(f"Erreur calcul statistiques: {e}")
        return {}
