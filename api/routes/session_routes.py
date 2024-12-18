# api/routes/session_routes.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List
from pydantic import BaseModel
from datetime import datetime, timedelta
from sqlalchemy import and_
from sqlalchemy.future import select

from ..dependencies import get_components
from ..models.requests import SessionCreate
from ..models.responses import SessionResponse
from core.utils.logger import get_logger

logger = get_logger("session_routes")
router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.post("/new", response_model=SessionResponse)
async def create_session(data: SessionCreate, components=Depends(get_components)) -> Dict:
    """Crée une nouvelle session pour l'utilisateur."""
    try:
        # Vérification de l'utilisateur
        async with components.db.session_factory() as session:
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
            active_session = active_sessions.scalar_one_or_none()

            if active_session:
                # Réutilisation de la session existante
                return active_session.to_dict()

            # Création d'une nouvelle session
            new_session = ChatSession(
                user_id=data.user_id,
                session_id=str(uuid.uuid4()),
                session_context={
                    "created_at": datetime.utcnow().isoformat(),
                    "source": data.session_metadata.get("source", "unknown"),
                    "history": []
                },
                _metadata=data.session_metadata
            )
            session.add(new_session)
            await session.commit()
            await session.refresh(new_session)
            
            logger.info(f"Nouvelle session créée: {new_session.session_id}")
            return new_session.to_dict()
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur création session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}", response_model=List[SessionResponse])
async def get_user_sessions(user_id: str, components=Depends(get_components)) -> List[Dict]:
    """Récupère toutes les sessions d'un utilisateur."""
    try:
        async with components.db.session_factory() as session:
            result = await session.execute(
                select(ChatSession)
                .where(ChatSession.user_id == user_id)
                .order_by(ChatSession.updated_at.desc())
            )
            sessions = result.scalars().all()
            return [s.to_dict() for s in sessions]
            
    except Exception as e:
        logger.error(f"Erreur récupération sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/del/{session_id}")
async def delete_session(session_id: str, components=Depends(get_components)) -> Dict:
    """Supprime une session et son historique."""
    try:
        async with components.db.session_factory() as session:
            # Vérification de l'existence de la session
            result = await session.execute(
                select(ChatSession).where(ChatSession.session_id == session_id)
            )
            chat_session = result.scalar_one_or_none()
            
            if not chat_session:
                raise HTTPException(status_code=404, detail="Session non trouvée")
            
            # Suppression de la session
            await session.delete(chat_session)
            await session.commit()
            
            logger.info(f"Session supprimée: {session_id}")
            return {"message": "Session supprimée avec succès"}
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur suppression session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}")
async def get_session_info(session_id: str, components=Depends(get_components)) -> Dict:
    """Récupère les informations d'une session spécifique."""
    try:
        async with components.db.session_factory() as session:
            result = await session.execute(
                select(ChatSession).where(ChatSession.session_id == session_id)
            )
            chat_session = result.scalar_one_or_none()
            
            if not chat_session:
                raise HTTPException(status_code=404, detail="Session non trouvée")
                
            return chat_session.to_dict()
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur récupération session: {e}")
        raise HTTPException(status_code=500, detail=str(e))