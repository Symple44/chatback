# api/routes/session_routes.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List
from pydantic import BaseModel
from datetime import datetime
from ..dependencies import get_components
from core.utils.logger import get_logger

logger = get_logger("session_routes")
router = APIRouter(prefix="/sessions", tags=["sessions"])

class SessionCreate(BaseModel):
    user_id: str

class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    messages_count: int

@router.post("/new", response_model=SessionResponse)
async def create_session(data: SessionCreate, components=Depends(get_components)) -> Dict:
    async with components.db.session_factory() as session:
        try:
            # Vérification de l'existence des sessions actives
            active_sessions = await session.execute(
                select(ChatSession)
                .where(
                    and_(
                        ChatSession.user_id == data.user_id,
                        ChatSession.is_active == True,
                        ChatSession.updated_at >= datetime.utcnow() - timedelta(hours=24)
                    )
                )
            )
            
            if active_sessions.scalar_one_or_none():
                # Réutilisation de la session existante
                return active_sessions.scalar_one_or_none()

            # Création d'une nouvelle session
            new_session = ChatSession(
                user_id=data.user_id,
                session_id=str(uuid.uuid4()),
                session_context={
                    "created_at": datetime.utcnow().isoformat(),
                    "source": data.metadata.get("source", "unknown"),
                    "history": []
                }
            )
            session.add(new_session)
            await session.commit()
            await session.refresh(new_session)
            
            return new_session

@router.get("/user/{user_id}", response_model=List[SessionResponse])
async def get_user_sessions(
    user_id: str,
    components=Depends(get_components)
) -> List[Dict]:
    """
    Récupère toutes les sessions d'un utilisateur.
    """
    try:
        sessions = await components.session_manager.get_user_sessions(user_id)
        return sessions
    except Exception as e:
        logger.error(f"Erreur récupération sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/del/{session_id}")
async def delete_session(
    session_id: str,
    components=Depends(get_components)
) -> Dict:
    """
    Supprime une session et son historique.
    """
    try:
        # Vérifier si la session existe
        session = await components.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session non trouvée")
            
        # Supprimer l'historique de la session
        await components.db.delete_session_history(session_id)
        
        # Supprimer la session
        await components.session_manager.delete_session(session_id)
        
        logger.info(f"Session supprimée: {session_id}")
        return {"message": "Session supprimée avec succès"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
