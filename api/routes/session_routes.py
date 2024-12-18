# api/routes/session_routes.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
from pydantic import BaseModel
from ..dependencies import get_components
from core.utils.logger import get_logger

logger = get_logger("session_routes")
router = APIRouter(prefix="/sessions", tags=["sessions"])

class SessionCreate(BaseModel):
    user_id: str

@router.post("/new")
async def create_session(
    data: SessionCreate,
    components=Depends(get_components)
) -> Dict:
    """
    Crée une nouvelle session pour l'utilisateur.
    """
    try:
        session = await components.session_manager.create_session(data.user_id)
        logger.info(f"Nouvelle session créée: {session}")
        return session
    except Exception as e:
        logger.error(f"Erreur création session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}")
async def get_session(
    session_id: str,
    components=Depends(get_components)
) -> Dict:
    """
    Récupère les informations d'une session.
    """
    try:
        session = await components.db_manager.get_session_info(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session non trouvée")
        return session
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur récupération session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
