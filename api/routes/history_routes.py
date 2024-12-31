# api/routes/history_routes.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict
from ..dependencies import get_components
from core.utils.logger import get_logger

logger = get_logger("history_routes")
router = APIRouter(prefix="/history", tags=["history"])

@router.get("/session/{session_id}")
async def get_session_history(
    session_id: str,
    components=Depends(get_components)
) -> List[Dict]:
    try:
        # Utiliser le SessionManager au lieu d'un appel direct à la DB
        history = await components.session_manager.get_session_history(session_id)
        return [h._asdict() for h in history] if history else []
    except Exception as e:
        logger.error(f"Erreur récupération historique: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}")
async def get_user_history(
    user_id: str,
    limit: int = 50,
    components=Depends(get_components)
) -> List[Dict]:
    try:
        return await components.session_manager.get_chat_history(user_id=user_id, limit=limit)
    except Exception as e:
        logger.error(f"Erreur récupération historique utilisateur: {e}")
        raise HTTPException(status_code=500, detail=str(e))