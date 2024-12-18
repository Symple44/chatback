# api/routes/health_routes.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
from ..dependencies import get_components
from core.utils.logger import get_logger

logger = get_logger("health_routes")
router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check(components=Depends(get_components)) -> Dict:
    try:
        return await components.check_health()
    except Exception as e:
        logger.error(f"Erreur health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))