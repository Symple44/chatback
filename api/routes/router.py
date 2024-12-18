# api/routes/router.py
from fastapi import APIRouter
from .chat_routes import router as chat_router
from .session_routes import router as session_router
from .history_routes import router as history_router
from .health_routes import router as health_router
from .user_routes import router as user_router

router = APIRouter(prefix="")

# Inclusion des différents routers
router.include_router(chat_router)
router.include_router(session_router)
router.include_router(history_router)
router.include_router(health_router)
router.include_router(user_router)
