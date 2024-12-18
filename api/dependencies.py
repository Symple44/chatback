# api/dependencies.py
from typing import Annotated
from fastapi import Depends
from core.utils.logger import get_logger

logger = get_logger("dependencies")

async def get_components():
    """
    Dépendance pour accéder aux composants de l'application.
    À importer depuis le main.py pour avoir accès à l'instance ComponentManager.
    """
    from main import components
    return components