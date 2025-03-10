# api/dependencies.py
from fastapi import Depends, HTTPException, status
from typing import Dict, Any, Optional

from core.utils.logger import get_logger

logger = get_logger("api_dependencies")

# Référence au ComponentManager global de main.py
# Importé ici pour éviter des imports circulaires
from main import components

async def get_components():
    """
    Dépendance pour obtenir les composants initialisés.
    
    Cette fonction garantit que le ComponentManager est initialisé avant utilisation.
    Les composants peuvent être accédés directement via la notation pointée:
        - components.redis_cache
        - components.cache_manager
        - components.table_extractor
        - components.table_detector (si activé)
        etc.
    
    Returns:
        ComponentManager: Instance initialisée du gestionnaire de composants
    """
    # S'assurer que tous les composants sont initialisés
    if not components.initialized:
        await components.initialize()
        
    return components