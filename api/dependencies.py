# api/dependencies.py
from fastapi import Depends, HTTPException, status
from typing import Dict, Any, Optional

from core.utils.logger import get_logger
from core.utils.metrics import metrics

# Import de la nouvelle structure de cache
from core.cache import redis_cache, cache_manager, CacheManager
from core.document_processing.table_extractor import TableExtractor
from core.config import settings

logger = get_logger("api_dependencies")

# Conteneur global pour les composants
class ComponentContainer:
    def __init__(self):
        """Initialise un conteneur de composants."""
        self._components = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialise les composants essentiels."""
        if self._initialized:
            return
        
        # Cache Redis principal
        if "redis_cache" not in self._components:
            await redis_cache.initialize()
            self._components["redis_cache"] = redis_cache
            logger.info("Cache Redis initialisé")
        
        # Cache Manager principal
        if "cache_manager" not in self._components:
            await cache_manager.initialize()
            self._components["cache_manager"] = cache_manager
            logger.info("Cache Manager initialisé")
        
        # Cache spécifique pour PDF
        if "pdf_cache" not in self._components:
            pdf_cache = CacheManager(redis_cache=redis_cache, namespace="pdf_extraction")
            await pdf_cache.initialize() 
            self._components["pdf_cache"] = pdf_cache
            logger.info("Cache PDF initialisé")
        
        # Autres composants seront créés à la demande
        
        self._initialized = True
        logger.info("Conteneur de composants initialisé")
    
    def __getattr__(self, name):
        """Accès aux composants via la notation pointée."""
        if not self._initialized:
            raise RuntimeError("Conteneur non initialisé")
            
        if name in self._components:
            return self._components[name]
            
        # Créer certains composants à la demande avec lazy loading
        if name == "table_extractor":
            logger.info("Initialisation lazy de l'extracteur de tableaux")
            # Utiliser le cache PDF pour l'extracteur de tableaux
            pdf_cache = self._components.get("pdf_cache")
            self._components["table_extractor"] = TableExtractor(cache_enabled=True)
            return self._components["table_extractor"]
            
        raise AttributeError(f"Composant {name} non trouvé")
    
    def __contains__(self, name):
        """Vérifie si un composant existe."""
        return name in self._components
    
    def get(self, name, default=None):
        """Obtient un composant avec une valeur par défaut."""
        try:
            return getattr(self, name)
        except (AttributeError, RuntimeError):
            return default

# Singleton global de composants
components_container = ComponentContainer()

async def get_components():
    """Dépendance pour obtenir les composants initialisés."""
    await components_container.initialize()
    return components_container

async def get_cache(namespace: str = "default"):
    """Dépendance pour obtenir un cache spécifique."""
    await components_container.initialize()
    
    cache_key = f"cache_{namespace}"
    
    # Réutiliser un cache existant si disponible
    if cache_key in components_container._components:
        return components_container._components[cache_key]
    
    # Créer un nouveau cache pour ce namespace
    specific_cache = CacheManager(redis_cache=redis_cache, namespace=namespace)
    await specific_cache.initialize()
    components_container._components[cache_key] = specific_cache
    
    return specific_cache

async def get_pdf_cache():
    """Dépendance pour obtenir le cache PDF."""
    await components_container.initialize()
    return components_container.pdf_cache

async def get_table_extractor(
    components=Depends(get_components),
    table_detector=Depends(get_table_detector)
):
    """Dépendance pour obtenir l'extracteur de tableaux avec le détecteur de tableaux IA."""
    await components.initialize()
    
    if "table_extractor" not in components._components:
        logger.info("Initialisation lazy de l'extracteur de tableaux")
        # Utiliser le cache PDF et le détecteur de tableaux
        pdf_cache = components.get("pdf_cache")
        extractor = TableExtractor(
            cache_enabled=True,
            table_detector=table_detector  # Passer le détecteur ici
        )
        components._components["table_extractor"] = extractor
    
    return components.table_extractor

async def get_table_detector(components=Depends(get_components)):
    """Dépendance pour obtenir le détecteur de tableaux IA."""
    if not hasattr(components, "table_detector") and settings.document.ENABLE_AI_TABLE_DETECTION:
        from core.document_processing.table_detection import TableDetectionModel
        table_detector = TableDetectionModel(cuda_manager=components.cuda_manager)
        await table_detector.initialize()
        components._components["table_detector"] = table_detector
        logger.info("Détecteur de tableaux par IA initialisé à la demande")
    
    return components.get("table_detector")