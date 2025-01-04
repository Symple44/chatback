# core/chat/processor_factory.py
from typing import Dict, Optional
from core.utils.logger import get_logger
from .generic_processor import GenericProcessor

logger = get_logger("processor_factory")

class ProcessorFactory:
    """Factory pour la création des processeurs."""
    
    _processors: Dict[str, object] = {}
    
    @classmethod
    async def get_processor(cls, business_type: Optional[str], components) -> 'GenericProcessor':
        """Récupère ou crée un processeur approprié."""
        try:
            # Si business_type est None ou une chaîne vide, utiliser le processeur générique
            if not business_type:
                logger.info("Aucun type métier spécifié, utilisation du processeur générique")
                if "generic" not in cls._processors:
                    cls._processors["generic"] = GenericProcessor(components)
                return cls._processors["generic"]

            # Pour un type métier spécifique
            processor_key = business_type.lower()
            if processor_key not in cls._processors:
                try:
                    # Import dynamique du processeur métier
                    module_path = f"core.business.{processor_key}.processeur"
                    class_name = f"Processeur{processor_key.capitalize()}"
                    
                    logger.info(f"Tentative de chargement du processeur métier: {module_path}.{class_name}")
                    module = __import__(module_path, fromlist=[class_name])
                    processor_class = getattr(module, class_name)
                    cls._processors[processor_key] = processor_class(components)
                    
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Module métier {processor_key} non trouvé, utilisation du processeur générique: {str(e)}")
                    if "generic" not in cls._processors:
                        cls._processors["generic"] = GenericProcessor(components)
                    return cls._processors["generic"]
                    
            return cls._processors[processor_key]
            
        except Exception as e:
            logger.error(f"Erreur création processeur: {e}")
            # Fallback vers processeur générique
            if "generic" not in cls._processors:
                cls._processors["generic"] = GenericProcessor(components)
            return cls._processors["generic"]
