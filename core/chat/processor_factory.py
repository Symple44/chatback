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
            # Si pas de type métier ou type générique, utiliser le processeur générique
            if not business_type or business_type == "generic":
                if "generic" not in cls._processors:
                    cls._processors["generic"] = GenericProcessor(components)
                return cls._processors["generic"]

            # Pour un type métier spécifique
            if business_type not in cls._processors:
                try:
                    # Import dynamique du processeur métier
                    module_name = f"core.business.{business_type.lower()}.processeur"
                    class_name = f"Processeur{business_type.capitalize()}"
                    
                    module = __import__(module_name, fromlist=[class_name])
                    processor_class = getattr(module, class_name)
                    cls._processors[business_type] = processor_class(components)
                    
                except ImportError:
                    logger.warning(f"Module métier {business_type} non trouvé, utilisation du processeur générique")
                    if "generic" not in cls._processors:
                        cls._processors["generic"] = GenericProcessor(components)
                    return cls._processors["generic"]
                    
            return cls._processors[business_type]
            
        except Exception as e:
            logger.error(f"Erreur création processeur: {e}")
            # Fallback vers processeur générique
            if "generic" not in cls._processors:
                cls._processors["generic"] = GenericProcessor(components)
            return cls._processors["generic"]
