# core/chat/processor_factory.py
from typing import Dict, Optional
from core.utils.logger import get_logger

logger = get_logger("processor_factory")

class ProcessorFactory:
    """Factory pour la création des processeurs."""
    
    _processors: Dict[str, object] = {}
    
    @classmethod
    async def get_processor(cls, business_type: Optional[str], components) -> 'BaseProcessor':
        """Récupère ou crée un processeur approprié."""
        
        # Si pas de type métier spécifié, utiliser le processeur générique
        processor_type = business_type or "generic"
        
        # Réutiliser le processeur existant s'il existe
        if processor_type in cls._processors:
            return cls._processors[processor_type]
            
        try:
            if processor_type == "generic":
                from .generic_processor import GenericProcessor
                processor = GenericProcessor(components)
            else:
                # Import dynamique du processeur métier
                module_name = f"core.business.{processor_type}.processeur"
                class_name = f"Processeur{processor_type.capitalize()}"
                
                module = __import__(module_name, fromlist=[class_name])
                processor_class = getattr(module, class_name)
                processor = processor_class(components)
                
            cls._processors[processor_type] = processor
            return processor
            
        except ImportError:
            logger.warning(f"Module métier {processor_type} non trouvé, utilisation du processeur générique")
            from .generic_processor import GenericProcessor
            return GenericProcessor(components)
