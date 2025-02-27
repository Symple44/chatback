# core/config/models/__init__.py
import os
from .base import ModelType
from .chat import CHAT_MODELS, MODEL_PERFORMANCE_CONFIGS
from .embedding import EMBEDDING_MODELS
from .summarizer import SUMMARIZER_MODELS
from .systems import SYSTEM_CONFIG

class ModelsConfig:
    """Configuration unifiée des modèles."""
    
    def __init__(self):
        # Noms des modèles par défaut
        self.MODEL_NAME = os.getenv("MODEL_NAME", "Mistral-Small-24B-Instruct-2501")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.MODEL_NAME_SUMMARIZER = os.getenv("MODEL_NAME_SUMMARIZER", "plguillou/t5-base-fr-sum-cnndm")
        self.MODEL_REVISION = os.getenv("MODEL_REVISION", "main")
        
        # Dictionnaires des configurations
        self.chat_models = CHAT_MODELS
        self.model_performance_configs = MODEL_PERFORMANCE_CONFIGS
        self.embedding_models = EMBEDDING_MODELS
        self.summarizer_models = SUMMARIZER_MODELS
        self.system_config = SYSTEM_CONFIG
        
        # Configurations des limites
        self.MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "8192"))
        self.MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "8192"))
        
    def get_model_config(self, model_name: str, model_type: ModelType):
        """Retourne la configuration d'un modèle."""
        if model_type == ModelType.CHAT:
            return self.chat_models.get(model_name)
        elif model_type == ModelType.EMBEDDING:
            return self.embedding_models.get(model_name)
        elif model_type == ModelType.SUMMARIZER:
            return self.summarizer_models.get(model_name)
        return None

__all__ = ['ModelsConfig', 'ModelType']