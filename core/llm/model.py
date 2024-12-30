# core/llm/model.py
from typing import Dict, List, Optional, Union, Generator
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    GenerationConfig
)
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .cuda_manager import CUDAManager
from .memory_manager import MemoryManager
from .prompt_builder import PromptBuilder
from .tokenizer_manager import TokenizerManager
from .auth_manager import HuggingFaceAuthManager

logger = get_logger("model")

class ModelInference:
    def __init__(self):
        """Initialise les composants de base du modèle d'inférence."""
        self.auth_manager = HuggingFaceAuthManager()
        self.cuda_manager = CUDAManager()
        self.memory_manager = MemoryManager()
        self.tokenizer_manager = TokenizerManager()
        self.prompt_builder = PromptBuilder()
        self.model = None
        self._initialized = False

    async def initialize(self):
        """Initialise le modèle de manière asynchrone."""
        if not self._initialized:
            try:
                # Authentification Hugging Face
                if not await self.auth_manager.setup_auth():
                    raise ValueError("Échec de l'authentification Hugging Face")

                # Vérification de l'accès au modèle
                if not self.auth_manager.get_model_access(settings.MODEL_NAME):
                    raise ValueError(f"Accès non autorisé au modèle {settings.MODEL_NAME}")

                # Initialisation du modèle
                await self._initialize_model()
                self._initialized = True
                logger.info(f"Modèle {settings.MODEL_NAME} initialisé avec succès")

            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du modèle: {e}")
                raise

    async def _initialize_model(self):
        """Configure et charge le modèle."""
        try:
            logger.info(f"Chargement du modèle {settings.MODEL_NAME}")
    
            # Récupération de la configuration mémoire depuis settings via memory_manager
            max_memory = self.memory_manager.get_optimal_memory_config()
            
            # Paramètres de chargement en utilisant les settings
            load_params = {
                "pretrained_model_name_or_path": settings.MODEL_NAME,
                "revision": settings.MODEL_REVISION,
                "device_map": "auto",
                "max_memory": max_memory,
                "trust_remote_code": True
            }
    
            # Configuration du type de données en fonction des settings
            if settings.USE_FP16:
                load_params["torch_dtype"] = torch.float16
            elif settings.USE_8BIT:
                load_params["load_in_8bit"] = True
            elif settings.USE_4BIT:
                load_params["load_in_4bit"] = True
                if hasattr(settings, "BNB_4BIT_COMPUTE_DTYPE"):
                    load_params["bnb_4bit_compute_dtype"] = getattr(torch, settings.BNB_4BIT_COMPUTE_DTYPE)
    
            logger.info(f"Configuration de chargement: {load_params}")
    
            # Chargement du modèle avec autocast si FP16 est activé
            with torch.cuda.amp.autocast(enabled=settings.USE_FP16):
                self.model = AutoModelForCausalLM.from_pretrained(**load_params)
    
            # Post-initialisation
            model_device = next(self.model.parameters()).device
            logger.info(f"Modèle chargé sur {model_device}")
            logger.info(f"Type de données: {next(self.model.parameters()).dtype}")
            metrics.increment_counter("model_loads")
    
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise

    async def generate_response(
        self,
        query: str,
        context_docs: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
        language: str = "fr",
        **kwargs
    ) -> Dict:
        """Génère une réponse basée sur une requête et son contexte."""
        if not self._initialized:
            raise RuntimeError("Le modèle n'est pas initialisé. Appelez initialize() d'abord.")

        try:
            # Construction du prompt
            prompt = self.prompt_builder.build_prompt(
                query=query,
                context_docs=context_docs,
                conversation_history=conversation_history,
                language=language
            )

            # Tokenisation
            inputs = self.tokenizer_manager.encode(
                prompt,
                max_length=settings.MAX_INPUT_LENGTH
            ).to(self.model.device)

            # Génération
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.tokenizer_manager.generation_config,
                    max_new_tokens=settings.MAX_OUTPUT_LENGTH,
                    **kwargs
                )

            # Décodage et post-traitement
            response = self.tokenizer_manager.decode_and_clean(outputs[0])

            return {
                "response": response,
                "prompt_tokens": len(inputs.input_ids[0]),
                "completion_tokens": len(outputs[0]) - len(inputs.input_ids[0]),
                "total_tokens": len(outputs[0])
            }

        except Exception as e:
            logger.error(f"Erreur génération réponse: {e}")
            metrics.increment_counter("generation_errors")
            raise

    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            await self.memory_manager.cleanup()
            await self.cuda_manager.cleanup()
            await self.tokenizer_manager.cleanup()
            
            if self.model is not None:
                try:
                    self.model.cpu()
                    del self.model
                    self.model = None
                except Exception as e:
                    logger.warning(f"Erreur nettoyage modèle: {e}")

            self._initialized = False

        except Exception as e:
            logger.error(f"Erreur nettoyage ressources: {e}")
