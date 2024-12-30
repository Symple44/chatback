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

logger = get_logger("model")

class ModelInference:
    def __init__(self):
        """Initialise le modèle d'inférence."""
        try:
            self.cuda_manager = CUDAManager()
            self.memory_manager = MemoryManager()
            self.tokenizer_manager = TokenizerManager()
            self.prompt_builder = PromptBuilder()
            
            self._initialize_model()
            
        except Exception as e:
            logger.error(f"Erreur initialisation modèle: {e}")
            raise

    def _initialize_model(self):
        """Configure et charge le modèle."""
        try:
            logger.info(f"Chargement du modèle {settings.MODEL_NAME}")

            # Récupération de la configuration mémoire optimisée
            max_memory = self.memory_manager.get_optimal_memory_config()
            
            # Paramètres de chargement
            load_params = self.cuda_manager.get_model_load_parameters(
                model_name=settings.MODEL_NAME,
                max_memory=max_memory
            )

            # Chargement du modèle
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
            
            if hasattr(self, 'model'):
                try:
                    self.model.cpu()
                    del self.model
                except Exception as e:
                    logger.warning(f"Erreur nettoyage modèle: {e}")

        except Exception as e:
            logger.error(f"Erreur nettoyage ressources: {e}")
