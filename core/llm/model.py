from typing import Dict, List, Optional, Union, Generator
from datetime import datetime
import torch
import asyncio
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .cuda_manager import CUDAManager, ModelPriority
from .model_manager import ModelManager
from .model_loader import ModelType, LoadedModel
from .embedding_manager import EmbeddingManager
from .summarizer import DocumentSummarizer
from .tokenizer_manager import TokenizerManager
from .auth_manager import HuggingFaceAuthManager

logger = get_logger("model")

class ModelInference:
    """Interface principale pour l'inférence des modèles."""
    
    def __init__(self):
        """Initialise les composants nécessaires."""
        self.auth_manager = HuggingFaceAuthManager()
        self.cuda_manager = CUDAManager()
        self.tokenizer_manager = TokenizerManager()
        self.model_manager = None
        self.embedding_manager = EmbeddingManager()
        self._initialized = False
        self.summarizer = DocumentSummarizer()
        
    async def initialize(self):
        """Initialise tous les composants de manière asynchrone."""
        if not self._initialized:
            try:
                logger.info("Démarrage de l'initialisation du système d'inférence")

                # 1. Authentification Hugging Face
                if not await self.auth_manager.setup_auth():
                    raise ValueError("Échec de l'authentification Hugging Face")

                # 2. Vérification de l'accès aux modèles
                models_to_check = [
                    settings.MODEL_NAME,
                    settings.EMBEDDING_MODEL,
                    settings.MODEL_NAME_SUMMARIZER
                ]
                for model in models_to_check:
                    if not self.auth_manager.get_model_access(model):
                        raise ValueError(f"Accès non autorisé au modèle {model}")

                # 3. Initialisation CUDA
                await self.cuda_manager.initialize()
                logger.info("CUDA initialisé")

                # 4. Initialisation du TokenizerManager
                await self.tokenizer_manager.initialize()
                logger.info("Tokenizer initialisé")

                # 5. Initialisation du ModelManager
                self.model_manager = ModelManager(self.cuda_manager, self.tokenizer_manager)
                await self.model_manager.initialize()
                logger.info("ModelManager initialisé")

                # 6. Initialisation de l'EmbeddingManager
                await self.embedding_manager.initialize(self.model_manager)
                logger.info("EmbeddingManager initialisé")

                # 7. Initialisation du Summarizer
                await self.summarizer.initialize(self.model_manager)
                logger.info("Summarizer initialisé")

                self._initialized = True
                logger.info("Initialisation complète du système d'inférence")

            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation: {e}")
                await self.cleanup()
                raise

    async def generate_response(
        self,
        messages: List[Dict],
        language: str = "fr",
        response_type: str = "comprehensive",
        **kwargs
    ) -> Dict:
        """Génère une réponse basée sur les messages."""
        try:
            metrics.increment_counter("generation_requests")
            chat_model = self.model_manager.current_models[ModelType.CHAT]
            
            if not chat_model:
                raise RuntimeError("Modèle de chat non initialisé")

            # Traitement des tokens
            inputs = self.tokenizer_manager.encode_with_truncation(
                messages,
                max_length=settings.MAX_INPUT_LENGTH,
                return_tensors="pt"
            )
            
            # Transfert vers le device approprié
            inputs = {k: v.to(chat_model.device) for k, v in inputs.items()}
            
            # Génération avec config adaptée
            generation_config = self._get_generation_config(response_type)
            with torch.cuda.amp.autocast(enabled=True):
                outputs = chat_model.model.generate(
                    **inputs,
                    generation_config=generation_config
                )

            # Décodage et nettoyage
            response_text = self.tokenizer_manager.decode_and_clean(outputs[0])

            return {
                "response": response_text,
                "tokens_used": self._count_tokens(inputs, outputs),
                "processing_time": metrics.get_timer_value("generation"),
                "metadata": {
                    "model": chat_model.model_name,
                    "model_type": "chat",
                    "language": language,
                    "response_type": response_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Erreur génération: {e}")
            metrics.increment_counter("generation_errors")
            return {
                "response": "Une erreur est survenue lors de la génération.",
                "tokens_used": {"total": 0},
                "error": str(e)
            }

    async def create_embedding(self, text: str) -> List[float]:
        """Crée un embedding pour un texte donné."""
        if not self._initialized:
            raise RuntimeError("Système non initialisé")
            
        try:
            return await self.embedding_manager.get_embeddings(text, use_cache=True)
            
        except Exception as e:
            logger.error(f"Erreur création embedding: {e}")
            metrics.increment_counter("embedding_errors")
            raise

    async def change_model(
        self,
        model_name: str,
        model_type: ModelType = ModelType.CHAT
    ) -> bool:
        """Change le modèle actif du type spécifié."""
        try:
            if not self._initialized:
                raise RuntimeError("Système non initialisé")

            result = await self.model_manager.change_model(model_name, model_type)
            if result:
                logger.info(f"Modèle {model_type.value} changé pour {model_name}")
                return True
                
            return False

        except Exception as e:
            logger.error(f"Erreur changement modèle: {e}")
            return False

    def _count_tokens(self, inputs: Dict, outputs: torch.Tensor) -> Dict[str, int]:
        """Compte les tokens utilisés."""
        return {
            "input": inputs["input_ids"].shape[1],
            "output": outputs.shape[1],
            "total": inputs["input_ids"].shape[1] + outputs.shape[1]
        }

    async def cleanup(self):
        """Nettoie toutes les ressources."""
        try:
            components_to_cleanup = [
                self.model_manager,
                self.embedding_manager,
                self.summarizer,
                self.cuda_manager,
                self.tokenizer_manager
            ]
            
            for component in components_to_cleanup:
                if component:
                    await component.cleanup()
            
            self._initialized = False
            logger.info("Nettoyage complet effectué")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage ressources: {e}")