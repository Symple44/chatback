from typing import Dict, List, Optional, Union, Generator, Any
from datetime import datetime
import torch
import asyncio
from core.config.config import settings
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
        self.auth_manager = None
        self.cuda_manager = None
        self.tokenizer_manager = None
        self.model_manager = None
        self.embedding_manager = None
        self._initialized = False
        self.summarizer = None
        
    async def initialize(self, components=None):  # Ajout du paramètre components
        """Initialise tous les composants de manière asynchrone."""
        if not self._initialized:
            try:
                logger.info("Démarrage de l'initialisation du système d'inférence")

                if components:  # Utilisation des composants passés
                    # Réutilisation des composants déjà initialisés
                    self.cuda_manager = components.cuda_manager
                    self.tokenizer_manager = components.tokenizer_manager
                    self.model_manager = components.model_manager
                    self.auth_manager = components.auth_manager
                    self.embedding_manager = components.embedding_manager
                    self.summarizer = components.summarizer
                else:
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

            # Traitement des tokens avec vérification des dimensions
            inputs = self.tokenizer_manager.encode_with_truncation(
                messages,
                max_length=min(settings.MAX_INPUT_LENGTH, chat_model.model.config.max_position_embeddings),
                return_tensors="pt"
            )
            
            # Vérification des dimensions du tenseur d'entrée
            if inputs["input_ids"].shape[1] > chat_model.model.config.max_position_embeddings:
                logger.warning(f"Input sequence trop longue ({inputs['input_ids'].shape[1]}), troncature à {chat_model.model.config.max_position_embeddings}")
                inputs = {k: v[:, :chat_model.model.config.max_position_embeddings] for k, v in inputs.items()}
            
            # Transfert vers le device approprié
            inputs = {k: v.to(chat_model.device) for k, v in inputs.items()}
            
            # Configuration de génération avec limites adaptatives
            generation_config = self._get_generation_config(response_type)
            
            # Assurer que la longueur max est dans les limites du modèle
            max_total_tokens = chat_model.model.config.max_position_embeddings
            current_length = inputs["input_ids"].shape[1]
            max_new_tokens = min(
                generation_config.get("max_length", settings.MAX_NEW_TOKENS),
                max_total_tokens - current_length
            )
            generation_config["max_new_tokens"] = max_new_tokens

            # Génération avec autocast et synchronisation CUDA
            with torch.cuda.amp.autocast(enabled=True):
                outputs = chat_model.model.generate(
                    **inputs,
                    **generation_config,
                    pad_token_id=chat_model.tokenizer.pad_token_id,
                    eos_token_id=chat_model.tokenizer.eos_token_id,
                )
                torch.cuda.synchronize()  # Assurer que la génération est terminée

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

    def _get_generation_config(self, response_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Retourne la configuration de génération du modèle avec ajustements selon response_type.
        Args:
            response_type: Type de réponse demandée
        Returns:
            Configuration de génération ajustée
        """
        try:
            # Récupération de la config complète du modèle
            model = self.model_manager.current_models[ModelType.CHAT]
            generation_config = model.config.get("generation_config", {}).copy()

            # Surcharge des paramètres spécifiques si response_type spécifié
            if response_type and response_type in settings.RESPONSE_TYPES:
                response_config = settings.RESPONSE_TYPES[response_type]
                if "temperature" in response_config:
                    generation_config["temperature"] = float(response_config["temperature"])
                    generation_config["do_sample"] = True  # Activer do_sample si température est définie
                if "max_tokens" in response_config:
                    generation_config["max_length"] = int(response_config["max_tokens"])

            return generation_config

        except Exception as e:
            logger.error(f"Erreur récupération config génération: {e}")
            return {}
        
    async def create_embedding(self, text: str) -> List[float]:
        """Crée un embedding pour un texte donné."""
        if not self._initialized:
            raise RuntimeError("Système non initialisé")
            
        try:
            embeddings = await self.embedding_manager.get_embeddings(text, use_cache=True)
            # Si on reçoit une liste de listes (2D), prendre le premier vecteur
            if embeddings and isinstance(embeddings[0], list):
                return embeddings[0]
            return embeddings
            
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