# core/llm/model.py
from typing import Dict, List, Optional, Union, Generator
from datetime import datetime
import torch
import asyncio
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    GenerationConfig
)
from sentence_transformers import SentenceTransformer
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .cuda_manager import CUDAManager
from .memory_manager import MemoryManager
from .prompt_system import PromptSystem
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
        self.model = None
        self.embedding_model = None
        self._initialized = False
        self.prompt_system = PromptSystem()

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

                # Initialisation des modèles
                await self._initialize_model()
                await self._initialize_embedding_model()
                
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
            if settings.USE_FP16:
                with torch.amp.autocast('cuda'):
                    self.model = AutoModelForCausalLM.from_pretrained(**load_params)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(**load_params)

            # Post-initialisation
            model_device = next(self.model.parameters()).device
            logger.info(f"Modèle chargé sur {model_device}")
            logger.info(f"Type de données: {next(self.model.parameters()).dtype}")
            metrics.increment_counter("model_loads")

        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise

    async def _initialize_embedding_model(self):
        """Initialise le modèle d'embeddings."""
        try:
            logger.info(f"Chargement du modèle d'embeddings {settings.EMBEDDING_MODEL}")
            
            # Configuration du device
            device = "cuda" if torch.cuda.is_available() and not settings.USE_CPU_ONLY else "cpu"
            
            # Chargement du modèle d'embeddings
            self.embedding_model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=device,
                cache_folder=str(settings.CACHE_DIR)
            )
            
            logger.info(f"Modèle d'embeddings chargé sur {device}")
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle d'embeddings: {e}")
            raise

    async def create_embedding(self, text: str) -> List[float]:
        """Crée un embedding pour un texte donné."""
        if not self._initialized:
            raise RuntimeError("Le modèle n'est pas initialisé")
            
        try:
            # Normalisation du texte
            if not isinstance(text, str):
                text = str(text)
            
            # Génération de l'embedding
            with torch.no_grad():
                embedding = self.embedding_model.encode(
                    text,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )
                
            # Conversion en liste
            return embedding.cpu().tolist()
            
        except Exception as e:
            logger.error(f"Erreur création embedding: {e}")
            metrics.increment_counter("embedding_errors")
            raise

    async def generate_response(
        self,
        query: str,
        context_docs: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
        language: str = "fr",
    ) -> Dict:
        """Génération de réponse optimisée."""
        try:
            start_time = datetime.utcnow()
            logger.info("Début de la génération de réponse")

            # Construction du prompt
            prompt = self.prompt_system.build_chat_prompt(
                messages=[
                    *([msg for msg in conversation_history[-5:]] if conversation_history else []),
                    {"role": "user", "content": query}
                ],
                context_docs=context_docs,
                lang=language
            )
            logger.info(f"Longueur du prompt: {len(prompt)} caractères")

            # Vérification mémoire GPU
            if not await self.memory_manager.check_memory_usage():
                await self.memory_manager.cleanup()

            # Tokenisation
            inputs = self.tokenizer_manager.encode(
                prompt,
                max_length=settings.MAX_INPUT_LENGTH,
                return_tensors="pt"
            ).to(self.device)
            logger.info(f"Nombre de tokens en entrée: {len(inputs.input_ids[0])}")

            # Configuration génération
            generation_config = GenerationConfig(
                do_sample=settings.DO_SAMPLE,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                top_k=settings.TOP_K,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                min_new_tokens=settings.MIN_NEW_TOKENS,
                repetition_penalty=settings.REPETITION_PENALTY,
                pad_token_id=self.tokenizer_manager.tokenizer.pad_token_id
            )

            # Génération avec FP16 si activé
            with torch.cuda.amp.autocast(enabled=settings.USE_FP16):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config,
                        use_cache=True
                    )

            # Post-traitement et réponse
            response = self.tokenizer_manager.decode_and_clean(outputs[0])
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            return {
                "response": response,
                "tokens_used": len(outputs[0]) - len(inputs.input_ids[0]),
                "processing_time": processing_time,
                "confidence": self._calculate_confidence(context_docs)
            }

        except Exception as e:
            logger.error(f"Erreur génération: {e}", exc_info=True)
            metrics.increment_counter("generation_errors")
            raise

    def _calculate_confidence(self, context_docs: Optional[List[Dict]]) -> float:
        """Calcule le score de confiance."""
        if not context_docs:
            return 0.5
        scores = [doc.get("score", 0.0) for doc in context_docs]
        return min(sum(scores) / len(scores) if scores else 0.5, 1.0)

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
                    logger.warning(f"Erreur nettoyage modèle génératif: {e}")
                    
            if self.embedding_model is not None:
                try:
                    self.embedding_model.cpu()
                    del self.embedding_model
                    self.embedding_model = None
                except Exception as e:
                    logger.warning(f"Erreur nettoyage modèle d'embeddings: {e}")

            self._initialized = False

        except Exception as e:
            logger.error(f"Erreur nettoyage ressources: {e}")
