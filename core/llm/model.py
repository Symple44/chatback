from typing import Dict, List, Optional, Union
from datetime import datetime
from contextlib import nullcontext
import torch
import asyncio
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
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
        """Initialise les composants de base du modèle."""
        self.auth_manager = HuggingFaceAuthManager()
        self.cuda_manager = CUDAManager()
        self.memory_manager = MemoryManager()
        self.tokenizer_manager = TokenizerManager()
        self.prompt_system = PromptSystem()
        self.model = None
        self.embedding_model = None
        self._initialized = False

    async def initialize(self):
        """Initialise le modèle de manière asynchrone."""
        if not self._initialized:
            try:
                if not await self.auth_manager.setup_auth():
                    raise ValueError("Échec de l'authentification Hugging Face")

                await self._initialize_model()
                await self._initialize_embedding_model()
                
                self._initialized = True
                logger.info(f"Modèle {settings.MODEL_NAME} initialisé")

            except Exception as e:
                logger.error(f"Erreur initialisation modèle: {e}")
                raise

    async def _initialize_model(self):
        """Initialise le modèle de génération."""
        try:
            max_memory = self.memory_manager.get_optimal_memory_config()
            
            load_params = {
                "pretrained_model_name_or_path": settings.MODEL_NAME,
                "revision": settings.MODEL_REVISION,
                "device_map": "auto",
                "max_memory": max_memory,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if settings.USE_FP16 else None,
                "load_in_8bit": settings.USE_8BIT,
                "load_in_4bit": settings.USE_4BIT
            }

            context = torch.amp.autocast('cuda') if settings.USE_FP16 else nullcontext()
            with context:
                self.model = AutoModelForCausalLM.from_pretrained(**load_params)

            logger.info(f"Modèle chargé sur {next(self.model.parameters()).device}")
            metrics.increment_counter("model_loads")

        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise

    async def _initialize_embedding_model(self):
        """Initialise le modèle d'embeddings."""
        try:
            device = "cuda" if torch.cuda.is_available() and not settings.USE_CPU_ONLY else "cpu"
            
            self.embedding_model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=device,
                cache_folder=str(settings.CACHE_DIR)
            )
            
            logger.info(f"Modèle d'embeddings chargé sur {device}")
            
        except Exception as e:
            logger.error(f"Erreur chargement embeddings: {e}")
            raise

    async def create_embedding(self, text: str) -> List[float]:
        """Crée un embedding pour un texte."""
        if not self._initialized:
            raise RuntimeError("Modèle non initialisé")
            
        try:
            with torch.no_grad():
                embedding = self.embedding_model.encode(
                    text,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )
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
        language: str = "fr"
    ) -> Dict:
        """Génère une réponse basée sur le contexte."""
        try:
            # Préparation du contexte
            context_parts = []
            relevant_sections = []
            
            if context_docs:
                for doc in context_docs:
                    if "processed_sections" in doc:
                        sections = sorted(
                            doc["processed_sections"],
                            key=lambda x: float(x.get("importance_score", 0)),
                            reverse=True
                        )
                        relevant_sections.extend(sections[:3])
                        content = "\n\n".join(s["content"] for s in sections[:3])
                    else:
                        content = doc.get("content", "")
                    
                    if content:
                        title = doc.get("title", "Document")
                        context_parts.append(f"[Source: {title}]\n{content}")

            formatted_context = "\n\n---\n\n".join(context_parts)

            # Construction du prompt
            prompt = self.prompt_system.build_chat_prompt(
                messages=conversation_history[-5:] if conversation_history else [],
                context=formatted_context,
                query=query,
                lang=language
            )

            # Configuration de génération
            generation_config = GenerationConfig(**settings.generation_config)

            # Génération avec gestion mémoire
            with metrics.timer("model_inference"):
                if settings.USE_FP16:
                    context_manager = torch.amp.autocast('cuda')
                else:
                    context_manager = nullcontext()

                with context_manager:
                    with torch.no_grad():
                        inputs = self.tokenizer_manager(
                            prompt, 
                            max_length=settings.MAX_INPUT_LENGTH,
                            return_tensors="pt"
                        ).to(self.model.device)
                        
                        outputs = self.model.generate(
                            **inputs,
                            generation_config=generation_config,
                            use_cache=True
                        )

                        response_text = self.tokenizer_manager.decode_and_clean(outputs[0])

            # Calcul des métriques
            input_tokens = len(inputs.input_ids[0])
            output_tokens = len(outputs[0]) - input_tokens
            confidence_score = sum(float(s.get("importance_score", 0)) for s in relevant_sections) / max(len(relevant_sections), 1)

            return {
                "response": response_text,
                "confidence_score": min(confidence_score, 1.0),
                "tokens_used": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                },
                "metadata": {
                    "model_version": settings.MODEL_NAME,
                    "timestamp": datetime.utcnow().isoformat(),
                    "language": language
                }
            }

        except Exception as e:
            logger.error(f"Erreur génération: {e}", exc_info=True)
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
                    logger.warning(f"Erreur nettoyage modèle génératif: {e}")
                    
            if self.embedding_model is not None:
                try:
                    self.embedding_model.cpu()
                    del self.embedding_model
                    self.embedding_model = None
                except Exception as e:
                    logger.warning(f"Erreur nettoyage embeddings: {e}")

            self._initialized = False
            logger.info("Ressources nettoyées")

        except Exception as e:
            logger.error(f"Erreur nettoyage ressources: {e}")