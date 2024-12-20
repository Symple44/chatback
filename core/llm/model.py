# core/llm/model.py
# core/llm/model.py
from typing import List, Dict, Optional, Any, AsyncIterator, Union
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TextIteratorStreamer,
    GenerationConfig
)
import asyncio
from datetime import datetime
import gc
import psutil
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .embeddings import EmbeddingsManager

logger = get_logger("model")

class ModelInference:
    def __init__(self):
        """Initialise le modèle d'inférence."""
        try:
            self.device = "cpu" if settings.USE_CPU_ONLY else self._get_optimal_device()
            logger.info(f"Utilisation du device: {self.device}")
            
            self.executor = ThreadPoolExecutor(max_workers=2)
            self.embeddings = EmbeddingsManager()
            self._setup_model()
            self._setup_tokenizer()
            
            # Configuration de génération par défaut
            self.generation_config = GenerationConfig(
                max_new_tokens=settings.MAX_NEW_TOKENS,
                min_new_tokens=settings.MIN_NEW_TOKENS,
                do_sample=settings.DO_SAMPLE,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                top_k=settings.TOP_K,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            logger.info("Modèle initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation modèle: {e}")
            raise

    def _get_optimal_device(self) -> str:
        """Détermine le meilleur device disponible."""
        if torch.cuda.is_available():
            # Vérifier la mémoire GPU disponible
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory >= 4 * (1024**3):  # 4GB minimum
                return "cuda"
        return "cpu"

    def _setup_model(self):
        """Configure le modèle avec optimisations."""
        try:
            logger.info(f"Chargement du modèle {settings.MODEL_NAME}")
            
            # Configuration du modèle
            model_kwargs = {
                "torch_dtype": torch.float16 if settings.USE_4BIT else torch.float32,
                "low_cpu_mem_usage": True,
                "device_map": "auto" if self.device == "cuda" else None
            }
            
            # Chargement avec retry
            for attempt in range(3):
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        settings.MODEL_NAME,
                        **model_kwargs
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"Tentative {attempt + 1} échouée: {e}")
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
            # Optimisations
            if settings.USE_4BIT:
                self.model = self.model.half()
            
            self.model.eval()  # Mode évaluation
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            logger.info("Modèle chargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise

    def _setup_tokenizer(self):
        """Configure le tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                use_fast=True,
                model_max_length=settings.MAX_INPUT_LENGTH
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Tokenizer configuré")
            
        except Exception as e:
            logger.error(f"Erreur configuration tokenizer: {e}")
            raise

    async def generate_response(
        self,
        query: str,
        context_docs: List[Dict],
        conversation_history: Optional[List] = None,
        language: str = "fr",
        generation_config: Optional[Dict] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Génère une réponse complète.
        """
        try:
            # Préparation du prompt
            prompt = self._prepare_prompt(
                query=query,
                context_docs=context_docs,
                conversation_history=conversation_history,
                language=language
            )
            
            # Tokenisation
            inputs = self._tokenize(prompt)
            
            # Configuration de génération
            gen_config = self.generation_config
            if generation_config:
                gen_config = GenerationConfig(**{
                    **gen_config.to_dict(),
                    **generation_config
                })
            
            # Génération
            with metrics.timer("model_inference"):
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=gen_config,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
            
            # Décodage et post-traitement
            response = self.tokenizer.decode(
                outputs.sequences[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            response = self._post_process_response(response)
            
            # Métriques
            metrics.increment_counter("model_generations")
            
            return {
                "response": response,
                "tokens_used": len(outputs.sequences[0]),
                "generation_time": metrics.get_timer_value("model_inference")
            }
            
        except Exception as e:
            logger.error(f"Erreur génération réponse: {e}")
            metrics.increment_counter("model_generation_errors")
            raise

    async def generate_streaming_response(
        self,
        query: str,
        context_docs: List[Dict],
        language: str = "fr"
    ) -> AsyncIterator[str]:
        """
        Génère une réponse en streaming.
        """
        try:
            # Préparation du prompt
            prompt = self._prepare_prompt(
                query=query,
                context_docs=context_docs,
                language=language
            )
            
            # Configuration du streaming
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Génération dans un thread séparé
            inputs = self._tokenize(prompt)
            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "generation_config": self.generation_config
            }
            
            thread = Thread(target=self._generate_in_thread, kwargs=generation_kwargs)
            thread.start()
            
            # Streaming des tokens
            buffer = ""
            async for token in self._async_tokens(streamer):
                buffer += token
                if len(buffer) >= settings.STREAM_CHUNK_SIZE:
                    yield buffer
                    buffer = ""
                await asyncio.sleep(settings.STREAM_DELAY)
            
            if buffer:
                yield buffer
                
        except Exception as e:
            logger.error(f"Erreur génération streaming: {e}")
            yield settings.ERROR_MESSAGE

    def _generate_in_thread(self, **kwargs):
        """Génère la réponse dans un thread séparé."""
        try:
            with torch.inference_mode():
                _ = self.model.generate(**kwargs)
        except Exception as e:
            logger.error(f"Erreur génération thread: {e}")

    async def _async_tokens(self, streamer) -> AsyncIterator[str]:
        """Convertit le streamer en générateur asynchrone."""
        loop = asyncio.get_event_loop()
        for token in streamer:
            yield await loop.run_in_executor(None, lambda: token)

    def _prepare_prompt(
        self,
        query: str,
        context_docs: List[Dict],
        conversation_history: Optional[List] = None,
        language: str = "fr"
    ) -> str:
        """Prépare le prompt avec contexte."""
        # Formatage du contexte
        context_parts = []
        if context_docs:
            for doc in context_docs:
                context_parts.append(
                    f"Source: {doc.get('title', 'Unknown')}\n"
                    f"Contenu: {doc.get('content', '')}\n"
                )
        
        context = "\n\n".join(context_parts) if context_parts else "Aucun contexte disponible."
        
        # Formatage de l'historique
        history = ""
        if conversation_history:
            history_parts = []
            for msg in conversation_history[-5:]:  # Limité aux 5 derniers messages
                history_parts.append(
                    f"Human: {msg.get('user')}\n"
                    f"Assistant: {msg.get('assistant')}"
                )
            history = "\n".join(history_parts)
        
        # Construction du prompt final
        return settings.CHAT_TEMPLATE.format(
            system=settings.SYSTEM_PROMPT,
            query=query,
            context=context,
            history=history,
            language=language
        )

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenise le texte."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.MAX_INPUT_LENGTH,
            padding=True
        ).to(self.device)
        
        return inputs

    def _post_process_response(self, response: str) -> str:
        """Post-traite la réponse générée."""
        # Extraction de la réponse
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        # Nettoyage
        response = response.replace("<|endoftext|>", "").strip()
        
        return response

    async def create_embedding(self, text: str) -> List[float]:
        """Crée un embedding pour un texte."""
        return await self.embeddings.generate_embedding(text)

    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            self.executor.shutdown(wait=True)
            await self.embeddings.cleanup()
            
            self.model.cpu()
            del self.model
            del self.tokenizer
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Ressources modèle nettoyées")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage modèle: {e}")

    def __del__(self):
        """Destructeur de la classe."""
        try:
            asyncio.run(self.cleanup())
        except:
            pass
    return '\n'.join(triggers)
