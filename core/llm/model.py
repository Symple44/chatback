# core/llm/model.py
from typing import List, Dict, Optional, Any, AsyncIterator, Union
from transformers import BitsAndBytesConfig
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TextIteratorStreamer,
    GenerationConfig
)
import asyncio
from datetime import datetime
import time
import gc
import psutil
import os
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .embeddings import EmbeddingsManager

logger = get_logger("model")

class ModelInference:
    def __init__(self):
        """Initialise le modèle d'inférence avec des optimisations matérielles."""
        try:
            # Force CUDA path et env vars avant tout
            os.environ["CUDA_HOME"] = "/usr/local/cuda"
            os.environ["PATH"] = f"{os.environ['PATH']}:/usr/local/cuda/bin"
            
            # Nettoyage initial agressif
            self._cleanup_memory()
            
            # Configuration CUDA optimisée pour RTX 3090
            if not settings.USE_CPU_ONLY:
                self._setup_cuda_environment()

            self.device = self._get_optimal_device()
            logger.info(f"Utilisation du device: {self.device}")

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    logger.info(f"Mémoire GPU {i}: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

            # Configuration BitsAndBytes pour quantification 4-bit
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True if settings.USE_4BIT else False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            ) if settings.USE_4BIT else None

            self.executor = ThreadPoolExecutor(max_workers=2)
            self.embeddings = EmbeddingsManager()
            
            # Initialisation séquentielle
            self._setup_model()
            self._setup_tokenizer()
            self._setup_generation_config()
            
            logger.info("Modèle initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation modèle: {e}")
            self._cleanup_memory()
            raise

    def _cleanup_memory(self):
        """Nettoyage agressif de la mémoire."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _setup_cuda_environment(self):
        """Configure l'environnement CUDA pour la RTX 3090."""
        if torch.cuda.is_available():
            # Optimisations spécifiques RTX 3090
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            
            # Configuration mémoire GPU optimisée pour 24GB
            torch.cuda.set_per_process_memory_fraction(0.95)  # Utilise 95% de la VRAM
            
            # Paramètres CUDA supplémentaires
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"  # Ampere architecture

    def _get_optimal_device(self) -> str:
        """Détermine le meilleur device disponible avec vérification détaillée."""
        if settings.USE_CPU_ONLY:
            return "cpu"

        try:
            if torch.cuda.is_available():
                # Vérification détaillée de la GPU
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                logger.info(f"Mémoire GPU totale détectée: {gpu_memory / 1e9:.2f} GB")
                
                # Vérifie si CUDA est réellement utilisable
                test_tensor = torch.tensor([1.0], device="cuda")
                del test_tensor
                
                return "cuda"
        except Exception as e:
            logger.warning(f"Erreur détection CUDA: {e}")
        
        return "cpu"

    def _setup_model(self):
        """Configure le modèle avec gestion optimisée de la mémoire."""
        try:
            logger.info(f"Chargement du modèle {settings.MODEL_NAME}")
            
            # Configuration adaptative selon la mémoire disponible
            gpu_memory = None
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
            
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }

            # Configuration spécifique pour GPU
            if self.device == "cuda":
                model_kwargs.update({
                    "device_map": "auto",
                    "max_memory": {0: "22GiB"},  # Réservé pour la RTX 3090
                    "quantization_config": self.quantization_config if settings.USE_4BIT else None
                })
            
            # Chargement avec retry et monitoring
            for attempt in range(3):
                try:
                    start_time = time.time()
                    self.model = AutoModelForCausalLM.from_pretrained(
                        settings.MODEL_NAME,
                        **model_kwargs
                    )
                    
                    load_time = time.time() - start_time
                    logger.info(f"Modèle chargé en {load_time:.2f}s")
                    
                    if self.device == "cuda":
                        memory_allocated = torch.cuda.memory_allocated() / 1e9
                        logger.info(f"Mémoire GPU allouée: {memory_allocated:.2f}GB")
                    
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"Tentative {attempt + 1} échouée: {e}")
                    self._cleanup_memory()
                    time.sleep(30)

            # Optimisations post-chargement
            self.model.eval()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
                
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise
    def _setup_generation_config(self):
        """Configure les paramètres de génération optimisés."""
        self.generation_config = GenerationConfig(
            max_new_tokens=int(settings.MAX_NEW_TOKENS),
            min_new_tokens=int(settings.MIN_NEW_TOKENS),
            do_sample=settings.DO_SAMPLE,
            temperature = max(float(settings.TEMPERATURE), 1e-6),
            top_p=settings.TOP_P,
            top_k=settings.TOP_K,
            num_beams=1,  # Optimisé pour la performance
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0,
            #early_stopping=True
        )

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
    ) -> Dict[str, Any]:
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
                "source": "model",  
                "confidence": 1.0,  
                "generation_time": metrics.timers.get("model_inference", 0.0)
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
