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

            if settings.USE_4BIT or settings.USE_8BIT:
                if not self._verify_bnb_installation():
                    logger.warning("Désactivation de la quantification")
                    settings.USE_4BIT = False
                    settings.USE_8BIT = False
                    
            # Configuration BitsAndBytes pour quantification 4-bit
            self.quantization_config = None
            if settings.USE_4BIT:
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
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
            
    def _verify_bnb_installation(self):
        """Vérifie l'installation de BitsAndBytes."""
        try:
            import bitsandbytes as bnb
            if not bnb.cuda.COMPILED_WITH_CUDA:
                logger.warning("BitsAndBytes n'est pas compilé avec CUDA")
                return False
                
            # Vérification de la version CUDA
            cuda_version = torch.version.cuda
            if cuda_version:
                logger.info(f"Version CUDA détectée: {cuda_version}")
                if not cuda_version.startswith("12."):
                    logger.warning(f"Version CUDA {cuda_version} pourrait ne pas être compatible")
            
            return True
        except ImportError:
            logger.error("BitsAndBytes n'est pas installé correctement")
            return False
            
    def _setup_cuda_environment(self):
        if torch.cuda.is_available() and not settings.USE_CPU_ONLY:
            # Optimisations RTX 3090
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Flash Attention si disponible
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                torch.backends.cuda.enable_flash_sdp(True)
            
            # Configuration mémoire optimisée
            torch.cuda.set_per_process_memory_fraction(0.85)
            
            # Environnement CUDA
            os.environ.update({
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:4096"
            })
            
            # Optimisation pour architecture Ampere
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.set_float32_matmul_precision('high')

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
        try:
            logger.info(f"Chargement du modèle {settings.MODEL_NAME}")
            
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "use_flash_attention_2": True,
                "trust_remote_code": True,
                "max_memory": {0: "20GiB"},  # RTX 3090 optimisé
                "quantization_config": self.quantization_config if settings.USE_4BIT else None,
                "low_cpu_mem_usage": True
            }

            if not settings.USE_4BIT and settings.USE_8BIT:
                model_kwargs["load_in_8bit"] = True
    
            # Configuration dispatch GPU
            if torch.cuda.is_available():
                n_gpu_layers = 32  # Optimisé pour 24GB VRAM
                model_kwargs.update({
                    "device_map": "balanced",
                    "max_memory": {0: "20GiB", "cpu": "24GiB"},
                    "offload_folder": "offload_folder",
                })
    
            # Chargement avec retry et monitoring
            for attempt in range(3):
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        settings.MODEL_NAME,
                        **model_kwargs
                    )
                    
                    # Post-chargement optimisations
                    if torch.cuda.is_available():
                        self.model.tie_weights()
                        torch.cuda.empty_cache()
                    
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"Tentative {attempt + 1} échouée: {e}")
                    self._cleanup_memory()
                    time.sleep(30)
    
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
            top_p=max(float(settings.TOP_P), 0.0),
            top_k=max(int(settings.TOP_K), 1),
            num_beams=1,  # Optimisé pour la performance
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=max(float(settings.REPETITION_PENALTY), 1.0),
            length_penalty=float(settings.LENGTH_PENALTY)
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
