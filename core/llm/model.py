# core/llm/model.py
from typing import List, Dict, Optional, Any, AsyncIterator, Union
from pathlib import Path
import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoConfig, 
    TextIteratorStreamer,
    GenerationConfig,
    BitsAndBytesConfig
)
import asyncio
from datetime import datetime
import time
import gc
import psutil
import os
import json
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
            # Vérification et configuration de l'authentification HF
            self._setup_huggingface_auth()
            
            # Initialisation des flags de contrôle
            self._cuda_initialized = False
            self._model_loaded = False
            self._tokenizer_loaded = False
            self.tokenizer = None
            self.model = None
            self.generation_config = None
            self.quantization_config = None
            
            # Nettoyage initial
            self._cleanup_memory()
            
            # Configuration CUDA si activée
            if not settings.USE_CPU_ONLY:
                self._setup_cuda_environment()
    
            # Détermination du device optimal
            self.device = self._get_optimal_device()
            logger.info(f"Utilisation du device: {self.device}")
    
            # Log des informations GPU uniquement si nécessaire
            if "cuda" in self.device and not self._cuda_initialized:
                self._log_gpu_info()
    
            # Configuration de la quantification après vérification CUDA
            self._setup_quantization()
            
            # Initialisation des composants dans l'ordre correct
            self.executor = ThreadPoolExecutor(max_workers=2)
            self.embeddings = EmbeddingsManager()
            
            # Fallback sur un modèle open source si besoin
            if not self._verify_model_access():
                logger.warning(f"Accès impossible à {settings.MODEL_NAME}, utilisation du modèle de fallback")
                self._use_fallback_model()
            
            # Initialiser d'abord le tokenizer
            self._setup_tokenizer()
            if not self.tokenizer:
                raise RuntimeError("Échec de l'initialisation du tokenizer")
                
            # Configurer la génération après le tokenizer
            self._setup_generation_config()
            
            # Initialiser le modèle en dernier
            self._setup_model()
            
            self._model_loaded = True
            logger.info("Modèle initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation modèle: {e}")
            asyncio.create_task(self.cleanup())  # Nettoyage asynchrone
            raise

    def _setup_huggingface_auth(self):
        """Configure l'authentification Hugging Face."""
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            logger.warning("Token Hugging Face non trouvé dans les variables d'environnement")
            return False
            
        try:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("Authentification Hugging Face réussie")
            return True
        except Exception as e:
            logger.error(f"Erreur authentification Hugging Face: {e}")
            return False

    def _verify_model_access(self):
        """Vérifie l'accès au modèle configuré."""
        try:
            from huggingface_hub import model_info
            info = model_info(settings.MODEL_NAME)
            return not info.private
        except Exception as e:
            logger.error(f"Erreur vérification accès modèle: {e}")
            return False

    def _use_fallback_model(self):
        """Configure un modèle de fallback open source."""
        FALLBACK_MODELS = [
            "mistralai/Mistral-7B-v0.2",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta"
        ]
        
        for model_name in FALLBACK_MODELS:
            try:
                from huggingface_hub import model_info
                info = model_info(model_name)
                if not info.private:
                    settings.MODEL_NAME = model_name
                    logger.info(f"Utilisation du modèle de fallback: {model_name}")
                    return True
            except Exception:
                continue
                
        raise RuntimeError("Aucun modèle de fallback disponible")

    def _setup_cuda_paths(self):
        """Configure les chemins CUDA."""
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
        os.environ["PATH"] = f"{os.environ['PATH']}:/usr/local/cuda/bin"

    def _setup_cuda_environment(self):
        """Configure l'environnement CUDA en évitant les doubles initialisations."""
        if torch.cuda.is_available():
            try:
                # Vérification de la version CUDA
                cuda_version = torch.version.cuda
                logger.info(f"Version CUDA détectée: {cuda_version}")
                
                # Configuration unique du device CUDA
                device_id = int(settings.CUDA_VISIBLE_DEVICES)
                if device_id >= 0 and device_id < torch.cuda.device_count():
                    torch.cuda.set_device(device_id)
                else:
                    logger.warning(f"Device ID invalide: {device_id}, utilisation du device 0")
                    torch.cuda.set_device(0)
    
                # Configuration des optimisations PyTorch une seule fois
                if not getattr(self, '_cuda_initialized', False):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.enabled = True
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    torch.backends.cudnn.allow_tf32 = True
    
                    # Flash Attention uniquement si disponible et non déjà initialisé
                    if (hasattr(torch.nn.functional, 'scaled_dot_product_attention') and
                        not getattr(torch.backends.cuda, '_flash_sdp_initialized', False)):
                        torch.backends.cuda.enable_flash_sdp(True)
                        setattr(torch.backends.cuda, '_flash_sdp_initialized', True)
    
                    # Configuration mémoire
                    memory_fraction = float(settings.CUDA_MEMORY_FRACTION)
                    if memory_fraction > 0 and memory_fraction <= 1:
                        torch.cuda.set_per_process_memory_fraction(memory_fraction)
                    
                    # Marquer comme initialisé
                    self._cuda_initialized = True
                    
                # Configuration du garbage collector CUDA
                gc.collect()
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
    
                logger.info(f"Environnement CUDA configuré avec succès pour device {torch.cuda.current_device()}")
                
            except Exception as e:
                logger.error(f"Erreur configuration CUDA: {e}")
                raise
    
    def _verify_cuda_setup(self):
        """Vérifie la configuration CUDA."""
        if torch.cuda.is_available():
            try:
                # Test basique CUDA
                test_tensor = torch.zeros(1, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
    
                # Vérification des capacités
                device = torch.cuda.current_device()
                capabilities = torch.cuda.get_device_capability(device)
                device_name = torch.cuda.get_device_name(device)
                logger.info(f"GPU {device}: {device_name} (Compute {capabilities[0]}.{capabilities[1]})")
    
                # Vérification de la mémoire
                total_memory = torch.cuda.get_device_properties(device).total_memory
                logger.info(f"Mémoire GPU totale: {total_memory / 1e9:.2f} GB")
    
                return True
            except Exception as e:
                logger.error(f"Erreur vérification CUDA: {e}")
                return False
        return False

    def _log_gpu_info(self):
        """Log les informations sur les GPUs disponibles."""
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"Mémoire GPU {i}: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    def _setup_quantization(self):
        """Configure les paramètres de quantification."""
        self.quantization_config = None
        try:
            if settings.USE_4BIT and self._verify_bnb_installation():
                logger.info("Configuration quantification 4-bit")
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16 if settings.USE_FP16 else torch.float32,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True
                )
            elif settings.USE_8BIT and self._verify_bnb_installation():
                logger.info("Configuration quantification 8-bit")
                self.quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
            else:
                logger.info("Utilisation du format FP16 standard")
                self.quantization_config = None
                
        except Exception as e:
            logger.error(f"Erreur configuration quantification: {e}")
            self.quantization_config = None

    def _setup_memory_config(self):
        """Configure la gestion mémoire en utilisant les paramètres du .env."""
        try:
            # Configuration des devices reconnus par PyTorch
            max_memory_config = {}
            config_str = json.loads(settings.MAX_MEMORY)
            
            # Parse uniquement les devices valides (GPU et CPU)
            for key, value in config_str.items():
                if key.isdigit():  # GPU devices
                    max_memory_config[int(key)] = value
                elif key == 'cpu':
                    max_memory_config['cpu'] = value
    
            logger.info(f"Configuration mémoire depuis .env:")
            logger.info(f"MAX_MEMORY: {max_memory_config}")
    
            # Configuration quantification
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=settings.USE_4BIT,
                bnb_4bit_compute_dtype=torch.float16 if settings.USE_FP16 else torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_threshold=6.0
            )
    
            # Configuration du modèle
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": "auto",
                "max_memory": max_memory_config,
                "torch_dtype": torch.float16 if settings.USE_FP16 else torch.float32,
                "low_cpu_mem_usage": True,
                "offload_folder": settings.OFFLOAD_FOLDER,
                "offload_state_dict": True
            }
    
            # Configuration CUDA si disponible
            if torch.cuda.is_available():
                cuda_memory = torch.cuda.get_device_properties(0).total_memory
                if settings.CUDA_MEMORY_FRACTION:
                    cuda_limit = int(cuda_memory * float(settings.CUDA_MEMORY_FRACTION))
                    torch.cuda.set_per_process_memory_fraction(float(settings.CUDA_MEMORY_FRACTION))
                    logger.info(f"Limite mémoire CUDA configurée: {cuda_limit / 1e9:.2f}GB")
    
            # Log de la configuration finale
            logger.info("Configuration mémoire finalisée:")
            logger.info(f"Device map: {model_kwargs['device_map']}")
            logger.info(f"Max memory: {model_kwargs['max_memory']}")
            logger.info(f"Using FP16: {settings.USE_FP16}")
            logger.info(f"Using 4-bit: {settings.USE_4BIT}")
    
            return model_kwargs
    
        except Exception as e:
            logger.error(f"Erreur configuration mémoire: {e}")
            raise
    
        def _verify_bnb_installation(self):
            """Vérifie l'installation de BitsAndBytes."""
            try:
                import bitsandbytes as bnb
                if not torch.cuda.is_available():
                    return False
                try:
                    test_input = torch.zeros(1, 1, device='cuda')
                    test_layer = bnb.nn.Linear8bitLt(1, 1).cuda()
                    _ = test_layer(test_input)
                    
                    cc_major, cc_minor = torch.cuda.get_device_capability()
                    compute_capability = f"{cc_major}.{cc_minor}"
                    logger.info(f"Test BnB réussi - Capacité CUDA: {compute_capability}")
                    
                    return float(compute_capability) >= 8.0
                    
                except Exception as e:
                    logger.error(f"Erreur test BnB CUDA: {e}")
                    return False
                    
            except ImportError:
                return False

    def _setup_model(self):
        """Configure le modèle avec la nouvelle gestion mémoire."""
        try:
            logger.info(f"Chargement du modèle {settings.MODEL_NAME}")
                
            # Obtention de la configuration mémoire
            model_kwargs = self._setup_memory_config()
            if not model_kwargs:
                raise RuntimeError("Échec de la configuration mémoire")
                
            # Configuration du modèle
            config = AutoConfig.from_pretrained(
                settings.MODEL_NAME,
                trust_remote_code=True
            )
            
            config.use_cache = True
            config.pretraining_tp = 1
            
            logger.info("Début du chargement du modèle...")
            
            # Type de données pour le modèle
            torch_dtype = torch.float16 if settings.USE_FP16 else torch.float32
            
            # Chargement du modèle
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_NAME,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                quantization_config=self.quantization_config,
                device_map="auto",
                **model_kwargs
            )
            
            if self.model is None:
                raise RuntimeError("Échec du chargement du modèle")
    
            # Optimisation post-chargement
            if torch.cuda.is_available():
                self.model.eval()
                # Activation de torch.compile() pour les performances
                if hasattr(torch, 'compile') and not settings.DEBUG:
                    self.model = torch.compile(
                        self.model,
                        mode="reduce-overhead",
                        fullgraph=False
                    )
    
            self._model_loaded = True
            logger.info("Modèle chargé avec succès")
            
            # Nettoyage final
            self._cleanup_memory()
            
            return True
    
        except Exception as e:
            self._model_loaded = False
            logger.error(f"Erreur chargement modèle: {e}")
            raise

    def _setup_tokenizer(self):
        """Configure le tokenizer avec gestion des erreurs."""
        logger.info("Initialisation du tokenizer...")
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                self._tokenizer_loaded = False
                self.tokenizer = AutoTokenizer.from_pretrained(
                    settings.MODEL_NAME,
                    use_fast=True,
                    model_max_length=settings.MAX_INPUT_LENGTH,
                    trust_remote_code=True,
                    token=os.getenv("HUGGING_FACE_HUB_TOKEN")
                )
                
                if self.tokenizer is None:
                    raise ValueError("Tokenizer non initialisé")
                    
                logger.info(f"Tokenizer chargé avec succès: {settings.MODEL_NAME}")
                
                # Configuration du token de padding
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info("Token de padding configuré")
                
                # Test du tokenizer
                self._test_tokenizer()
                self._tokenizer_loaded = True
                return
                
            except Exception as e:
                logger.warning(f"Tentative {attempt + 1} échouée: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    raise RuntimeError(f"Échec initialisation tokenizer après {max_retries} tentatives") from e

    def _test_tokenizer(self):
        """
        Vérifie le bon fonctionnement du tokenizer en testant une entrée simple.
        """
        logger.info("Test du tokenizer en cours...")
        test_input = "Test simple du tokenizer"
    
        try:
            # Tokenisation
            tokens = self.tokenizer(test_input)
            logger.info(f"Tokens générés pour '{test_input}' : {tokens}")
    
            # Vérification du type de sortie
            if not isinstance(tokens, (dict, BatchEncoding)):
                logger.error(f"Type inattendu pour les tokens : {type(tokens)}")
                raise ValueError("Le tokenizer n'a pas retourné un dictionnaire ou un BatchEncoding.")
    
            # Vérification des clés nécessaires
            required_keys = {"input_ids", "attention_mask"}
            if isinstance(tokens, BatchEncoding):
                tokens = tokens.data  # Conversion explicite en dictionnaire
            missing_keys = required_keys - tokens.keys()
            if missing_keys:
                logger.error(f"Clés manquantes dans les tokens : {missing_keys}")
                raise ValueError(f"Clés manquantes dans les tokens : {missing_keys}")
    
            logger.info("Test du tokenizer réussi.")
        except Exception as e:
            logger.error(f"Erreur lors du test du tokenizer : {e}")
            raise ValueError(f"Test du tokenizer échoué : {e}")

    def _setup_generation_config(self):
        """Configure les paramètres de génération."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer non initialisé")

        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"
            
        try:
            self.generation_config = GenerationConfig(
                max_new_tokens=int(settings.MAX_NEW_TOKENS),
                min_new_tokens=int(settings.MIN_NEW_TOKENS),
                do_sample=settings.DO_SAMPLE.lower() == 'true',
                temperature=float(settings.TEMPERATURE),
                top_p=float(settings.TOP_P),
                top_k=int(settings.TOP_K),
                num_beams=int(settings.NUM_BEAMS),
                repetition_penalty=float(settings.REPETITION_PENALTY),
                length_penalty=float(settings.LENGTH_PENALTY),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            logger.info("Configuration de génération initialisée")
        except Exception as e:
            logger.error(f"Erreur configuration génération: {e}")
            raise

    async def generate_response(
        self,
        query: str,
        context_docs: List[Dict],
        conversation_history: Optional[List] = None,
        language: str = "fr",
        generation_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Génère une réponse avec meilleure gestion des erreurs."""
        try:
            # Vérification plus stricte de l'état
            if not self.is_model_ready():
                raise RuntimeError("Modèle non initialisé ou non fonctionnel")
                
            # Préparation du prompt
            prompt = self._prepare_prompt(
                query=query,
                context_docs=context_docs,
                conversation_history=conversation_history,
                language=language
            )
                
            # Tokenisation avec gestion d'erreur
            try:
                inputs = self._tokenize(prompt)
            except Exception as e:
                logger.error(f"Erreur tokenisation: {e}")
                raise RuntimeError("Erreur lors de la tokenisation") from e
                
            # Configuration de génération
            gen_config = self.generation_config
            if generation_config:
                gen_config = GenerationConfig(**{
                    **gen_config.to_dict(),
                    **generation_config
                })
                
            # Génération avec mesure du temps et gestion d'erreur
            with metrics.timer("model_inference"):
                with torch.inference_mode():
                    try:
                        outputs = self.model.generate(
                            **inputs,
                            generation_config=gen_config,
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                    except Exception as e:
                        logger.error(f"Erreur génération: {e}")
                        raise RuntimeError("Erreur lors de la génération") from e
                
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
        """Génère une réponse en streaming."""
        try:
            prompt = self._prepare_prompt(
                query=query,
                context_docs=context_docs,
                language=language
            )
            
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            inputs = self._tokenize(prompt)
            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "generation_config": self.generation_config
            }
            
            thread = Thread(target=self._generate_in_thread, kwargs=generation_kwargs)
            thread.start()
            
            buffer = ""
            chunk_size = int(os.getenv("STREAM_CHUNK_SIZE", "50"))
            stream_delay = float(os.getenv("STREAM_DELAY", "0.02"))
            
            async for token in self._async_tokens(streamer):
                buffer += token
                if len(buffer) >= chunk_size:
                    yield buffer
                    buffer = ""
                await asyncio.sleep(stream_delay)
            
            if buffer:
                yield buffer
                
        except Exception as e:
            logger.error(f"Erreur génération streaming: {e}")
            yield "Désolé, une erreur est survenue."

    def _check_available_memory(self) -> bool:
        """Vérifie la mémoire disponible avant le chargement du modèle."""
        try:
            if torch.cuda.is_available():
                # Vérification GPU
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_allocated = torch.cuda.memory_allocated()
                gpu_memory_available = gpu_memory - gpu_memory_allocated
                
                required_gpu_memory = 20 * (1024 ** 3)  # 20GB en bytes
                if gpu_memory_available < required_gpu_memory:
                    logger.warning(f"Mémoire GPU disponible insuffisante : {gpu_memory_available / 1e9:.2f}GB < {required_gpu_memory / 1e9:.2f}GB")
                    return False
                    
            # Vérification RAM
            ram = psutil.virtual_memory()
            required_ram = 24 * (1024 ** 3)  # 24GB en bytes
            if ram.available < required_ram:
                logger.warning(f"Mémoire RAM disponible insuffisante : {ram.available / 1e9:.2f}GB < {required_ram / 1e9:.2f}GB")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur vérification mémoire : {e}")
            return False
            
    def _get_optimal_device(self) -> str:
        """Détermine le meilleur device disponible avec vérification détaillée."""
        if settings.USE_CPU_ONLY:
            return "cpu"
    
        try:
            if torch.cuda.is_available() and self._verify_cuda_setup():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                logger.info(f"Mémoire GPU totale détectée: {gpu_memory / 1e9:.2f} GB")
                    
                # Test simple pour vérifier l'accès GPU
                test_tensor = torch.tensor([1.0], device="cuda")
                del test_tensor
                
                return f"cuda:{torch.cuda.current_device()}"
        except Exception as e:
            logger.warning(f"Erreur détection CUDA: {e}")
        
        return "cpu"

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
        history_limit = int(os.getenv("MAX_HISTORY_MESSAGES", "5"))
        if conversation_history:
            history_parts = []
            for msg in conversation_history[-history_limit:]:
                history_parts.append(
                    f"Human: {msg.get('user')}\n"
                    f"Assistant: {msg.get('assistant')}"
                )
            history = "\n".join(history_parts)
        
        # Construction du prompt final en utilisant le template de settings
        return settings.CHAT_TEMPLATE.format(
            system=settings.SYSTEM_PROMPT,
            query=query,
            context=context,
            history=history,
            language=language
        )

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
        stream_buffer_size = int(os.getenv("STREAM_BUFFER_SIZE", "10"))
        buffer = []
        
        try:
            for token in streamer:
                buffer.append(token)
                if len(buffer) >= stream_buffer_size:
                    combined = "".join(buffer)
                    yield await loop.run_in_executor(None, lambda: combined)
                    buffer = []
            
            if buffer:  # Vider le buffer restant
                combined = "".join(buffer)
                yield await loop.run_in_executor(None, lambda: combined)
        except Exception as e:
            logger.error(f"Erreur streaming tokens: {e}")

    def _verify_tokenizer(self) -> bool:
        """Vérifie que le tokenizer est correctement initialisé et fonctionnel."""
        try:
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                logger.error("Tokenizer non initialisé")
                return False
                
            # Test basique
            test_text = "Test du tokenizer"
            tokens = self.tokenizer(test_text, return_tensors="pt")
            
            # Vérifications
            required_keys = ["input_ids", "attention_mask"]
            for key in required_keys:
                if key not in tokens:
                    logger.error(f"Tokenizer mal configuré: {key} manquant")
                    return False
            
            # Test encode/decode
            decoded = self.tokenizer.decode(tokens["input_ids"][0])
            if not decoded or len(decoded.strip()) == 0:
                logger.error("Erreur decode tokenizer")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur vérification tokenizer: {e}")
            return False
            
    def _verify_tokenizer_state(self) -> bool:
        """Vérifie l'état complet du tokenizer avant utilisation."""
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            logger.error("Tokenizer non initialisé")
            return False
            
        if not self._tokenizer_loaded:
            logger.error("Tokenizer non chargé correctement")
            return False
            
        if not hasattr(self.tokenizer, 'encode') or not hasattr(self.tokenizer, 'decode'):
            logger.error("Méthodes du tokenizer manquantes")
            return False
            
        return True
    
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenise le texte avec gestion des erreurs."""
        try:
            # Vérification de l'état
            if not self._verify_tokenizer_state():
                raise RuntimeError("Tokenizer non fonctionnel")
                
            # Vérification de l'entrée
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Texte d'entrée invalide")
                
            # Tokenisation avec paramètres de sécurité
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=settings.MAX_INPUT_LENGTH,
                padding=True,
                return_attention_mask=True,
                add_special_tokens=True
            )
            
            # Vérification de la sortie
            if "input_ids" not in inputs or "attention_mask" not in inputs:
                raise RuntimeError("Sortie du tokenizer invalide")
                
            # Déplacement sur le bon device
            return {k: v.to(self.device) for k, v in inputs.items()}
            
        except Exception as e:
            logger.error(f"Erreur lors de la tokenisation: {e}")
            raise RuntimeError("Erreur de tokenisation")

    def _post_process_response(self, response: str) -> str:
        """Post-traite la réponse générée."""
        # Nettoyage des balises spéciales
        response = response.replace("<|assistant|>", "").replace("<|human|>", "")
        response = response.replace("<|system|>", "").replace("<|endoftext|>", "")
        
        # Nettoyage des espaces superflus
        response = "\n".join(line.strip() for line in response.split("\n"))
        response = "\n".join(filter(None, response.split("\n")))
        
        # Autres nettoyages basés sur les paramètres
        max_length = int(os.getenv("MAX_RESPONSE_LENGTH", "4096"))
        if len(response) > max_length:
            response = response[:max_length] + "..."
        
        return response.strip()

    async def create_embedding(self, text: str) -> List[float]:
        """Crée un embedding pour un texte."""
        return await self.embeddings.generate_embedding(text)

    def is_model_ready(self) -> bool:
        """Vérifie si le modèle est prêt à être utilisé."""
        return (
            hasattr(self, 'model') and 
            self.model is not None and 
            hasattr(self, 'tokenizer') and 
            self.tokenizer is not None and 
            self._tokenizer_loaded and
            self._model_loaded
        )
    
    def _ensure_model_ready(self):
        """S'assure que le modèle est prêt, sinon tente de le réinitialiser."""
        if not self._is_model_ready():
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                raise RuntimeError("Tokenizer non initialisé")
                
            # Si le tokenizer est ok mais pas le modèle, on tente de recharger
            if hasattr(self, 'model') and self.model is None:
                try:
                    logger.warning("Tentative de rechargement du modèle...")
                    self._setup_model()
                except Exception as e:
                    logger.error(f"Échec du rechargement du modèle: {e}")
                    raise RuntimeError("Échec de l'initialisation du modèle")

    def _cleanup_memory(self):
        """Nettoie la mémoire de manière optimisée."""
        logger.info("Nettoyage mémoire...")
        try:
            gc.collect()
            
            if torch.cuda.is_available():
                # Synchronisation GPU
                torch.cuda.synchronize()
                
                # Vider le cache CUDA
                torch.cuda.empty_cache()
                
                # Obtenir les statistiques mémoire
                allocated = torch.cuda.memory_allocated(0)
                max_allocated = torch.cuda.max_memory_allocated(0)
                
                logger.info(f"Mémoire GPU allouée: {allocated / 1e9:.2f}GB")
                logger.info(f"Pic mémoire GPU: {max_allocated / 1e9:.2f}GB")
                
                # Forcer la libération mémoire si nécessaire
                if allocated > 0.9 * torch.cuda.get_device_properties(0).total_memory:
                    torch.cuda.synchronize()
                    del self.model
                    self.model = None
                    torch.cuda.empty_cache()
                    logger.info("Libération forcée de la mémoire GPU")
                    
            # Obtenir les statistiques RAM
            memory = psutil.virtual_memory()
            logger.info(f"RAM disponible: {memory.available / 1e9:.2f}GB")
            logger.info(f"Utilisation RAM: {memory.percent}%")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage mémoire: {e}")
            
    async def cleanup(self):
        """Nettoie les ressources de manière asynchrone."""
        try:
            logger.info("Début du nettoyage des ressources...")
            
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
            
            if hasattr(self, 'embeddings'):
                await self.embeddings.cleanup()
            
            # Conserver le tokenizer chargé
            if hasattr(self, 'model'):
                try:
                    if torch.cuda.is_available():
                        self.model.cpu()
                    del self.model
                    self.model = None
                except Exception as e:
                    logger.warning(f"Erreur nettoyage modèle: {e}")
            
            # Nettoyage mémoire
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception as e:
                    logger.warning(f"Erreur nettoyage CUDA: {e}")
            
            logger.info("Nettoyage des ressources terminé")
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")
            
    def __del__(self):
        """Destructeur de la classe avec gestion de la coroutine cleanup."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                logger.warning("Boucle asyncio non disponible pour cleanup")
        except Exception as e:
            logger.error(f"Erreur dans le destructeur: {e}")
