# core/llm/model.py
from typing import List, Dict, Optional, Any, AsyncIterator, Union
from pathlib import Path
import torch
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
            # Initialisation des flags de contrôle
            self._cuda_initialized = False
            self._model_loaded = False
            self.model = None
            self.tokenizer = None
            
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
            self._setup_tokenizer()  # Doit réussir avant de continuer
            if not self.tokenizer:
                raise RuntimeError("Échec de l'initialisation du tokenizer")
            self._setup_model()  # Initialisation du modèle après le tokenizer
            self._setup_generation_config()
            
            self._model_loaded = True
            logger.info("Modèle initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation modèle: {e}")
            asyncio.create_task(self.cleanup())  # Nettoyage asynchrone
            raise

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
    
    def _get_optimal_device(self) -> str:
        """Détermine le meilleur device disponible avec vérification détaillée."""
        if settings.USE_CPU_ONLY:
            return "cpu"
    
        try:
            if torch.cuda.is_available() and self._verify_cuda_setup():
                return f"cuda:{torch.cuda.current_device()}"
        except Exception as e:
            logger.warning(f"Erreur détection CUDA: {e}")
        
        return "cpu"

    def _log_gpu_info(self):
        """Log les informations sur les GPUs disponibles."""
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"Mémoire GPU {i}: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    def _setup_quantization(self):
        """Configure les paramètres de quantification."""
        self.quantization_config = None
        try:
            if settings.USE_4BIT:
                if not self._verify_bnb_installation():
                    logger.warning("Désactivation de la quantification 4-bit")
                    return
                
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
        except Exception as e:
            logger.error(f"Erreur configuration quantification: {e}")

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
        try:
            logger.info(f"Chargement du modèle {settings.MODEL_NAME}")
            
            # Configuration de base du modèle
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "max_memory": {
                    0: f"{int(settings.MEMORY_LIMIT)}MB",
                    "cpu": "24GB"
                }
            }

            # Ajout de la configuration de quantification si activée
            if self.quantization_config:
                model_kwargs["quantization_config"] = self.quantization_config
            
            # Configuration du modèle
            config = AutoConfig.from_pretrained(
                settings.MODEL_NAME,
                trust_remote_code=True,
                revision=settings.MODEL_REVISION
            )
            
            # Optimisations
            config.use_cache = True
            config.pretraining_tp = settings.MODEL_PARALLEL_SIZE
            
            # Nettoyage préventif
            self._cleanup_memory()
            
            # Chargement du modèle
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_NAME,
                config=config,
                **model_kwargs
            )
            
            # Post-configuration
            if torch.cuda.is_available():
                self.model.eval()
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.enable_input_require_grads()
            
            logger.info("Modèle chargé avec succès")
            self._cleanup_memory()
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            self._cleanup_memory()
            raise

    def _setup_tokenizer(self):
        logger.info("Initialisation du tokenizer...")
        try:
            # Charger le tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                use_fast=True,
                model_max_length=settings.MAX_INPUT_LENGTH
            )
            logger.info(f"Tokenizer chargé avec succès : {settings.MODEL_NAME}")
    
            # Configurer le token de padding si nécessaire
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Token de padding défini sur le token de fin (EOS).")
    
            # Tester le tokenizer
            self._test_tokenizer()
    
            logger.info("Configuration du tokenizer terminée avec succès.")
        except ValueError as ve:
            logger.error(f"Erreur de configuration : {ve}")
            raise ValueError(f"Configuration du tokenizer échouée : {ve}")
        except Exception as e:
            logger.error(f"Erreur imprévue lors de la configuration du tokenizer : {e}")
            raise RuntimeError(f"Erreur imprévue dans le tokenizer : {e}")

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

    async def generate_response(
        self,
        query: str,
        context_docs: List[Dict],
        conversation_history: Optional[List] = None,
        language: str = "fr",
        generation_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Génère une réponse."""
        try:
            prompt = self._prepare_prompt(
                query=query,
                context_docs=context_docs,
                conversation_history=conversation_history,
                language=language
            )
            
            inputs = self._tokenize(prompt)
            
            gen_config = self.generation_config
            if generation_config:
                gen_config = GenerationConfig(**{
                    **gen_config.to_dict(),
                    **generation_config
                })
            
            with metrics.timer("model_inference"):
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=gen_config,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
            
            response = self.tokenizer.decode(
                outputs.sequences[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            response = self._post_process_response(response)
            
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

    def _get_optimal_device(self) -> str:
        """Détermine le meilleur device disponible."""
        if settings.USE_CPU_ONLY:
            return "cpu"

        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                logger.info(f"Mémoire GPU totale détectée: {gpu_memory / 1e9:.2f} GB")
                
                test_tensor = torch.tensor([1.0], device="cuda")
                del test_tensor
                
                return "cuda"
        except Exception as e:
            logger.warning(f"Erreur détection CUDA: {e}")
        
        return "cpu"

    def _cleanup_memory(self):
        """
        Méthode pour nettoyer les ressources utilisées par le modèle et libérer la mémoire GPU/CPU.
        """
        logger.info("Début du nettoyage des ressources...")

        # Libération du modèle et du tokenizer
        try:
            if self.model is not None:
                logger.info("Déplacement du modèle sur CPU avant suppression.")
                self.model.cpu()  # Déplacement sur CPU pour éviter des blocages GPU
                del self.model
                logger.info("Modèle supprimé de la mémoire.")
            
            if self.tokenizer is not None:
                del self.tokenizer
                logger.info("Tokenizer supprimé de la mémoire.")
        except Exception as e:
            logger.warning(f"Erreur lors de la suppression du modèle ou du tokenizer: {e}")

        # Libération explicite de la mémoire GPU
        if torch.cuda.is_available():
            try:
                logger.info("Libération de la mémoire GPU.")
                torch.cuda.empty_cache()  # Vide le cache des allocations CUDA
                torch.cuda.synchronize()  # Synchronisation pour éviter des conflits asynchrones
            except Exception as e:
                logger.warning(f"Erreur lors de la libération de la mémoire GPU: {e}")
        
        # Nettoyage de la mémoire système
        try:
            logger.info("Collecte des ordures (Garbage Collection).")
            gc.collect()  # Collecte explicite des ordures pour libérer la mémoire CPU
        except Exception as e:
            logger.warning(f"Erreur lors du Garbage Collection: {e}")

        logger.info("Nettoyage des ressources terminé.")

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
    
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenise le texte avec gestion des erreurs."""
        try:
            # Vérification du tokenizer
            if not self._verify_tokenizer():
                raise RuntimeError("Tokenizer non fonctionnel")
                
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=settings.MAX_INPUT_LENGTH,
                padding=True
            ).to(self.device)
    
            # Log de debug pour la taille des inputs
            input_length = inputs["input_ids"].shape[-1]
            logger.debug(f"Longueur des tokens d'entrée: {input_length}")
            
            if input_length >= settings.MAX_INPUT_LENGTH:
                logger.warning("Entrée tronquée à la longueur maximale")
            
            return inputs
            
        except Exception as e:
            logger.error(f"Erreur lors de la tokenisation: {e}")
            raise

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

    async def cleanup(self):
        """Nettoie les ressources de manière asynchrone."""
        try:
            logger.info("Début du nettoyage des ressources...")
            
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
            
            if hasattr(self, 'embeddings'):
                await self.embeddings.cleanup()
            
            if hasattr(self, 'model'):
                try:
                    self.model.cpu()
                    del self.model
                except Exception as e:
                    logger.warning(f"Erreur nettoyage modèle: {e}")
            
            if hasattr(self, 'tokenizer'):
                try:
                    del self.tokenizer
                except Exception as e:
                    logger.warning(f"Erreur nettoyage tokenizer: {e}")
            
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
