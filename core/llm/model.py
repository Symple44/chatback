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
from .auth_manager import HuggingFaceAuthManager

logger = get_logger("model")

class ModelInference:
    def __init__(self):
        """Initialise le modèle d'inférence."""
        try:
            self.auth_manager = HuggingFaceAuthManager()
            self.cuda_manager = CUDAManager()
            self.memory_manager = MemoryManager()
            self.tokenizer_manager = TokenizerManager()
            self.prompt_builder = PromptBuilder()
            
            self._setup_model()
            
        except Exception as e:
            logger.error(f"Erreur initialisation modèle: {e}")
            raise

    async def _setup_model(self):
        """Configure et initialise le modèle."""
        try:
            # Authentification Hugging Face
            if not await self.auth_manager.setup_auth():
                raise ValueError("Échec de l'authentification Hugging Face")

            # Vérification de l'accès au modèle
            if not self.auth_manager.get_model_access(settings.MODEL_NAME):
                raise ValueError(f"Accès non autorisé au modèle {settings.MODEL_NAME}")

            # Initialisation du modèle
            self._initialize_model()

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

# core/llm/cuda_manager.py
import torch
from typing import Dict, Any
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("cuda_manager")

class CUDAManager:
    def __init__(self):
        """Initialise le gestionnaire CUDA."""
        self.setup_cuda_environment()

    def setup_cuda_environment(self):
        """Configure l'environnement CUDA."""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA non disponible - utilisation du CPU")
                return

            # Configuration du device
            device_id = 0
            torch.cuda.set_device(device_id)
            
            # Logging des informations GPU
            self._log_gpu_info(device_id)
            
        except Exception as e:
            logger.error(f"Erreur configuration CUDA: {e}")
            raise

    def _log_gpu_info(self, device_id: int):
        """Log les informations sur le GPU."""
        try:
            gpu_name = torch.cuda.get_device_name(device_id)
            gpu_cap = torch.cuda.get_device_capability(device_id)
            gpu_mem = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            
            logger.info(f"GPU {device_id}: {gpu_name} (Compute {gpu_cap[0]}.{gpu_cap[1]})")
            logger.info(f"Mémoire GPU totale: {gpu_mem:.2f} GB")
            logger.info(f"Utilisation du device: cuda:{device_id}")
            
        except Exception as e:
            logger.error(f"Erreur logging GPU: {e}")

    def get_model_load_parameters(self, model_name: str, max_memory: Dict) -> Dict[str, Any]:
        """Retourne les paramètres optimaux pour le chargement du modèle."""
        load_params = {
            "pretrained_model_name_or_path": model_name,
            "revision": settings.MODEL_REVISION,
            "device_map": "auto",
            "max_memory": max_memory,
            "trust_remote_code": True,
        }

        if settings.USE_FP16:
            load_params["torch_dtype"] = torch.float16
        elif settings.USE_8BIT:
            load_params["load_in_8bit"] = True
        elif settings.USE_4BIT:
            load_params["load_in_4bit"] = True

        return load_params

    async def cleanup(self):
        """Nettoie les ressources CUDA."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# core/llm/memory_manager.py
import gc
import psutil
import torch
from typing import Dict
from core.utils.logger import get_logger
from core.config import settings

logger = get_logger("memory_manager")

class MemoryManager:
    def __init__(self):
        """Initialise le gestionnaire de mémoire."""
        self.cleanup_memory()

    def cleanup_memory(self):
        """Nettoie la mémoire."""
        try:
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1024**3
                max_memory = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"Mémoire GPU allouée: {allocated:.2f}GB")
                logger.info(f"Pic mémoire GPU: {max_memory:.2f}GB")

            memory = psutil.virtual_memory()
            logger.info(f"RAM disponible: {memory.available/1024**3:.2f}GB")
            logger.info(f"Utilisation RAM: {memory.percent}%")

        except Exception as e:
            logger.error(f"Erreur nettoyage mémoire: {e}")

    def get_optimal_memory_config(self) -> Dict:
        """Calcule la configuration mémoire optimale."""
        try:
            if not torch.cuda.is_available():
                return {"cpu": f"{psutil.virtual_memory().available / 1024**3:.0f}GB"}

            device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            cuda_limit = min(device_memory * 0.95, float(settings.MAX_MEMORY.get("0", "22").replace("GiB", "")))
            
            return {
                "0": f"{cuda_limit:.0f}GiB",
                "cpu": f"{psutil.virtual_memory().available / 1024**3:.0f}GB"
            }

        except Exception as e:
            logger.error(f"Erreur calcul config mémoire: {e}")
            return settings.MAX_MEMORY

    async def cleanup(self):
        """Nettoie les ressources mémoire."""
        self.cleanup_memory()

# core/llm/tokenizer_manager.py
from transformers import AutoTokenizer, GenerationConfig
from typing import Dict, Any
import torch
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("tokenizer_manager")

class TokenizerManager:
    def __init__(self):
        """Initialise le gestionnaire de tokenizer."""
        self._setup_tokenizer()

    def _setup_tokenizer(self):
        """Configure le tokenizer."""
        try:
            logger.info("Initialisation du tokenizer...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                revision=settings.MODEL_REVISION,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.generation_config = GenerationConfig(**settings.generation_config)
            logger.info(f"Tokenizer initialisé: {settings.MODEL_NAME}")

        except Exception as e:
            logger.error(f"Erreur configuration tokenizer: {e}")
            raise

    def encode(self, text: str, max_length: int = None) -> Dict[str, torch.Tensor]:
        """Encode le texte en tokens."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

    def decode_and_clean(self, token_ids: torch.Tensor) -> str:
        """Décode et nettoie la sortie du modèle."""
        response = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        if "Réponse:" in response:
            response = response.split("Réponse:")[-1].strip()

        return response

    async def cleanup(self):
        """Nettoie les ressources du tokenizer."""
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

# core/llm/prompt_builder.py
from typing import List, Dict, Optional
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("prompt_builder")

class PromptBuilder:
    def build_prompt(
        self,
        query: str,
        context_docs: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
        language: str = "fr"
    ) -> str:
        """Construit le prompt complet."""
        try:
            # Contexte des documents
            context = self._build_context(context_docs) if context_docs else ""
            
            # Historique de conversation
            history = self._build_history(conversation_history) if conversation_history else ""
            
            # Construction du prompt final
            prompt_parts = []
            
            if settings.SYSTEM_PROMPT:
                prompt_parts.append(f"Système: {settings.SYSTEM_PROMPT}")

            if context:
                prompt_parts.append(f"Contexte:\n{context}")

            if history:
                prompt_parts.append(f"Historique:\n{history}")

            prompt_parts.append(f"Question: {query}")
            prompt_parts.append("Réponse:")

            return "\n\n".join(prompt_parts)

        except Exception as e:
            logger.error(f"Erreur construction prompt: {e}")
            return query

    def _build_context(self, context_docs: List[Dict]) -> str:
        """Construit la section contexte du prompt."""
        context_parts = []
        for doc in context_docs:
            content = doc.get('content', '').strip()
            if content:
                source = doc.get('title', 'Document')
                page = doc.get('metadata', {}).get('page', '?')
                context_parts.append(f"[{source} (p.{page})]\n{content}")
        return "\n\n".join(context_parts)

    def _build_history(self, conversation_history: List[Dict]) -> str:
        """Construit la section historique du prompt."""
        history_parts = []
        for entry in conversation_history[-5:]:
            if isinstance(entry, dict) and 'query' in entry and 'response' in entry:
                history_parts.append(f"Q: {entry['query']}\nR: {entry['response']}")
        return "\n\n".join(history_parts)
