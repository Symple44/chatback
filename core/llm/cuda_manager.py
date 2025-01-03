# core/llm/cuda_manager.py
from typing import Dict, Any
import torch
import torch.cuda
import torch.backends.cudnn as cudnn
from dataclasses import dataclass
import logging
import json
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("cuda_manager")

@dataclass
class CUDAConfig:
    """Configuration CUDA avec support INT8."""
    device_id: int = 0
    memory_fraction: float = 0.95
    torch_dtype: torch.dtype = torch.float16
    module_loading: str = "LAZY"
    launch_blocking: bool = False
    enable_timing: bool = False
    max_split_size_mb: int = 4096
    garbage_collection_threshold: float = 0.9
    enable_tf32: bool = True
    allow_fp16_reduced_precision: bool = True
    deterministic: bool = False
    benchmark: bool = True
    max_memory: Dict[str, str] = None
    offload_folder: str = None
    use_flash_attention: bool = True

class CUDAManager:
    def __init__(self):
        """Initialise le gestionnaire CUDA."""
        self.config = self._load_config()
        self.setup_cuda_environment()

    def _load_config(self) -> CUDAConfig:
        """Charge la configuration CUDA depuis les settings."""
        try:
            max_memory = json.loads(settings.MAX_MEMORY) if hasattr(settings, 'MAX_MEMORY') else None
        except json.JSONDecodeError:
            logger.warning("Erreur parsing MAX_MEMORY, utilisation des valeurs par défaut")
            max_memory = {"0": "22GiB", "cpu": "24GB"}

        # Conversion explicite des strings en booléens
        def str_to_bool(val: str) -> bool:
            return str(val).lower() == 'true'

        return CUDAConfig(
            device_id=int(settings.CUDA_VISIBLE_DEVICES),
            memory_fraction=float(settings.CUDA_MEMORY_FRACTION),
            module_loading=settings.CUDA_MODULE_LOADING,
            max_split_size_mb=int(settings.PYTORCH_CUDA_ALLOC_CONF.split(",")[0].split(":")[1]),
            garbage_collection_threshold=float(settings.PYTORCH_CUDA_ALLOC_CONF.split(",")[1].split(":")[1]),
            benchmark=str_to_bool(settings.CUDNN_BENCHMARK),
            deterministic=str_to_bool(settings.CUDNN_DETERMINISTIC),
            max_memory=max_memory,
            offload_folder=settings.OFFLOAD_FOLDER,
            use_flash_attention=str_to_bool(settings.USE_FLASH_ATTENTION)
        )

    def setup_cuda_environment(self):
        """Configure l'environnement CUDA de manière optimale."""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA non disponible - utilisation du CPU")
                return

            # Configuration du device
            torch.cuda.set_device(self.config.device_id)
            
            # Optimisation de la mémoire
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_stats'):
                logger.info(f"Statistiques mémoire initiales: {torch.cuda.memory_stats()}")

            # Configuration de cuDNN
            cudnn.enabled = True
            cudnn.benchmark = self.config.benchmark
            cudnn.deterministic = self.config.deterministic
            cudnn.allow_tf32 = self.config.enable_tf32

            # Configuration des opérations tensorielles
            torch.backends.cuda.matmul.allow_tf32 = self.config.enable_tf32
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
                self.config.allow_fp16_reduced_precision
            )

            # Configuration de la mémoire
            if self.config.memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(
                    self.config.memory_fraction,
                    self.config.device_id
                )

            # Activation du garbage collector CUDA
            if hasattr(torch.cuda, 'memory_stats'):
                torch.cuda.memory_stats(device=self.config.device_id)
            
            # Configuration des streams
            torch.cuda.Stream(device=self.config.device_id)
            
            # Logging des informations GPU
            self._log_gpu_info()
            
        except Exception as e:
            logger.error(f"Erreur configuration CUDA: {e}")
            raise

    def _log_gpu_info(self):
        """Log les informations détaillées sur le GPU."""
        try:
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)

            info = {
                "name": torch.cuda.get_device_name(device),
                "compute_capability": f"{gpu_properties.major}.{gpu_properties.minor}",
                "total_memory": f"{gpu_properties.total_memory / 1024**3:.2f} GB",
                "multi_processor_count": gpu_properties.multi_processor_count,
                "device_id": device,
                "cuda_version": torch.version.cuda,
                "memory_allocated": f"{torch.cuda.memory_allocated(device) / 1024**3:.2f} GB",
                "memory_reserved": f"{torch.cuda.memory_reserved(device) / 1024**3:.2f} GB"
            }

            logger.info("Configuration GPU:")
            for key, value in info.items():
                logger.info(f"  {key}: {value}")

        except Exception as e:
            logger.error(f"Erreur logging GPU: {e}")

    def get_model_load_parameters(self) -> Dict[str, Any]:
        """Retourne les paramètres optimaux pour le chargement du modèle."""
        load_params = {
            "device_map": "auto",
            "torch_dtype": self.config.torch_dtype,
            "max_memory": self.config.max_memory,
            "offload_folder": self.config.offload_folder,
            "use_flash_attention": self.config.use_flash_attention,
            "trust_remote_code": True
        }

        # Configuration de la quantification
        if settings.USE_8BIT:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=float(settings.LLM_INT8_THRESHOLD),
                llm_int8_enable_fp32_cpu_offload=settings.LLM_INT8_ENABLE_FP32_CPU_OFFLOAD.lower() == "true",
                llm_int8_skip_modules=None,
                quant_method="bitsandbytes"  # Spécification explicite de la méthode
            )
            load_params["quantization_config"] = quantization_config
            
            if settings.LLM_INT8_ENABLE_FP32_CPU_OFFLOAD.lower() == "true":
                logger.info("Activation de l'offloading CPU FP32 pour INT8")
                load_params["device_map"] = {
                    "transformer.word_embeddings": "cpu",
                    "transformer.final_layernorm": "cpu",
                    "lm_head": "cpu"
                }

        elif settings.USE_4BIT:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, settings.BNB_4BIT_COMPUTE_DTYPE),
                bnb_4bit_quant_type=settings.BNB_4BIT_QUANT_TYPE,
                bnb_4bit_use_double_quant=settings.BNB_4BIT_USE_DOUBLE_QUANT.lower() == "true",
                quant_method="bitsandbytes"
            )
            load_params["quantization_config"] = quantization_config

        return load_params

    async def cleanup(self):
        """Nettoie les ressources CUDA."""
        if torch.cuda.is_available():
            try:
                # Libération de la mémoire CUDA
                torch.cuda.empty_cache()
                
                # Synchronisation des streams
                torch.cuda.current_stream().synchronize()
                
                # Reset du device
                torch.cuda.device(self.config.device_id).empty_cache()
                
                logger.info("Ressources CUDA nettoyées")
                
            except Exception as e:
                logger.error(f"Erreur nettoyage CUDA: {e}")

    @property
    def device(self) -> torch.device:
        """Retourne le device courant."""
        return torch.device(f"cuda:{self.config.device_id}" if torch.cuda.is_available() else "cpu")

    def memory_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de mémoire actuelles."""
        if not torch.cuda.is_available():
            return {}
            
        return {
            "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            "cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
            "max_allocated": f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        }