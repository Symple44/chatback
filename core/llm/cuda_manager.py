# core/llm/cuda_manager.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch
import torch.cuda
import torch.backends.cudnn as cudnn
import os
import json
import psutil
from enum import Enum
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("cuda_manager")

class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"

@dataclass
class CUDAConfig:
    """Configuration CUDA complète."""
    device_type: DeviceType = DeviceType.AUTO
    device_id: int = 0
    memory_fraction: float = 0.95
    compute_dtype: torch.dtype = torch.float16
    module_loading: str = "LAZY"
    max_split_size_mb: int = 4096
    gc_threshold: float = 0.9
    enable_tf32: bool = True
    allow_fp16: bool = True
    deterministic: bool = False
    benchmark: bool = True
    max_memory: Optional[Dict[str, str]] = None
    offload_folder: Optional[str] = None
    use_flash_attention: bool = True
    quantization_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.device_type == DeviceType.AUTO:
            self.device_type = DeviceType.CUDA if torch.cuda.is_available() else DeviceType.CPU

class CUDAManager:
    def __init__(self):
        """Initialise le gestionnaire CUDA."""
        self.config = self._load_config()
        self.device = self._setup_device()
        self._initialized = False
        self.memory_stats = {}

    def _load_config(self) -> CUDAConfig:
        """Charge et valide la configuration CUDA."""
        def str_to_bool(val) -> bool:
            """Convertit une string ou un booléen en booléen."""
            if isinstance(val, bool):
                return val
            return str(val).lower() == "true"
            
        try:
            # Chargement de la configuration mémoire
            try:
                max_memory = json.loads(settings.MAX_MEMORY)
            except (json.JSONDecodeError, AttributeError):
                max_memory = {"0": "22GiB", "cpu": "24GB"}

            # Configuration de la quantization
            quantization_config = None
            if settings.USE_8BIT or settings.USE_4BIT:
                quantization_config = {
                    "load_in_8bit": settings.USE_8BIT,
                    "load_in_4bit": settings.USE_4BIT,
                    "bnb_4bit_compute_dtype": getattr(torch, settings.BNB_4BIT_COMPUTE_DTYPE, "float16"),
                    "bnb_4bit_quant_type": settings.BNB_4BIT_QUANT_TYPE,
                    "bnb_4bit_use_double_quant": str_to_bool(settings.BNB_4BIT_USE_DOUBLE_QUANT)
                }

            return CUDAConfig(
                device_type=DeviceType.AUTO,
                device_id=int(settings.CUDA_VISIBLE_DEVICES),
                memory_fraction=float(settings.CUDA_MEMORY_FRACTION),
                compute_dtype=torch.float16 if settings.USE_FP16 else torch.float32,
                module_loading=settings.CUDA_MODULE_LOADING,
                max_split_size_mb=int(settings.PYTORCH_CUDA_ALLOC_CONF.split(",")[0].split(":")[1]),
                gc_threshold=float(settings.PYTORCH_CUDA_ALLOC_CONF.split(",")[1].split(":")[1]),
                enable_tf32=True,
                allow_fp16=settings.USE_FP16,
                deterministic=str_to_bool(settings.CUDNN_DETERMINISTIC),
                benchmark=str_to_bool(settings.CUDNN_BENCHMARK),
                max_memory=max_memory,
                offload_folder=settings.OFFLOAD_FOLDER,
                use_flash_attention=str_to_bool(settings.USE_FLASH_ATTENTION),
                quantization_config=quantization_config
            )
        except Exception as e:
            logger.error(f"Erreur chargement config CUDA: {e}")
            raise

    def _setup_device(self) -> torch.device:
        """Configure le device PyTorch."""
        if self.config.device_type == DeviceType.CPU:
            return torch.device("cpu")
            
        if not torch.cuda.is_available():
            logger.warning("CUDA non disponible - utilisation du CPU")
            return torch.device("cpu")
            
        try:
            # Sélection et configuration du GPU
            torch.cuda.set_device(self.config.device_id)
            device = torch.device(f"cuda:{self.config.device_id}")
            
            # Configuration de la mémoire
            if self.config.memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(
                    self.config.memory_fraction, 
                    self.config.device_id
                )
            
            return device
            
        except Exception as e:
            logger.error(f"Erreur setup device CUDA: {e}")
            return torch.device("cpu")

    async def initialize(self):
        """Initialise l'environnement CUDA de manière asynchrone."""
        if self._initialized:
            return

        try:
            # Nettoyage initial
            self.cleanup_memory()

            # Configuration CUDA
            if self.config.device_type == DeviceType.CUDA:
                # Configuration cuDNN
                cudnn.enabled = True
                cudnn.benchmark = self.config.benchmark
                cudnn.deterministic = self.config.deterministic
                cudnn.allow_tf32 = self.config.enable_tf32

                # Configuration des opérations tensorielles
                torch.backends.cuda.matmul.allow_tf32 = self.config.enable_tf32
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = self.config.allow_fp16

                # Création du dossier d'offload si nécessaire
                if self.config.offload_folder:
                    os.makedirs(self.config.offload_folder, exist_ok=True)

            # Configuration des threads CPU
            torch.set_num_threads(int(settings.MKL_NUM_THREADS))
            
            self._initialized = True
            self.update_memory_stats()
            self._log_system_info()
            
        except Exception as e:
            logger.error(f"Erreur initialisation CUDA: {e}")
            raise

    def get_model_load_parameters(self) -> Dict[str, Any]:
        """Retourne les paramètres optimaux pour le chargement du modèle."""
        # Conversion du format de la mémoire
        try:
            max_memory_config = json.loads(settings.MAX_MEMORY)
            # Convertir les clés pour utiliser le format correct de device
            corrected_memory = {}
            for k, v in max_memory_config.items():
                if k.isdigit():
                    corrected_memory[f"cuda:{k}"] = v
                else:
                    corrected_memory[k] = v
        except (json.JSONDecodeError, AttributeError):
            corrected_memory = {"cuda:0": "22GiB", "cpu": "24GB"}

        load_params = {
            "torch_dtype": self.config.compute_dtype,
            "trust_remote_code": True
        }

        if self.config.device_type == DeviceType.CUDA:
            # Configuration du device mapping
            if self.config.quantization_config and self.config.quantization_config.get("load_in_8bit"):
                load_params["device_map"] = {
                    "model.embed_tokens": self.device,
                    "model.norm": self.device,
                    "lm_head": self.device,
                    "model": "auto"
                }
                load_params["quantization_config"] = self.config.quantization_config
            else:
                load_params["device_map"] = "auto"

            # Configuration de l'attention
            if self.config.use_flash_attention and not self.config.quantization_config:
                load_params["attn_implementation"] = "flash_attention_2"
            
            # Configuration de la mémoire
            load_params["max_memory"] = corrected_memory
                
            load_params.update({
                "low_cpu_mem_usage": True,
                "offload_folder": self.config.offload_folder
            })

        return load_params

    def cleanup_memory(self):
        """Nettoie la mémoire CUDA."""
        if self.config.device_type == DeviceType.CUDA:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as e:
                logger.error(f"Erreur nettoyage mémoire CUDA: {e}")

    def update_memory_stats(self):
        """Met à jour les statistiques de mémoire."""
        try:
            self.memory_stats = {
                "cuda": {
                    "allocated": f"{torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB",
                    "reserved": f"{torch.cuda.memory_reserved(self.device) / 1024**3:.2f} GB",
                    "max_allocated": f"{torch.cuda.max_memory_allocated(self.device) / 1024**3:.2f} GB"
                } if self.config.device_type == DeviceType.CUDA else {},
                "ram": {
                    "total": f"{psutil.virtual_memory().total / 1024**3:.2f} GB",
                    "available": f"{psutil.virtual_memory().available / 1024**3:.2f} GB",
                    "used": f"{psutil.virtual_memory().used / 1024**3:.2f} GB",
                    "percent": f"{psutil.virtual_memory().percent}%"
                }
            }
        except Exception as e:
            logger.error(f"Erreur mise à jour stats mémoire: {e}")

    def _log_system_info(self):
        """Log les informations système détaillées."""
        try:
            if self.config.device_type == DeviceType.CUDA:
                device = torch.cuda.current_device()
                gpu_properties = torch.cuda.get_device_properties(device)
                
                cuda_info = {
                    "name": torch.cuda.get_device_name(device),
                    "compute_capability": f"{gpu_properties.major}.{gpu_properties.minor}",
                    "total_memory": f"{gpu_properties.total_memory / 1024**3:.2f} GB",
                    "multi_processor_count": gpu_properties.multi_processor_count,
                    "device_id": device,
                    "cuda_version": torch.version.cuda
                }

                logger.info("Configuration GPU:")
                for key, value in cuda_info.items():
                    logger.info(f"  {key}: {value}")

            logger.info("Configuration mémoire:")
            for device_type, stats in self.memory_stats.items():
                for key, value in stats.items():
                    logger.info(f"  {device_type}.{key}: {value}")

        except Exception as e:
            logger.error(f"Erreur logging système: {e}")

    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            self.cleanup_memory()
            self._initialized = False
            logger.info("Ressources CUDA nettoyées")
        except Exception as e:
            logger.error(f"Erreur nettoyage ressources: {e}")

    @property
    def is_initialized(self) -> bool:
        """Retourne l'état d'initialisation."""
        return self._initialized

    @property
    def current_device(self) -> torch.device:
        """Retourne le device courant."""
        return self.device
