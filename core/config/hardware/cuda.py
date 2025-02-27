# core/config/hardware/cuda.py
from enum import Enum
from typing import Dict, Any
import os
from dataclasses import dataclass

# Import avec gestion d'erreur
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

class DeviceType(str, Enum):
    """Types de dispositifs disponibles."""
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"

class ModelPriority(str, Enum):
    """Priorités pour l'allocation de mémoire par type de modèle."""
    HIGH = "high"      # Modèles principaux de chat
    MEDIUM = "medium"  # Modèles de summarization
    LOW = "low"        # Modèles d'embedding

@dataclass
class CUDAConfig:
    """Configuration CUDA complète."""
    device_type: DeviceType = DeviceType.AUTO
    device_id: int = 0
    memory_configs: Dict[ModelPriority, Dict[str, str]] = None
    compute_dtype: str = "float16"
    module_loading: str = "LAZY"
    max_split_size_mb: int = 4096
    gc_threshold: float = 0.9
    enable_tf32: bool = True
    efficient_offload: bool = True
    allow_fp16: bool = True
    deterministic: bool = False
    benchmark: bool = True
    use_flash_attention: bool = True
    offload_folder: str = "offload_folder"
    
    @staticmethod
    def from_profile(profile: Dict[str, Any]) -> 'CUDAConfig':
        """
        Crée une configuration CUDA à partir d'un profil matériel.
        
        Args:
            profile: Profil matériel
            
        Returns:
            Configuration CUDA
        """
        # Auto-détection du type de dispositif
        device_type = DeviceType.CUDA if CUDA_AVAILABLE else DeviceType.CPU
        
        # Extraction des configs CUDA
        cuda_config = profile["gpu"]["cuda_config"]
        
        # Configuration mémoire par type de modèle
        memory_configs = {
            ModelPriority.HIGH: {"0": f"{profile['gpu']['vram_allocation']['model'].replace('GB', '')}GiB", 
                                "cpu": profile["cpu"]["ram_allocation"]["model_offload"]},
            ModelPriority.MEDIUM: {"0": f"{profile['gpu']['vram_allocation'].get('cache', '4').replace('GB', '')}GiB", 
                                  "cpu": "8GB"},
            ModelPriority.LOW: {"0": f"{profile['gpu']['vram_allocation']['embeddings'].replace('GB', '')}GiB", 
                               "cpu": "4GB"}
        }
        
        return CUDAConfig(
            device_type=device_type,
            device_id=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
            memory_configs=memory_configs,
            compute_dtype="float16" if os.environ.get("USE_FP16", "true").lower() == "true" else "float32",
            module_loading=os.environ.get("CUDA_MODULE_LOADING", "LAZY"),
            max_split_size_mb=cuda_config.get("max_split_size_mb", 4096),
            gc_threshold=float(os.environ.get("CUDA_MEMORY_FRACTION", "0.95")),
            enable_tf32=True,
            efficient_offload=True,
            allow_fp16=os.environ.get("USE_FP16", "true").lower() == "true",
            deterministic=False,
            benchmark=os.environ.get("CUDNN_BENCHMARK", "true").lower() == "true",
            use_flash_attention=cuda_config.get("flash_attention", True)
        )