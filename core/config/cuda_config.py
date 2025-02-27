# core/config/cuda_config.py
from typing import Dict, Any, Optional, Union, List
from enum import Enum
from dataclasses import dataclass, field
import os
import logging
import torch

logger = logging.getLogger(__name__)

# Vérification de la disponibilité CUDA
try:
    CUDA_AVAILABLE = torch.cuda.is_available()
    logger.info(f"CUDA disponible: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        CUDA_VERSION = torch.version.cuda
        logger.info(f"Version CUDA: {CUDA_VERSION}")
except Exception as e:
    CUDA_AVAILABLE = False
    CUDA_VERSION = None
    logger.warning(f"Erreur lors de la vérification CUDA: {e}")

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
    """Configuration CUDA complète et validée."""
    device_type: DeviceType = DeviceType.AUTO
    device_id: int = 0
    memory_configs: Dict[ModelPriority, Dict[str, str]] = field(default_factory=dict)
    compute_dtype: torch.dtype = torch.float16
    module_loading: str = "LAZY"
    max_split_size_mb: int = 4096
    gc_threshold: float = 0.9
    enable_tf32: bool = True
    allow_fp16: bool = True
    deterministic: bool = False
    benchmark: bool = True
    use_flash_attention: bool = True
    offload_folder: str = "offload_folder"

    def __post_init__(self):
        """Initialisation et validation supplémentaire."""
        # Auto-détection du type de dispositif
        if self.device_type == DeviceType.AUTO:
            self.device_type = DeviceType.CUDA if CUDA_AVAILABLE else DeviceType.CPU
        
        # Configuration mémoire par défaut si non spécifiée
        if not self.memory_configs:
            self.memory_configs = {
                ModelPriority.HIGH: {"0": "16GiB", "cpu": "12GB"},    # Pour modèles de chat
                ModelPriority.MEDIUM: {"0": "4GiB", "cpu": "8GB"},   # Pour summarizers
                ModelPriority.LOW: {"0": "2GiB", "cpu": "4GB"}       # Pour embeddings
            }

# Configuration CUDA de base
BASE_CUDA_CONFIG = CUDAConfig(
    device_type=DeviceType.AUTO,
    device_id=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
    compute_dtype=torch.float16,
    module_loading=os.environ.get("CUDA_MODULE_LOADING", "LAZY"),
    max_split_size_mb=4096,
    gc_threshold=float(os.environ.get("CUDA_MEMORY_FRACTION", "0.95")),
    enable_tf32=True,
    allow_fp16=os.environ.get("USE_FP16", "true").lower() == "true",
    deterministic=False,
    benchmark=os.environ.get("CUDNN_BENCHMARK", "true").lower() == "true",
    use_flash_attention=True,
    offload_folder=os.environ.get("OFFLOAD_FOLDER", "offload_folder")
)

# Configuration CUDA optimisée pour RTX 3090
RTX3090_CUDA_CONFIG = CUDAConfig(
    device_type=DeviceType.CUDA,
    device_id=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
    memory_configs={
        ModelPriority.HIGH: {"0": "20GiB", "cpu": "20GB"},
        ModelPriority.MEDIUM: {"0": "4GiB", "cpu": "8GB"},
        ModelPriority.LOW: {"0": "2GiB", "cpu": "4GB"}
    },
    compute_dtype=torch.float16,
    module_loading="LAZY",
    max_split_size_mb=4096,
    gc_threshold=0.95,
    enable_tf32=True,
    allow_fp16=True,
    deterministic=False,
    benchmark=True,
    use_flash_attention=True,
    offload_folder="offload_folder"
)

# Configuration CUDA pour CPU seulement
CPU_ONLY_CONFIG = CUDAConfig(
    device_type=DeviceType.CPU,
    device_id=0,
    memory_configs={
        ModelPriority.HIGH: {"cpu": "16GB"},
        ModelPriority.MEDIUM: {"cpu": "8GB"},
        ModelPriority.LOW: {"cpu": "4GB"}
    },
    compute_dtype=torch.float32,
    module_loading="LAZY",
    max_split_size_mb=4096,
    gc_threshold=0.95,
    enable_tf32=False,
    allow_fp16=False,
    deterministic=False,
    benchmark=False,
    use_flash_attention=False
)

def get_cuda_config() -> CUDAConfig:
    """
    Détermine la configuration CUDA optimale basée sur le matériel détecté.
    
    Returns:
        CUDAConfig: Configuration CUDA optimisée
    """
    # Import ici pour éviter les dépendances circulaires
    from core.config.hardware_config import get_hardware_config, HardwareDetector
    
    # Détection matérielle
    hardware = HardwareDetector.detect_hardware()
    gpu_info = hardware["gpu"]
    hardware_config = get_hardware_config()
    
    # Si CUDA n'est pas disponible, retourner la configuration CPU
    if not CUDA_AVAILABLE:
        logger.info("CUDA non disponible, utilisation de la configuration CPU")
        return CPU_ONLY_CONFIG
    
    # Configuration basée sur le GPU détecté
    gpu_name = gpu_info.get("name", "").upper()
    
    if "3090" in gpu_name:
        logger.info("Utilisation de la configuration CUDA pour RTX 3090")
        config = RTX3090_CUDA_CONFIG
    else:
        logger.info(f"Utilisation de la configuration CUDA de base pour {gpu_name}")
        config = BASE_CUDA_CONFIG
    
    # Ajustements basés sur la configuration matérielle détectée
    if hardware_config:
        vram_allocation = hardware_config.get("gpu", {}).get("vram_allocation", {})
        if vram_allocation:
            # Conversion en format attendu par CUDAConfig
            config.memory_configs[ModelPriority.HIGH]["0"] = f"{vram_allocation.get('model', '16').replace('GB', '')}GiB"
            config.memory_configs[ModelPriority.MEDIUM]["0"] = f"{vram_allocation.get('cache', '4').replace('GB', '')}GiB"
            config.memory_configs[ModelPriority.LOW]["0"] = f"{vram_allocation.get('embeddings', '2').replace('GB', '')}GiB"
    
    return config

def setup_cuda_environment():
    """
    Configure les variables d'environnement CUDA basées sur les configurations optimales.
    """
    if not CUDA_AVAILABLE:
        logger.warning("CUDA n'est pas disponible, configuration ignorée")
        return
    
    config = get_cuda_config()
    
    # Configuration des variables d'environnement
    os.environ.update({
        "PYTORCH_CUDA_ALLOC_CONF": f"max_split_size_mb:{config.max_split_size_mb},"
                                   f"garbage_collection_threshold:{config.gc_threshold}",
        "PYTORCH_ENABLE_MEM_EFFICIENT_OFFLOAD": str(True).lower(),
        "CUDA_MODULE_LOADING": config.module_loading,
        "CUDNN_BENCHMARK": str(config.benchmark).lower(),
        "TF32_AVAILABLE": str(config.enable_tf32).lower()
    })
    
    logger.info("Variables d'environnement CUDA configurées")

def optimize_torch_settings():
    """
    Optimise les paramètres PyTorch pour les performances maximales.
    """
    if not CUDA_AVAILABLE:
        return
        
    try:
        config = get_cuda_config()
        
        # Activation de cuDNN
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = config.benchmark
        torch.backends.cudnn.deterministic = config.deterministic
        
        # Configuration d'opérations tensorielles optimisées
        if config.enable_tf32 and hasattr(torch.backends.cuda, "matmul"):
            # Disponible à partir de PyTorch 1.7
            torch.backends.cuda.matmul.allow_tf32 = config.enable_tf32
            
        if config.allow_fp16 and hasattr(torch.backends.cuda, "matmul"):
            # Disponible à partir de PyTorch 1.12
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = config.allow_fp16
            
        # Configuration des memory formats optimisés
        if config.device_type == DeviceType.CUDA:
            # Optimisation des formats de mémoire pour les convolutions
            torch.backends.cuda.preferred_memory_format = torch.channels_last
        
        logger.info("Paramètres PyTorch optimisés pour les performances")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'optimisation des paramètres PyTorch: {e}")