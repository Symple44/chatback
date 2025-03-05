# core/llm/cuda_manager.py
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List
import gc
import torch
import torch.cuda
import torch.backends.cudnn as cudnn
import os
import json
import psutil
from enum import Enum

from core.config.config import settings
from core.config.models import SYSTEM_CONFIG
from core.utils.logger import get_logger

logger = get_logger("cuda_manager")

class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"

class ModelPriority(Enum):
    HIGH = "high"      # Modèles principaux de chat
    MEDIUM = "medium"  # Modèles de summarization
    LOW = "low"       # Modèles d'embedding

@dataclass
class CUDAConfig:
    """Configuration CUDA complète."""
    device_type: DeviceType = DeviceType.AUTO
    device_id: int = 0
    memory_configs: Dict[ModelPriority, Dict[str, str]] = None
    compute_dtype: torch.dtype = torch.float16
    module_loading: str = "LAZY"
    max_split_size_mb: int = 4096
    gc_threshold: float = 0.9
    enable_tf32: bool = True
    allow_fp16: bool = True
    deterministic: bool = False
    benchmark: bool = True
    use_flash_attention: bool = True

    def __post_init__(self):
        if self.device_type == DeviceType.AUTO:
            self.device_type = DeviceType.CUDA if torch.cuda.is_available() else DeviceType.CPU
        
        # Configuration mémoire par défaut si non spécifiée
        if not self.memory_configs:
            self.memory_configs = {
                ModelPriority.HIGH: {"0": "16GiB", "cpu": "12GB"},    # Pour modèles de chat
                ModelPriority.MEDIUM: {"0": "4GiB", "cpu": "8GB"},   # Pour summarizers
                ModelPriority.LOW: {"0": "2GiB", "cpu": "4GB"}       # Pour embeddings
            }

class CUDAManager:
    def __init__(self):
        """Initialise le gestionnaire CUDA."""
        self.config = self._load_config()
        self.device = None
        self._initialized = False
        self.memory_stats = {}
        self.current_allocations = {p: 0 for p in ModelPriority}

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
            
            # Configuration de la mémoire
            for priority in ModelPriority:
                self.current_allocations[priority] = 0

            return torch.device(f"cuda:{self.config.device_id}")
            
        except Exception as e:
            logger.error(f"Erreur setup device CUDA: {e}")
            return torch.device("cpu")
    
    def _load_config(self) -> CUDAConfig:
        """Charge et valide la configuration CUDA."""
        try:
            
            # Accès direct à la configuration hardware à travers settings
            hardware_config = settings.hardware
            cuda_config = hardware_config.cuda
                
            # Conversion des configurations mémoire pour les priorités de modèles
            memory_configs = {
                ModelPriority.HIGH: {"0": "16GiB", "cpu": "12GB"},
                ModelPriority.MEDIUM: {"0": "4GiB", "cpu": "8GB"},
                ModelPriority.LOW: {"0": "2GiB", "cpu": "4GB"}
            }

            return CUDAConfig(
                device_type=cuda_config.device_type,
                device_id=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
                memory_configs=memory_configs,
                compute_dtype=cuda_config.compute_dtype,
                module_loading=cuda_config.module_loading,
                max_split_size_mb=cuda_config.max_split_size_mb,
                gc_threshold=cuda_config.gc_threshold,
                enable_tf32=cuda_config.enable_tf32,
                allow_fp16=cuda_config.allow_fp16,
                deterministic=cuda_config.deterministic,
                benchmark=cuda_config.benchmark,
                use_flash_attention=cuda_config.use_flash_attention
            )
            
        except Exception as e:
            logger.error(f"Erreur chargement config CUDA: {e}")
            # Fallback vers une configuration par défaut en cas d'erreur
            return CUDAConfig()

    async def initialize(self):
        """Initialise l'environnement CUDA."""
        if self._initialized:
            return

        try:
            # Application de la configuration système
            self._apply_system_config()
            
            # Nettoyage initial
            self.cleanup_memory()
            
            # Configuration du device
            self.device = self._setup_device()

            # Configuration CUDA
            if self.config.device_type == DeviceType.CUDA:
                # Configuration cuDNN
                cudnn.enabled = True
                cudnn.benchmark = self.config.benchmark and SYSTEM_CONFIG["optimization"]["enable_cuda_graphs"]
                cudnn.deterministic = self.config.deterministic
                cudnn.allow_tf32 = self.config.enable_tf32

                # Configuration des opérations tensorielles
                torch.backends.cuda.matmul.allow_tf32 = self.config.enable_tf32
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = self.config.allow_fp16

                # Configuration des optimisations mémoire
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                    f"max_split_size_mb:{self.config.max_split_size_mb},"
                    f"garbage_collection_threshold:{self.config.gc_threshold}"
                )

                # Application des optimisations système
                if SYSTEM_CONFIG["optimization"]["enable_channels_last"]:
                    torch.backends.cuda.preferred_memory_format = torch.channels_last

                if SYSTEM_CONFIG["optimization"]["compile_mode"] == "reduce-overhead":
                    torch._dynamo.config.optimize_ddp = True
                    torch._dynamo.config.automatic_dynamic_shapes = True

            # Configuration des threads CPU selon SYSTEM_CONFIG
            thread_config = settings.hardware.thread_config
            for env_var, thread_count in thread_config.items():
                os.environ[env_var] = str(thread_count)
            torch.set_num_threads(settings.hardware.thread_config.get("mkl_num_threads", 16))

            self._initialized = True
            self.update_memory_stats()
            self._log_system_info()
            
        except Exception as e:
            logger.error(f"Erreur initialisation CUDA: {e}")
            raise

    def _apply_system_config(self):
        """Applique la configuration système globale."""
        try:
            # Configuration de la mémoire
            memory_config = SYSTEM_CONFIG["memory_management"]
            self.config.gc_threshold = memory_config["gpu_memory_fraction"]
            
            # Mise à jour des limites mémoire
            if hasattr(self.config, "memory_configs"):
                for priority in self.config.memory_configs:
                    if "0" in self.config.memory_configs[priority]:
                        max_gpu_mem = int(float(memory_config["gpu_memory_fraction"]) * 24)  # 24GB pour RTX 3090
                        self.config.memory_configs[priority][0] = f"{max_gpu_mem}GiB"
                    
                    if "cpu" in self.config.memory_configs[priority]:
                        self.config.memory_configs[priority]["cpu"] = memory_config["cpu_memory_limit"]

            # Configuration des dossiers
            self.config.offload_folder = memory_config["offload_folder"]
            os.makedirs(self.config.offload_folder, exist_ok=True)

            # Application des optimisations
            if SYSTEM_CONFIG["optimization"]["enable_amp"]:
                torch.amp.autocast(device_type='cuda', enabled=True)

            logger.info("Configuration système appliquée avec succès")
            
        except Exception as e:
            logger.error(f"Erreur application configuration système: {e}")
            raise

    def _check_memory_availability(self, priority: ModelPriority) -> bool:
        """Vérifie si assez de VRAM est disponible pour la priorité donnée."""
        if not torch.cuda.is_available():
            return False

        try:
            required_memory = {
                ModelPriority.HIGH: 16,   # 16GB pour les modèles principaux
                ModelPriority.MEDIUM: 4,   # 4GB pour les summarizers
                ModelPriority.LOW: 2       # 2GB pour les embeddings
            }[priority]

            total_allocated = sum(self.current_allocations.values())
            available_memory = (24 - total_allocated)  # 24GB total pour RTX 3090
            
            return available_memory >= required_memory

        except Exception as e:
            logger.error(f"Erreur vérification mémoire: {e}")
            return False

    def allocate_memory(self, priority: ModelPriority) -> bool:
        """Alloue de la mémoire pour un type de modèle."""
        try:
            if not self._check_memory_availability(priority):
                return False

            memory_sizes = {
                ModelPriority.HIGH: 16,
                ModelPriority.MEDIUM: 4,
                ModelPriority.LOW: 2
            }
            
            self.current_allocations[priority] += memory_sizes[priority]
            return True

        except Exception as e:
            logger.error(f"Erreur allocation mémoire: {e}")
            return False

    def release_memory(self, priority: ModelPriority):
        """Libère la mémoire allouée pour un type de modèle."""
        try:
            memory_sizes = {
                ModelPriority.HIGH: 16,
                ModelPriority.MEDIUM: 4,
                ModelPriority.LOW: 2
            }
            
            self.current_allocations[priority] = max(
                0,
                self.current_allocations[priority] - memory_sizes[priority]
            )
            
            self.cleanup_memory()

        except Exception as e:
            logger.error(f"Erreur libération mémoire: {e}")
    
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

    def cleanup_memory(self):
        """Nettoie la mémoire CUDA."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gc.collect()  # Forcer la collection des objets Python
                torch.cuda.synchronize()
            except Exception as e:
                logger.error(f"Erreur nettoyage mémoire CUDA: {e}")

    async def cleanup(self):
        """Nettoie proprement toutes les ressources."""
        try:
            logger.info("Début du nettoyage des ressources CUDA...")
            
            # 1. Réinitialiser d'abord les allocations
            for priority in ModelPriority:
                self.current_allocations[priority] = 0
            
            # 2. Attendre que toutes les opérations en cours soient terminées
            if torch.cuda.is_available():
                try:
                    # Synchroniser pour s'assurer que toutes les opérations sont terminées
                    torch.cuda.synchronize()
                    
                    # Libérer la mémoire avec une protection contre les erreurs
                    try:
                        torch.cuda.empty_cache()
                    except Exception as cuda_error:
                        logger.warning(f"Avertissement lors du nettoyage du cache CUDA: {cuda_error}")
                    
                    # Collecter les objets Python
                    gc.collect()
                    
                    # Attendre à nouveau pour s'assurer que tout est terminé
                    try:
                        torch.cuda.synchronize()
                    except Exception as sync_error:
                        logger.warning(f"Avertissement lors de la synchronisation CUDA finale: {sync_error}")
                    
                    logger.info(f"Mémoire CUDA libérée: {torch.cuda.memory_allocated()/1024**2:.2f}MB allouée")
                except Exception as cuda_error:
                    logger.warning(f"Avertissement nettoyage CUDA: {cuda_error}")
            
            # 3. Réinitialiser l'état
            self._initialized = False
            self.memory_stats.clear()
            
            logger.info("Ressources CUDA nettoyées avec succès")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage ressources CUDA: {e}")
            # Ne pas propager l'exception pour éviter d'interrompre le nettoyage des autres composants
            
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
                    "cuda_version": torch.version.cuda,
                    "allocated_memory": self.current_allocations,
                    "system_config": {
                        "memory_management": SYSTEM_CONFIG["memory_management"],
                        "optimization": SYSTEM_CONFIG["optimization"],
                        "thread_config": SYSTEM_CONFIG["thread_config"]
                    }
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