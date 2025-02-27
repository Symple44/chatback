# core/config/hardware/profiles.py
from typing import Dict, Any

# Configuration spécifique pour AMD 8845HS (16 cœurs) + RTX 3090 (24GB VRAM)
AMD_RTX3090_CONFIG = {
    "gpu": {
        "vram_total": "24GB",
        "vram_allocation": {
            "model": "20GB",       # Modèle principal
            "embeddings": "2GB",    # Modèles d'embedding
            "cache": "1GB",         # Cache GPU
            "reserve": "1GB"        # Marge de sécurité
        },
        "cuda_config": {
            "max_split_size_mb": 4096,
            "memory_efficient_attention": True,
            "flash_attention": True,
            "use_cuda_graphs": True
        }
    },
    "cpu": {
        "ram_total": "25GB",
        "ram_allocation": {
            "application": "16GB",
            "model_offload": "6GB",
            "reserve": "3GB"
        },
        "thread_config": {
            "workers": 16,           # 100% des cœurs physiques
            "inference_threads": 12,  # Pour les inférences
            "io_threads": 4           # Pour les opérations I/O
        }
    },
    "search_optimization": {
        "batch_sizes": {
            "rag": {
                "embedding_batch": 64,
                "search_batch": 32
            },
            "hybrid": {
                "embedding_batch": 48,
                "search_batch": 24,
                "rerank_batch": 16
            },
            "semantic": {
                "inference_batch": 32,
                "concept_batch": 16
            }
        },
        "concurrent_searches": 8,
        "cache_size_mb": 2048,
        "prefetch_size": 8
    }
}

# Configurations basées sur les modèles de GPU
GPU_CONFIGS = {
    "RTX3090": AMD_RTX3090_CONFIG,
    "RTX3080": {
        "gpu": {
            "vram_total": "10GB",
            "vram_allocation": {
                "model": "8GB", 
                "embeddings": "1GB",
                "cache": "0.5GB",
                "reserve": "0.5GB" 
            }
        },
        "cpu": AMD_RTX3090_CONFIG["cpu"],  # Même config CPU
        "search_optimization": {
            "batch_sizes": {
                "rag": {
                    "embedding_batch": 32, 
                    "search_batch": 16
                },
                "hybrid": {
                    "embedding_batch": 24,
                    "search_batch": 12,
                    "rerank_batch": 8 
                },
                "semantic": {
                    "inference_batch": 16,
                    "concept_batch": 8 
                }
            },
            "concurrent_searches": 4,
            "cache_size_mb": 1024,
            "prefetch_size": 4
        }
    }
}

# Configuration par défaut (fallback)
DEFAULT_HARDWARE_CONFIG = {
    "gpu": {
        "vram_total": "8GB",
        "vram_allocation": {
            "model": "6GB",
            "embeddings": "1GB",
            "cache": "512MB",
            "reserve": "512MB"
        }
    },
    "cpu": {
        "ram_total": "16GB",
        "ram_allocation": {
            "application": "10GB",
            "model_offload": "4GB",
            "reserve": "2GB"
        },
        "thread_config": {
            "workers": 4,
            "inference_threads": 2,
            "io_threads": 2
        }
    },
    "search_optimization": {
        "batch_sizes": {
            "rag": {
                "embedding_batch": 16,
                "search_batch": 8
            },
            "hybrid": {
                "embedding_batch": 12,
                "search_batch": 6,
                "rerank_batch": 4
            },
            "semantic": {
                "inference_batch": 8,
                "concept_batch": 4
            }
        },
        "concurrent_searches": 2,
        "cache_size_mb": 1024,
        "prefetch_size": 4
    }
}

class HardwareProfiles:
    """Gestionnaire de profils matériels."""
    
    @staticmethod
    def get_profile(hardware_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Détermine le profil matériel approprié basé sur la détection.
        
        Args:
            hardware_info: Informations matérielles détectées
            
        Returns:
            Profil matériel approprié
        """
        gpu_name = hardware_info["gpu"]["name"]
        cpu_model = hardware_info["cpu"]["model"]
        
        # Vérification environnement d'abord
        env_gpu = os.environ.get("GPU_NAME")
        env_cpu = os.environ.get("CPU_MODEL")
        
        if env_gpu and "3090" in env_gpu:
            return GPU_CONFIGS["RTX3090"]
        elif env_gpu and "3080" in env_gpu:
            return GPU_CONFIGS["RTX3080"]
        
        # Détection automatique 
        if "3090" in gpu_name:
            return GPU_CONFIGS["RTX3090"]
        elif "3080" in gpu_name:
            return GPU_CONFIGS["RTX3080"]
        elif "AMD" in cpu_model and "8845HS" in cpu_model:
            return GPU_CONFIGS["RTX3090"]
        
        # Profil par défaut
        return DEFAULT_HARDWARE_CONFIG