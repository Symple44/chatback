# core/config/hardware_config.py
from typing import Dict, Any

# Configuration spécifique RTX 3090 (24GB VRAM)
RTX3090_CONFIG = {
    "gpu": {
        "vram_total": "24GB",
        "vram_allocation": {
            "model": "20GB",      # Modèle principal
            "embeddings": "2GB",   # Modèles d'embedding
            "cache": "1GB",       # Cache GPU
            "reserve": "1GB"      # Marge de sécurité
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
            "workers": 12,          # 75% des cœurs physiques
            "inference_threads": 8,  # Pour les inférences
            "io_threads": 4         # Pour les opérations I/O
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
        "concurrent_searches": 4,
        "cache_size_mb": 2048,
        "prefetch_size": 8
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

def get_hardware_config(gpu_name: str = None) -> Dict[str, Any]:
    """Retourne la configuration matérielle appropriée."""
    if gpu_name and "3090" in gpu_name:
        return RTX3090_CONFIG
    return DEFAULT_HARDWARE_CONFIG