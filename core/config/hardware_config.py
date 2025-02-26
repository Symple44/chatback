# core/config/hardware_config.py
from typing import Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)

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
    },
    # Optimisations pour core/search/search_manager.py et strategies.py
    "search_config": {
        "max_docs": 10,            # Nombre maximum de documents à retourner
        "min_score": 0.3,          # Score minimum de confiance
        "rag": {
            "vector_weight": 0.7,  # Poids pour le score vectoriel
            "semantic_weight": 0.3  # Poids pour le score sémantique
        },
        "hybrid": {
            "rerank_top_k": 10,    # Nombre de documents à reranker
            "window_size": 3       # Taille de la fenêtre glissante
        },
        "semantic": {
            "max_concepts": 5,     # Nombre maximum de concepts
            "boost_exact": True    # Booster les correspondances exactes
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
    },
    "search_config": {
        "max_docs": 5,
        "min_score": 0.3,
        "rag": {
            "vector_weight": 0.7,
            "semantic_weight": 0.3
        }
    }
}

def detect_hardware() -> Dict[str, Optional[str]]:
    """Détecte le matériel à partir des variables d'environnement ou de la détection automatique."""
    hardware_info = {
        "cpu_model": os.environ.get("CPU_MODEL", None),
        "gpu_name": os.environ.get("GPU_NAME", None)
    }
    
    # Si la détection automatique est nécessaire, la compléter ici
    # Pour l'instant, nous utilisons uniquement les variables d'environnement
    
    logger.info(f"Matériel détecté: CPU={hardware_info['cpu_model']}, GPU={hardware_info['gpu_name']}")
    return hardware_info

def get_hardware_config() -> Dict[str, Any]:
    """Retourne la configuration matérielle appropriée basée sur la détection."""
    hardware = detect_hardware()
    
    # Détection de configuration spécifique
    if (hardware["cpu_model"] and "8845HS" in hardware["cpu_model"]) or \
       (hardware["gpu_name"] and "3090" in hardware["gpu_name"]):
        logger.info("Configuration AMD 8845HS + RTX 3090 détectée")
        return AMD_RTX3090_CONFIG
    
    # Détection générique pour RTX 3090
    if hardware["gpu_name"] and "3090" in hardware["gpu_name"]:
        logger.info("Configuration RTX 3090 générique détectée")
        return AMD_RTX3090_CONFIG
    
    logger.info("Utilisation de la configuration matérielle par défaut")
    return DEFAULT_HARDWARE_CONFIG