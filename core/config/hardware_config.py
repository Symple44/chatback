# core/config/hardware_config.py
from typing import Dict, Any, Optional, Tuple
import os
import logging
import platform
import psutil
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Tentative d'importation de torch avec gestion d'erreur
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    logger.warning("PyTorch n'est pas disponible, la détection GPU est désactivée")

class HardwareDetector:
    """Détecteur de matériel pour la configuration automatique."""

    @staticmethod
    def detect_cpu() -> Dict[str, Any]:
        """Détecte les caractéristiques du CPU."""
        cpu_info = {
            "model": os.environ.get("CPU_MODEL", "Indéterminé"),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "architecture": platform.machine(),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        }
        
        # Tentative de détection du modèle si non spécifié
        if cpu_info["model"] == "Indéterminé":
            try:
                if platform.system() == "Linux":
                    with open("/proc/cpuinfo", "r") as f:
                        for line in f:
                            if "model name" in line:
                                cpu_info["model"] = line.split(":")[1].strip()
                                break
                elif platform.system() == "Darwin":  # macOS
                    import subprocess
                    cpu_info["model"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).strip().decode()
                elif platform.system() == "Windows":
                    import subprocess
                    cpu_info["model"] = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().split("\n")[1].strip()
            except Exception as e:
                logger.warning(f"Impossible de détecter le modèle CPU: {e}")
        
        return cpu_info

    @staticmethod
    def detect_gpu() -> Dict[str, Any]:
        """Détecte les caractéristiques du GPU."""
        gpu_info = {
            "available": CUDA_AVAILABLE,
            "name": os.environ.get("GPU_NAME", "Indéterminé"),
            "vram_gb": 0,
            "compute_capability": "0.0"
        }

        if CUDA_AVAILABLE:
            try:
                # Obtenir les infos du premier GPU (index 0)
                device = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(device)
                
                gpu_info.update({
                    "name": properties.name,
                    "vram_gb": round(properties.total_memory / (1024**3), 2),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "multi_processor_count": properties.multi_processor_count,
                    "cuda_version": torch.version.cuda
                })
            except Exception as e:
                logger.warning(f"Erreur lors de la détection GPU: {e}")
                
        return gpu_info

    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """Détecte l'ensemble du matériel disponible."""
        cpu_info = HardwareDetector.detect_cpu()
        gpu_info = HardwareDetector.detect_gpu()
        
        # Création de l'objet hardware complet
        hardware = {
            "cpu": cpu_info,
            "gpu": gpu_info,
            "system": {
                "platform": platform.system(),
                "release": platform.release(),
                "version": platform.version()
            }
        }
        
        return hardware

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
        },
        "search_config": AMD_RTX3090_CONFIG["search_config"]
    }
}

def get_hardware_config() -> Dict[str, Any]:
    """
    Détermine la configuration matérielle optimale basée sur la détection.
    
    Returns:
        Dict[str, Any]: Configuration matérielle optimisée pour les performances
    """
    # Détection du matériel
    hardware = HardwareDetector.detect_hardware()
    gpu_name = hardware["gpu"]["name"]
    cpu_model = hardware["cpu"]["model"]
    
    # Journalisation des informations matérielles détectées
    logger.info(f"Matériel détecté: CPU={cpu_model}, GPU={gpu_name}")
    
    # Configuration via variables d'environnement (priorité maximale)
    env_gpu = os.environ.get("GPU_NAME")
    env_cpu = os.environ.get("CPU_MODEL")
    
    # Tentative de correspondance avec une configuration optimisée
    if env_gpu and "3090" in env_gpu:
        logger.info("Configuration RTX 3090 sélectionnée via variable d'environnement")
        return GPU_CONFIGS["RTX3090"]
    elif env_gpu and "3080" in env_gpu:
        logger.info("Configuration RTX 3080 sélectionnée via variable d'environnement")
        return GPU_CONFIGS["RTX3080"]
    
    # Détection automatique 
    if "3090" in gpu_name:
        logger.info("Configuration RTX 3090 détectée automatiquement")
        return GPU_CONFIGS["RTX3090"]
    elif "3080" in gpu_name:
        logger.info("Configuration RTX 3080 détectée automatiquement")
        return GPU_CONFIGS["RTX3080"]
    elif "AMD" in cpu_model and "8845HS" in cpu_model:
        logger.info("Configuration AMD 8845HS détectée, utilisation RTX 3090")
        return GPU_CONFIGS["RTX3090"]
    
    # Configuration par défaut si aucune correspondance
    logger.info("Utilisation de la configuration matérielle par défaut")
    return DEFAULT_HARDWARE_CONFIG

def get_optimal_thread_config() -> Dict[str, int]:
    """
    Détermine la configuration optimale des threads basée sur le CPU détecté.
    
    Returns:
        Dict[str, int]: Configuration optimale des threads
    """
    cpu_info = HardwareDetector.detect_cpu()
    physical_cores = cpu_info["physical_cores"]
    logical_cores = cpu_info["logical_cores"]
    
    # Configuration basée sur le nombre de cœurs disponibles
    if physical_cores >= 16:
        return {
            "workers": physical_cores,
            "inference_threads": min(16, physical_cores),
            "io_threads": min(8, physical_cores // 2),
            "mkl_threads": min(16, physical_cores),
            "omp_threads": min(16, physical_cores)
        }
    elif physical_cores >= 8:
        return {
            "workers": physical_cores,
            "inference_threads": min(8, physical_cores),
            "io_threads": min(4, physical_cores // 2),
            "mkl_threads": min(8, physical_cores),
            "omp_threads": min(8, physical_cores)
        }
    else:
        return {
            "workers": max(2, physical_cores),
            "inference_threads": max(2, physical_cores),
            "io_threads": 2,
            "mkl_threads": max(4, physical_cores),
            "omp_threads": max(4, physical_cores)
        }

def export_hardware_info(output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Exporte les informations matérielles détectées dans un fichier JSON.
    
    Args:
        output_path: Chemin où sauvegarder le fichier JSON (optionnel)
        
    Returns:
        Dict[str, Any]: Informations matérielles détectées
    """
    hardware = HardwareDetector.detect_hardware()
    config = get_hardware_config()
    
    result = {
        "hardware_detected": hardware,
        "config_applied": config,
        "thread_config": get_optimal_thread_config()
    }
    
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Informations matérielles exportées dans {output_path}")
    
    return result