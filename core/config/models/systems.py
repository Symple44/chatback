# core/config/models/systems.py
import os
from typing import Dict, Any    

# Configuration système globale basée sur les valeurs de config.py
SYSTEM_CONFIG = {
    "memory_management": {
        "gpu_memory_fraction": float(os.environ.get("CUDA_MEMORY_FRACTION", "0.95")),
        "cpu_memory_limit": os.environ.get("CPU_MEMORY_LIMIT", "24GB"),
        "offload_folder": os.environ.get("OFFLOAD_FOLDER", "offload_folder"),
        "cleanup_interval": 300
    },
    "optimization": {
        "enable_amp": True,
        "enable_cuda_graphs": True,
        "enable_channels_last": True,
        "compile_mode": "reduce-overhead"
    }
}