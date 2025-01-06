# core/config/models.py
from typing import Dict, List, Any
from pydantic import BaseModel

# Configuration des modèles disponibles avec paramètres optimisés pour RTX 3090 24GB
AVAILABLE_MODELS = {
    "mistral-7b-instruct-v0.3": {
        "display_name": "Mistral-7B Instruct v0.3",
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
        "type": "chat",
        "languages": ["fr", "en"],
        "context_length": 8192,
        "gpu_requirements": {
            "vram_required": "16GB",
            "recommended_batch_size": 32
        },
        "quantization": "bitsandbytes-4bit",
        "capabilities": ["chat", "instruction"],
        "load_params": {
            "load_in_4bit": True,
            "use_flash_attention_2": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "max_memory": {
                "0": "22GiB",
                "cpu": "24GB"
            }
        }
    },
    "mixtral-8x7b-instruct": {
        "display_name": "Mixtral 8x7B Instruct",
        "path": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "type": "chat",
        "languages": ["fr", "en", "es", "de", "it"],
        "context_length": 32768,
        "gpu_requirements": {
            "vram_required": "20GB",
            "recommended_batch_size": 16
        },
        "quantization": "bitsandbytes-4bit",
        "capabilities": ["chat", "instruction", "function_calling"],
        "load_params": {
            "load_in_4bit": True,
            "use_flash_attention_2": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "max_memory": {
                "0": "22GiB",
                "cpu": "24GB"
            }
        }
    },
    "qwen-14b-chat": {
        "display_name": "Qwen 14B Chat",
        "path": "Qwen/Qwen-14B-Chat",
        "type": "chat",
        "languages": ["fr", "en"],
        "context_length": 8192,
        "gpu_requirements": {
            "vram_required": "18GB",
            "recommended_batch_size": 24
        },
        "quantization": "bitsandbytes-4bit",
        "capabilities": ["chat", "instruction"],
        "load_params": {
            "load_in_4bit": True,
            "use_flash_attention_2": True,
            "trust_remote_code": True,
            "bnb_4bit_compute_dtype": "float16",
            "max_memory": {
                "0": "22GiB",
                "cpu": "24GB"
            }
        }
    },
    "zephyr-7b-beta": {
        "display_name": "Zephyr 7B Beta",
        "path": "HuggingFaceH4/zephyr-7b-beta",
        "type": "chat",
        "languages": ["fr", "en"],
        "context_length": 8192,
        "gpu_requirements": {
            "vram_required": "16GB",
            "recommended_batch_size": 32
        },
        "quantization": "bitsandbytes-4bit",
        "capabilities": ["chat", "instruction", "function_calling"],
        "load_params": {
            "load_in_4bit": True,
            "use_flash_attention_2": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "max_memory": {
                "0": "22GiB",
                "cpu": "24GB"
            }
        }
    }
}

# Configuration CUDA optimisée pour RTX 3090
CUDA_CONFIG = {
    "device_type": "cuda",
    "compute_capability": "8.6",
    "memory_config": {
        "max_split_size_mb": 4096,
        "gpu_memory_fraction": 0.95,
        "offload_folder": "offload_folder"
    },
    "optimization": {
        "use_flash_attention": True,
        "use_cuda_graphs": True,
        "cudnn_benchmark": True,
        "enable_tf32": True,
        "allow_fp16_reduced_precision_reduction": True
    }
}

# Configuration de performance par modèle
MODEL_PERFORMANCE_CONFIGS = {
    "mistral-7b-instruct-v0.3": {
        "batch_size": 32,
        "prefetch_factor": 2,
        "num_workers": 4,
        "pin_memory": True,
        "generation_config": {
            "max_length": 4096,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        }
    },
    "mixtral-8x7b-instruct": {
        "batch_size": 16,
        "prefetch_factor": 2,
        "num_workers": 4,
        "pin_memory": True,
        "generation_config": {
            "max_length": 8192,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        }
    },
    "qwen-14b-chat": {
        "batch_size": 24,
        "prefetch_factor": 2,
        "num_workers": 4,
        "pin_memory": True,
        "generation_config": {
            "max_length": 4096,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        }
    },
    "zephyr-7b-beta": {
        "batch_size": 32,
        "prefetch_factor": 2,
        "num_workers": 4,
        "pin_memory": True,
        "generation_config": {
            "max_length": 4096,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        }
    }
}

# Configuration du système pour tous les modèles
SYSTEM_CONFIG = {
    "memory_management": {
        "gpu_memory_fraction": 0.95,
        "cpu_memory_limit": "24GB",
        "offload_folder": "offload_folder",
        "cleanup_interval": 300
    },
    "optimization": {
        "enable_amp": True,
        "enable_cuda_graphs": True,
        "enable_channels_last": True,
        "compile_mode": "reduce-overhead"
    },
    "thread_config": {
        "mkl_num_threads": 16,
        "omp_num_threads": 16,
        "numpy_num_threads": 16
    }
}