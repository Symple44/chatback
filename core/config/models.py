# core/config/models.py
from typing import Dict, List, Any
from pydantic import BaseModel
import torch

# Configuration des modèles disponibles avec paramètres optimisés pour RTX 3090 24GB
# Configuration des modèles d'embedding disponibles
EMBEDDING_MODELS = {
    "paraphrase-multilingual-mpnet-base-v2": {
        "display_name": "paraphrase multilingual mpnet base v2",
        "path": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "type": "embedding",
        "languages": ["fr"],
        "embedding_dimension": 768,
        "max_sequence_length": 512,
        "gpu_requirements": {
            "vram_required": "2GB",
            "recommended_batch_size": 32
        },
        "quantization": None,
        "capabilities": ["embedding", "retrieval"],
        "load_params": {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "max_memory": {
                "0": "2GiB",
                "cpu": "4GB"
            }
        }
    },
    "multilingual-e5-large": {
        "display_name": "E5 Large v2",
        "path": "intfloat/multilingual-e5-large",
        "type": "embedding",
        "languages": ["fr", "en", "de", "es", "it"],
        "embedding_dimension": 1024,
        "max_sequence_length": 512,
        "gpu_requirements": {
            "vram_required": "2GB",
            "recommended_batch_size": 32
        },
        "quantization": None,
        "capabilities": ["embedding", "retrieval"],
        "load_params": {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "max_memory": {
                "0": "2GiB",
                "cpu": "4GB"
            }
        }
    },
    "bge-large-v1.5": {
        "display_name": "BGE Large v1.5",
        "path": "BAAI/bge-large-v1.5",
        "type": "embedding",
        "languages": ["fr", "en", "de", "es", "it", "zh"],
        "embedding_dimension": 1024,
        "max_sequence_length": 512,
        "gpu_requirements": {
            "vram_required": "2GB",
            "recommended_batch_size": 32
        },
        "quantization": None,
        "capabilities": ["embedding", "retrieval"],
        "load_params": {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "max_memory": {
                "0": "2GiB",
                "cpu": "4GB"
            }
        }
    }
}

# Configuration des modèles de summarization disponibles
SUMMARIZER_MODELS = {
    "t5-base-fr-sum-cnndm": {
        "display_name": "t5-base-fr-sum-cnndm",
        "path": "plguillou/t5-base-fr-sum-cnndm",
        "type": "summarization",
        "languages": ["fr"],
        "max_sequence_length": 1024,
        "gpu_requirements": {
            "vram_required": "4GB",
            "recommended_batch_size": 16
        },
        "quantization": "bitsandbytes-4bit",
        "capabilities": ["summarization"],
        "load_params": {
            "load_in_4bit": True,
            "device_map": "auto",
            "max_memory": {
                "0": "4GiB",
                "cpu": "8GB"
            }
        }
    },
    "mT5_multilingual_XLSum": {
        "display_name": "mT5 Base Multilingual Summarizer",
        "path": "csebuetnlp/mT5_multilingual_XLSum",
        "type": "summarization",
        "languages": ["fr", "en", "de", "es", "it"],
        "max_sequence_length": 1024,
        "gpu_requirements": {
            "vram_required": "4GB",
            "recommended_batch_size": 16
        },
        "quantization": "bitsandbytes-4bit",
        "capabilities": ["summarization"],
        "load_params": {
            "load_in_4bit": True,
            "device_map": "auto",
            "max_memory": {
                "0": "4GiB",
                "cpu": "8GB"
            }
        }
    },
    "mbart-large-cc25": {
        "display_name": "mBART Large Multilingual",
        "path": "facebook/mbart-large-cc25",
        "type": "summarization",
        "languages": ["fr", "en", "de", "es", "it"],
        "max_sequence_length": 1024,
        "gpu_requirements": {
            "vram_required": "4GB",
            "recommended_batch_size": 16
        },
        "quantization": "bitsandbytes-4bit",
        "capabilities": ["summarization"],
        "load_params": {
            "load_in_4bit": True,
            "device_map": "auto",
            "max_memory": {
                "0": "4GiB",
                "cpu": "8GB"
            }
        }
    }
}

# Configuration des modèles de chat/instruction disponibles
AVAILABLE_MODELS = {
    "Mistral-7B-Instruct-v0.3": {
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
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4",
            "max_memory": {
                "0": "22GiB",
                "cpu": "24GB"
            }
        }
    },
    "Mixtral-8x7B-Instruct-v0.1": {
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
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4",
            "max_memory": {
                "0": "22GiB",
                "cpu": "24GB"
            }
        }
    },
    "Qwen-14B-Chat": {
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
            "bnb_4bit_compute_dtype": torch.float16,
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
            "bnb_4bit_compute_dtype": torch.float16,
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

# Configuration de performance pour les modèles d'embedding
EMBEDDING_PERFORMANCE_CONFIGS = {
    "paraphrase-multilingual-mpnet-base-v2": {
        "batch_size": 32,
        "normalize_embeddings": True,
        "max_length": 512,
        "pooling_strategy": "mean",
        "preprocessing": {
            "truncation": True,
            "padding": "max_length",
            "add_special_tokens": True
        }
    },
    "e5-large-v2": {
        "batch_size": 32,
        "normalize_embeddings": True,
        "max_length": 512,
        "pooling_strategy": "mean",
        "preprocessing": {
            "truncation": True,
            "padding": "max_length",
            "add_special_tokens": True
        }
    },
    "bge-large": {
        "batch_size": 32,
        "normalize_embeddings": True,
        "max_length": 512,
        "pooling_strategy": "cls",
        "preprocessing": {
            "truncation": True,
            "padding": "max_length",
            "add_special_tokens": True
        }
    }
}

# Configuration de performance pour les modèles de summarization
SUMMARIZER_PERFORMANCE_CONFIGS = {
    "t5-base-fr-sum-cnndm": {
        "batch_size": 16,
        "min_length": 50,
        "max_length": 150,
        "length_penalty": 2.0,
        "early_stopping": True,
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "preprocessing": {
            "truncation": True,
            "padding": "longest",
            "max_length": 1024
        }
    },
    "mt5-base-multi-sum": {
        "batch_size": 16,
        "min_length": 50,
        "max_length": 150,
        "length_penalty": 2.0,
        "early_stopping": True,
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "preprocessing": {
            "truncation": True,
            "padding": "longest",
            "max_length": 1024
        }
    },
    "bart-large-multi": {
        "batch_size": 16,
        "min_length": 50,
        "max_length": 150,
        "length_penalty": 2.0,
        "early_stopping": True,
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "preprocessing": {
            "truncation": True,
            "padding": "longest",
            "max_length": 1024
        }
    }
}

# Configuration globale du système
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