# core/config/models.py
from typing import Dict, List, Any, Optional
import torch
import os
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Import des configurations matérielles et CUDA
from core.config.cuda_config import DeviceType, get_cuda_config

# Import de la configuration globale pour rester compatible 
from core.config.config import settings

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
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "max_memory": {0: "16GiB", "cpu": "12GB"},
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "llm_int8_enable_fp32_cpu_offload": True
            },
            "attn_implementation": "flash_attention_2"
        },
        "generation_config": {
            "preprocessing": {
                "max_input_length": 6144,
                "max_output_length": 1024,
                "max_prompt_length": 4096,
                "max_context_length": 8192,
                "truncation_strategy": "left"
            },
            "generation_params": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "do_sample": True,
                "num_beams": 4,
                "min_new_tokens": 32,
                "max_new_tokens": 1024,
                "length_penalty": 1.0,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            },
            "response_types": {
                "comprehensive": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_new_tokens": 1024,
                    "style": "Je vais vous donner une explication détaillée."
                },
                "concise": {
                    "temperature": 0.5,
                    "top_p": 0.90,
                    "max_new_tokens": 256,
                    "style": "Je vais être bref et précis."
                },
                "technical": {
                    "temperature": 0.3,
                    "top_p": 0.85,
                    "max_new_tokens": 2048,
                    "style": "Je vais fournir des détails techniques."
                },
                "clarification": {
                    "temperature": 0.4,
                    "top_p": 0.90,
                    "max_new_tokens": 128,
                    "style": "Je vais demander des précisions."
                }
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
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "max_memory": {0: "20GiB", "cpu": "24GB"},
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "llm_int8_enable_fp32_cpu_offload": True
            },
            "low_cpu_mem_usage": True,
            "offload_folder": "offload_folder",
            "attn_implementation": "flash_attention_2"
        },
        "generation_config": {
            "preprocessing": {
                "max_input_length": 24576,
                "max_output_length": 4096,
                "max_prompt_length": 8192,
                "max_context_length": 32768,
                "truncation_strategy": "left"
            },
            "generation_params": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "do_sample": True,
                "num_beams": 4,
                "min_new_tokens": 32,
                "max_new_tokens": 4096,
                "length_penalty": 1.0,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            },
            "response_types": {
                "comprehensive": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_new_tokens": 4096,
                    "style": "Je vais vous donner une explication détaillée."
                },
                "concise": {
                    "temperature": 0.5,
                    "top_p": 0.90,
                    "max_new_tokens": 512,
                    "style": "Je vais être bref et précis."
                },
                "technical": {
                    "temperature": 0.3,
                    "top_p": 0.85,
                    "max_new_tokens": 8192,
                    "style": "Je vais fournir des détails techniques."
                },
                "clarification": {
                    "temperature": 0.4,
                    "top_p": 0.90,
                    "max_new_tokens": 256,
                    "style": "Je vais demander des précisions."
                }
            }
        }
    },
    "Mistral-Small-24B-Instruct-2501": {
        "display_name": "Mistral Small 24B Instruct",
        "path": "mistralai/Mistral-Small-24B-Instruct-2501",
        "type": "chat",
        "languages": ["fr", "en", "es", "de", "it"],
        "context_length": 32768,
        "gpu_requirements": {
            "vram_required": "22GB",
            "recommended_batch_size": 4
        },
        "quantization": "bitsandbytes-4bit",
        "capabilities": ["chat", "instruction", "function_calling"],
        "load_params": {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "max_memory": {0: "22GiB", "cpu": "22GB"},
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_use_nested_quant": True,
                "llm_int8_enable_fp32_cpu_offload": True
            },
            "low_cpu_mem_usage": True,
            "offload_folder": "offload_folder",
            "attn_implementation": "flash_attention_2"
        },
        "generation_config": {
            "preprocessing": {
                "max_input_length": 24576,
                "max_output_length": 4096,
                "max_prompt_length": 8192,
                "max_context_length": 32768,
                "truncation_strategy": "left"
            },
            "generation_params": {
                "temperature": 0.15,  # Plus bas car modèle plus grand
                "top_p": 0.95,
                "top_k": 50,
                "do_sample": True,
                "num_beams": 2,
                "min_new_tokens": 32,
                "max_new_tokens": 4096,
                "length_penalty": 1.0,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            },
            "response_types": {
                "comprehensive": {
                    "temperature": 0.15,
                    "top_p": 0.95,
                    "max_new_tokens": 4096,
                    "style": "Je vais vous donner une explication détaillée."
                },
                "concise": {
                    "temperature": 0.1,
                    "top_p": 0.90,
                    "max_new_tokens": 512,
                    "style": "Je vais être bref et précis."
                },
                "technical": {
                    "temperature": 0.05,
                    "top_p": 0.85,
                    "max_new_tokens": 8192,
                    "style": "Je vais fournir des détails techniques."
                },
                "clarification": {
                    "temperature": 0.1,
                    "top_p": 0.90,
                    "max_new_tokens": 256,
                    "style": "Je vais demander des précisions."
                }
            }
        }
    }
}

MODEL_PERFORMANCE_CONFIGS = {
    "Mistral-7B-Instruct-v0.3": {
        "batch_size": 32,
        "prefetch_factor": 2,
        "num_workers": 4,
        "pin_memory": True,
        "dataloader_params": {
            "persistent_workers": True,
            "shuffle_buffer_size": 10000
        },
        "execution": {
            "forward_batch_size": 32,
            "mixed_precision": True,
            "gradient_checkpointing": False
        },
        "optimization": {
            "weight_tying": True,
            "memory_efficient_attention": True,
            "use_kernel_injection": True
        }
    },
    "Mixtral-8x7B-Instruct-v0.1": {
        "batch_size": 16,
        "prefetch_factor": 2,
        "num_workers": 4,
        "pin_memory": True,
        "dataloader_params": {
            "persistent_workers": True,
            "shuffle_buffer_size": 10000
        },
        "execution": {
            "forward_batch_size": 16,
            "mixed_precision": True,
            "gradient_checkpointing": True
        },
        "optimization": {
            "weight_tying": True,
            "memory_efficient_attention": True,
            "use_kernel_injection": True
        }
    },
    "Mistral-Small-24B-Instruct-2501": {
        "batch_size": 4,
        "prefetch_factor": 2,
        "num_workers": 4,
        "pin_memory": True,
        "dataloader_params": {
            "persistent_workers": True,
            "shuffle_buffer_size": 10000
        },
        "execution": {
            "forward_batch_size": 4,
            "mixed_precision": True,
            "gradient_checkpointing": True
        },
        "optimization": {
            "weight_tying": True,
            "memory_efficient_attention": True,
            "use_kernel_injection": True,
            "enable_cuda_graph": True,
            "enable_flash_attention": True,
            "enable_mem_efficient_cross_attention": True,
            "enable_jit_fuser": True
        }
    }
}

# Configuration des modèles d'embedding
EMBEDDING_MODELS = {
    "paraphrase-multilingual-mpnet-base-v2": {
        "display_name": "Paraphrase Multilingual MPNet Base v2",
        "path": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "type": "embedding",
        "languages": ["fr", "en", "de", "es", "it"],
        "embedding_dimension": 768,
        "max_sequence_length": 512,
        "gpu_requirements": {
            "vram_required": "2GB",
            "recommended_batch_size": 32
        },
        "load_params": {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "max_memory": {0: "2GiB", "cpu": "4GB"}
        },
        "generation_config": {
            "preprocessing": {
                "max_input_length": 512,
                "truncation_strategy": "right",
                "add_special_tokens": True,
                "padding": "max_length"
            },
            "embedding_params": {
                "normalize_embeddings": True,
                "pooling_strategy": "mean",
                "batch_size": 32
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
        "load_params": {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "max_memory": {0: "2GiB", "cpu": "4GB"}
        },
        "generation_config": {
            "preprocessing": {
                "max_input_length": 512,
                "truncation_strategy": "right",
                "add_special_tokens": True,
                "padding": "max_length"
            },
            "embedding_params": {
                "normalize_embeddings": True,
                "pooling_strategy": "cls",
                "batch_size": 32
            }
        }
    }
}

# Configuration des modèles de résumé
SUMMARIZER_MODELS = {
    "t5-base-fr-sum-cnndm": {
        "display_name": "T5 Base FR Summarizer",
        "path": "plguillou/t5-base-fr-sum-cnndm",
        "type": "summarization",
        "languages": ["fr"],
        "max_sequence_length": 1024,
        "gpu_requirements": {
            "vram_required": "4GB",
            "recommended_batch_size": 16
        },
        "load_params": {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "max_memory": {0: "4GiB", "cpu": "8GB"},
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "quant_method": "bitsandbytes"
            }
        },
        "generation_config": {
            "preprocessing": {
                "max_input_length": 1024,
                "max_output_length": 150,
                "truncation_strategy": "left",
                "padding": "longest",
                "add_special_tokens": True
            },
            "generation_params": {
                "min_length": 50,
                "max_length": 150,
                "length_penalty": 2.0,
                "early_stopping": True,
                "num_beams": 4,
                "no_repeat_ngram_size": 3,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "top_k": 50
            }
        }
    },
    "mt5-multilingual-base": {
        "display_name": "mT5 Base Multilingual Summarizer",
        "path": "csebuetnlp/mT5_multilingual_XLSum",
        "type": "summarization",
        "languages": ["fr", "en", "de", "es", "it"],
        "max_sequence_length": 1024,
        "gpu_requirements": {
            "vram_required": "4GB",
            "recommended_batch_size": 16
        },
        "load_params": {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "max_memory": {0: "4GiB", "cpu": "8GB"},
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "quant_method": "bitsandbytes"
            }
        },
        "generation_config": {
            "preprocessing": {
                "max_input_length": 1024,
                "max_output_length": 150,
                "truncation_strategy": "left",
                "padding": "longest",
                "add_special_tokens": True
            },
            "generation_params": {
                "min_length": 50,
                "max_length": 150,
                "length_penalty": 2.0,
                "early_stopping": True,
                "num_beams": 4,
                "no_repeat_ngram_size": 3,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "top_k": 50
            }
        }
    }
}

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
    },
    "thread_config": {
        "mkl_num_threads": int(os.environ.get("MKL_NUM_THREADS", "16")),
        "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "16")),
        "numpy_num_threads": int(os.environ.get("NUMEXPR_NUM_THREADS", "16"))
    }
}

def get_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Récupère la configuration d'un modèle par son nom.
    
    Args:
        model_name: Nom du modèle à rechercher
        
    Returns:
        Dict[str, Any]: Configuration du modèle ou None si non trouvé
    """
    # Chercher dans toutes les catégories de modèles
    if model_name in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_name]
    elif model_name in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[model_name]
    elif model_name in SUMMARIZER_MODELS:
        return SUMMARIZER_MODELS[model_name]
    
    # Chercher par chemin si le nom exact n'est pas trouvé
    for models_dict in [AVAILABLE_MODELS, EMBEDDING_MODELS, SUMMARIZER_MODELS]:
        for key, config in models_dict.items():
            if config.get("path") == model_name:
                return config
    
    return None

def get_optimized_load_params(model_name: str) -> Dict[str, Any]:
    """
    Obtient les paramètres de chargement optimisés pour un modèle spécifique.
    
    Args:
        model_name: Nom du modèle
        
    Returns:
        Dict[str, Any]: Paramètres de chargement optimisés
    """
    config = get_model_config(model_name)
    if not config:
        logger.warning(f"Configuration non trouvée pour {model_name}, utilisation des paramètres par défaut")
        return {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "max_memory": {0: "16GiB", "cpu": "12GB"}
        }
    
    # Récupération des paramètres de base
    load_params = config.get("load_params", {}).copy()
    
    # Adaptation des paramètres de mémoire selon la configuration CUDA
    cuda_config = get_cuda_config()
    model_type = config.get("type", "chat")
    
    # Mise à jour de max_memory selon le type de modèle
    if model_type == "chat":
        load_params["max_memory"] = cuda_config.memory_configs.get("high", {0: "16GiB", "cpu": "12GB"})
    elif model_type == "summarization":
        load_params["max_memory"] = cuda_config.memory_configs.get("medium", {0: "4GiB", "cpu": "8GB"})
    else:  # embedding ou autre
        load_params["max_memory"] = cuda_config.memory_configs.get("low", {0: "2GiB", "cpu": "4GB"})
    
    # Configuration de l'implementation de l'attention
    if cuda_config.use_flash_attention and "attn_implementation" not in load_params:
        load_params["attn_implementation"] = "flash_attention_2"
    
    return load_params

def export_model_configs(output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Exporte les configurations des modèles dans un fichier JSON.
    
    Args:
        output_path: Chemin de sortie optionnel
        
    Returns:
        Dict[str, Any]: Configurations des modèles
    """
    # Structure pour l'export
    configs = {
        "chat_models": {name: config["display_name"] for name, config in AVAILABLE_MODELS.items()},
        "embedding_models": {name: config["display_name"] for name, config in EMBEDDING_MODELS.items()},
        "summarizer_models": {name: config["display_name"] for name, config in SUMMARIZER_MODELS.items()},
        "default_models": {
            "chat": settings.MODEL_NAME,
            "embedding": settings.EMBEDDING_MODEL,
            "summarizer": settings.MODEL_NAME_SUMMARIZER
        }
    }
    
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(configs, f, indent=2)
        
        logger.info(f"Configurations des modèles exportées dans {output_path}")
    
    return configs