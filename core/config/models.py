# core/config/models.py
from typing import Dict, List, Any
from pydantic import BaseModel
import torch

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
        # Nouvelle section pour les paramètres de génération spécifiques au modèle
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
                "num_beams": 1,
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
        # Configuration spécifique pour Mixtral
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
                "num_beams": 1,
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
            "recommended_batch_size": 8
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
                "llm_int8_enable_fp32_cpu_offload": True
            },
            "low_cpu_mem_usage": True,
            "offload_folder": "offload_folder",
            "attn_implementation": "flash_attention_2"
        },
        # Configuration spécifique pour le modèle 24B
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
                "num_beams": 1,
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
        "batch_size": 8,
        "prefetch_factor": 2,
        "num_workers": 4,
        "pin_memory": True,
        "dataloader_params": {
            "persistent_workers": True,
            "shuffle_buffer_size": 10000
        },
        "execution": {
            "forward_batch_size": 8,
            "mixed_precision": True,
            "gradient_checkpointing": True
        },
        "optimization": {
            "weight_tying": True,
            "memory_efficient_attention": True,
            "use_kernel_injection": True
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
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True
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
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True
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

# Configuration système globale
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