# core/config/models/chat.py
import os
from typing import Dict, Any

# Configuration des modèles de chat/instruction disponibles
CHAT_MODELS = {
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
            "torch_dtype": "float16",
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
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
            "torch_dtype": "float16",
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
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
            "torch_dtype": "float16",
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
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

# Configuration de performance par modèle
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