# core/config/models/summarizer.py
from typing import Dict, Any

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
            "torch_dtype": "float16",
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
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
            "torch_dtype": "float16",
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
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