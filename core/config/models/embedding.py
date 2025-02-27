# core/config/models/embedding.py
from typing import Dict, Any

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
            "torch_dtype": "float16"
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
            "torch_dtype": "float16"
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