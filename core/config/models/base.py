# core/config/models/base.py
from enum import Enum
from typing import Dict, Any, List, Optional

class ModelType(Enum):
    """Types de modèles supportés."""
    CHAT = "chat"
    EMBEDDING = "embedding" 
    SUMMARIZER = "summarizer"

def get_load_params(model_config: Dict[str, Any], model_type: ModelType, gpu_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Génère les paramètres de chargement optimisés pour un modèle.
    
    Args:
        model_config: Configuration du modèle
        model_type: Type de modèle
        gpu_config: Configuration GPU
        
    Returns:
        Paramètres de chargement
    """
    # Base params
    load_params = model_config.get("load_params", {}).copy()
    
    # Optimiser les allocations mémoire selon le type
    if model_type == ModelType.CHAT:
        load_params["max_memory"] = {"0": gpu_config["vram_allocation"]["model"], 
                                   "cpu": gpu_config["ram_allocation"]["model_offload"]}
    elif model_type == ModelType.SUMMARIZER:
        load_params["max_memory"] = {"0": gpu_config["vram_allocation"].get("cache", "4GB"), 
                                   "cpu": "8GB"}
    elif model_type == ModelType.EMBEDDING:
        load_params["max_memory"] = {"0": gpu_config["vram_allocation"]["embeddings"], 
                                   "cpu": "4GB"}
    
    # Flash attention si disponible
    if gpu_config["cuda_config"].get("flash_attention") and "attn_implementation" not in load_params:
        load_params["attn_implementation"] = "flash_attention_2"
    
    return load_params