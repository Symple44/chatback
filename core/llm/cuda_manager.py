# core/llm/cuda_manager.py
import torch
from typing import Dict, Any
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("cuda_manager")

class CUDAManager:
    def __init__(self):
        """Initialise le gestionnaire CUDA."""
        self.setup_cuda_environment()

    def setup_cuda_environment(self):
        """Configure l'environnement CUDA."""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA non disponible - utilisation du CPU")
                return

            # Configuration du device
            device_id = 0
            torch.cuda.set_device(device_id)
            
            # Logging des informations GPU
            self._log_gpu_info(device_id)
            
        except Exception as e:
            logger.error(f"Erreur configuration CUDA: {e}")
            raise

    def _log_gpu_info(self, device_id: int):
        """Log les informations sur le GPU."""
        try:
            gpu_name = torch.cuda.get_device_name(device_id)
            gpu_cap = torch.cuda.get_device_capability(device_id)
            gpu_mem = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            
            logger.info(f"GPU {device_id}: {gpu_name} (Compute {gpu_cap[0]}.{gpu_cap[1]})")
            logger.info(f"Mémoire GPU totale: {gpu_mem:.2f} GB")
            logger.info(f"Utilisation du device: cuda:{device_id}")
            
        except Exception as e:
            logger.error(f"Erreur logging GPU: {e}")

    def get_model_load_parameters(self, model_name: str, max_memory: Dict) -> Dict[str, Any]:
        """Retourne les paramètres optimaux pour le chargement du modèle."""
        load_params = {
            "pretrained_model_name_or_path": model_name,
            "revision": settings.MODEL_REVISION,
            "device_map": "auto",
            "max_memory": max_memory,
            "trust_remote_code": True,
        }

        if settings.USE_FP16:
            load_params["torch_dtype"] = torch.float16
        elif settings.USE_8BIT:
            load_params["load_in_8bit"] = True
        elif settings.USE_4BIT:
            load_params["load_in_4bit"] = True

        return load_params

    async def cleanup(self):
        """Nettoie les ressources CUDA."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
