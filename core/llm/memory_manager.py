# core/llm/memory_manager.py
import gc
import psutil
import torch
from typing import Dict
from core.utils.logger import get_logger
from core.config import settings

logger = get_logger("memory_manager")

class MemoryManager:
    def __init__(self):
        """Initialise le gestionnaire de mémoire."""
        self.cleanup_memory()

    def cleanup_memory(self):
        """Nettoie la mémoire."""
        try:
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1024**3
                max_memory = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"Mémoire GPU allouée: {allocated:.2f}GB")
                logger.info(f"Pic mémoire GPU: {max_memory:.2f}GB")

            memory = psutil.virtual_memory()
            logger.info(f"RAM disponible: {memory.available/1024**3:.2f}GB")
            logger.info(f"Utilisation RAM: {memory.percent}%")

        except Exception as e:
            logger.error(f"Erreur nettoyage mémoire: {e}")

    def get_optimal_memory_config(self) -> Dict:
        """Calcule la configuration mémoire optimale."""
        try:
            if not torch.cuda.is_available():
                return {"cpu": f"{psutil.virtual_memory().available / 1024**3:.0f}GB"}

            device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            cuda_limit = min(device_memory * 0.95, float(settings.MAX_MEMORY.get("0", "22").replace("GiB", "")))
            
            return {
                "0": f"{cuda_limit:.0f}GiB",
                "cpu": f"{psutil.virtual_memory().available / 1024**3:.0f}GB"
            }

        except Exception as e:
            logger.error(f"Erreur calcul config mémoire: {e}")
            return settings.MAX_MEMORY

    async def cleanup(self):
        """Nettoie les ressources mémoire."""
        self.cleanup_memory()
