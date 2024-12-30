# core/llm/memory_manager.py
import gc
import psutil
import torch
import json
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

    def get_optimal_memory_config(self) -> Dict[str, str]:
        """Calcule la configuration mémoire optimale en utilisant les paramètres de settings."""
        try:
            # Charger la configuration MAX_MEMORY depuis settings
            if isinstance(settings.MAX_MEMORY, str):
                try:
                    memory_config = json.loads(settings.MAX_MEMORY)
                except json.JSONDecodeError:
                    logger.error("Erreur lors du parsing de MAX_MEMORY")
                    memory_config = {"0": "20GiB", "cpu": "24GB"}
            else:
                memory_config = settings.MAX_MEMORY

            logger.info(f"Configuration mémoire: {memory_config}")
            return memory_config

        except Exception as e:
            logger.error(f"Erreur lors de la lecture de la configuration mémoire: {e}")
            # Configuration par défaut en cas d'erreur
            return {"0": "20GiB", "cpu": "24GB"}

    async def cleanup(self):
        """Nettoie les ressources mémoire."""
        self.cleanup_memory()
