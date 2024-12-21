# core/utils/logger.py
import logging
from typing import Optional
from pathlib import Path

class LoggerManager:
    def __init__(self):
        self.initialized = False
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

    async def initialize(self):
        """Initialise le système de logs."""
        if not self.initialized:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(self.log_dir / "app.log")
                ]
            )
            self.initialized = True

def get_logger(name: str) -> logging.Logger:
    """Retourne un logger configuré."""
    return logging.getLogger(name)

# Instance unique du manager de logs
logger_manager = LoggerManager()
