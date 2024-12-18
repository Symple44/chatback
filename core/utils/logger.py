import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict
from logging.handlers import RotatingFileHandler
import os
from core.config import settings

class CustomJsonFormatter(logging.Formatter):
    """Formateur personnalisé pour les logs en JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formate un enregistrement de log en JSON."""
        log_object = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Ajout des informations d'exception si présentes
        if record.exc_info:
            log_object["exception"] = {
                "type": str(record.exc_info[0].__name__),
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }

        # Ajout des attributs supplémentaires
        if hasattr(record, "extra_data"):
            log_object["extra"] = record.extra_data

        return json.dumps(log_object)

def setup_logger(name: str = "chatbot") -> logging.Logger:
    """
    Configure et retourne un logger avec les paramètres appropriés.
    
    Args:
        name: Nom du logger
        
    Returns:
        Logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Création du dossier de logs si nécessaire
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Handler pour fichier avec rotation
    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, f"{name}.log"),
        maxBytes=10_000_000,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(CustomJsonFormatter())

    # Handler pour console
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)

    # Ajout des handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class LoggerAdapter(logging.LoggerAdapter):
    """Adaptateur pour ajouter des informations contextuelles aux logs."""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Traite le message de log en ajoutant des informations contextuelles.
        """
        extra = kwargs.get("extra", {})
        if self.extra:
            extra.update(self.extra)
        kwargs["extra"] = {"extra_data": extra}
        return msg, kwargs

def get_logger(name: str, **kwargs) -> LoggerAdapter:
    """
    Obtient un logger avec contexte.
    
    Args:
        name: Nom du logger
        kwargs: Informations contextuelles supplémentaires
        
    Returns:
        LoggerAdapter configuré
    """
    logger = setup_logger(name)
    return LoggerAdapter(logger, kwargs)

# Création du logger principal
logger = get_logger("chatbot", version="2.0.0", environment=os.getenv("ENV", "production"))

# Exemple d'utilisation:
# logger.info("Message", extra={"user_id": "123", "action": "login"})
# logger.error("Erreur", exc_info=True, extra={"context": "authentication"})