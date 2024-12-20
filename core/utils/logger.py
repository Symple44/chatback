# core/utils/logger.py
import logging
import json
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
import os
import traceback
import threading
from functools import wraps
import gzip

from core.config import settings

class CustomJsonFormatter(logging.Formatter):
    """Formateur personnalisé pour les logs en JSON."""
    
    def __init__(self, **kwargs):
        self.extra_fields = kwargs.pop("extra_fields", {})
        super().__init__()
        self.default_keys = {
            "timestamp", "name", "level",
            "module", "function", "line", "message"
        }

    def format(self, record: logging.LogRecord) -> str:
        """Formate un enregistrement de log en JSON."""
        log_object = {
            "timestamp": datetime.utcnow().isoformat(),
            "name": record.name,
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "thread": threading.current_thread().name,
            "process": os.getpid()
        }

        # Ajout des informations d'exception si présentes
        if record.exc_info:
            log_object["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.format_traceback(record.exc_info[2])
            }

        # Ajout des attributs supplémentaires
        if hasattr(record, "extra_data"):
            log_object.update(record.extra_data)

        # Ajout des champs par défaut
        log_object.update(self.extra_fields)

        return json.dumps(log_object)

    def format_traceback(self, tb) -> List[Dict[str, str]]:
        """Formate le traceback de manière structurée."""
        formatted = []
        for filename, line, func, text in traceback.extract_tb(tb):
            formatted.append({
                "file": filename,
                "line": line,
                "function": func,
                "text": text
            })
        return formatted

class AsyncRotatingFileHandler(RotatingFileHandler):
    """Handler de fichier rotatif avec support asynchrone."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()

    def emit(self, record):
        """Émet un enregistrement de manière thread-safe."""
        with self.lock:
            super().emit(record)

class LoggerManager:
    """Gestionnaire centralisé des loggers."""
    
    def __init__(self):
        """Initialise le gestionnaire de logs."""
        self.log_dir = Path(settings.LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.loggers: Dict[str, logging.Logger] = {}
        self.default_level = getattr(logging, settings.LOG_LEVEL.upper())
        
        # Configuration du logging de base
        logging.basicConfig(
            level=self.default_level,
            format=settings.LOG_FORMAT
        )

    async def initialize(self):
        """Initialise le gestionnaire de logs."""
        # S'assurer que le répertoire de logs existe
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration initiale des handlers
        self._setup_default_handlers()
        
        # Nettoyage initial des anciens logs
        await self._cleanup_old_logs()

    def _setup_default_handlers(self):
        """Configure les handlers par défaut."""
        # Handler pour la console en mode debug
        if settings.DEBUG:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
            logging.getLogger().addHandler(console_handler)

    def get_logger(
        self,
        name: str,
        level: Optional[str] = None,
        extra_fields: Optional[Dict] = None
    ) -> logging.Logger:
        """
        Récupère ou crée un logger.
        
        Args:
            name: Nom du logger
            level: Niveau de log optionnel
            extra_fields: Champs supplémentaires
            
        Returns:
            Logger configuré
        """
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level or self.default_level)
        logger.handlers.clear()

        # Handler pour fichier avec rotation
        file_handler = AsyncRotatingFileHandler(
            filename=self.log_dir / f"{name}.log",
            maxBytes=settings.LOG_FILE_MAX_BYTES,
            backupCount=settings.LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setFormatter(CustomJsonFormatter(
            extra_fields=extra_fields or {}
        ))
        logger.addHandler(file_handler)

        # Handler pour erreurs
        error_handler = TimedRotatingFileHandler(
            filename=self.log_dir / "error.log",
            when="midnight",
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(CustomJsonFormatter(
            extra_fields={"source": name}
        ))
        logger.addHandler(error_handler)

        # Handler console en mode debug
        if settings.DEBUG:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(console_handler)

        self.loggers[name] = logger
        return logger

    async def cleanup_old_logs(self, days: int = 30):
        """
        Nettoie et archive les anciens logs.
        
        Args:
            days: Nombre de jours à conserver
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            archive_dir = self.log_dir / "archive"
            archive_dir.mkdir(exist_ok=True)

            for log_file in self.log_dir.glob("*.log.*"):
                if log_file.stat().st_mtime < cutoff.timestamp():
                    # Compression et archivage
                    archive_path = archive_dir / f"{log_file.name}.gz"
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(archive_path, 'wb') as f_out:
                            f_out.write(f_in.read())
                    
                    # Suppression du fichier original
                    log_file.unlink()

            logger.info(f"Nettoyage des logs terminé")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage des logs: {e}")

    async def _cleanup_old_logs(self):
        """Nettoie les anciens logs au démarrage."""
        await self.cleanup_old_logs()

# Création de l'instance unique
logger_manager = LoggerManager()

def get_logger(
    name: str,
    level: Optional[str] = None,
    extra_fields: Optional[Dict] = None
) -> logging.Logger:
    """
    Fonction utilitaire pour obtenir un logger.
    
    Args:
        name: Nom du logger
        level: Niveau de log optionnel
        extra_fields: Champs supplémentaires
        
    Returns:
        Logger configuré
    """
    return logger_manager.get_logger(name, level, extra_fields)

# Instance de logger pour ce module
logger = get_logger("logger")
