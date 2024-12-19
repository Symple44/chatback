# core/utlis/logger.py
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
from concurrent.futures import ThreadPoolExecutor
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
        """
        Formate un enregistrement de log en JSON avec des métadonnées enrichies.
        """
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

        # Ajout des informations d'exception
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            log_object["exception"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
                "traceback": self.format_traceback(exc_tb)
            }

        # Ajout des champs supplémentaires
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
    """Handler de fichier rotatif asynchrone."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=1)

    def emit(self, record):
        """Émet un enregistrement de manière asynchrone."""
        try:
            self.executor.submit(self._emit_sync, record)
        except Exception:
            self.handleError(record)

    def _emit_sync(self, record):
        """Émission synchrone de l'enregistrement."""
        try:
            msg = self.format(record)
            self.do_write(msg)
            self.do_rollover_check()
        except Exception:
            self.handleError(record)

    def do_rollover_check(self):
        """Vérifie et effectue la rotation si nécessaire."""
        if self.should_rollover():
            self.do_rollover()

    def do_write(self, msg):
        """Écrit le message avec gestion du verrouillage."""
        with self.lock:
            if self.stream is None:
                self.stream = self._open()
            self.stream.write(msg + self.terminator)
            self.stream.flush()

class LoggerManager:
    """Gestionnaire centralisé des loggers."""
    
    def __init__(self):
        """Initialise le gestionnaire de logs."""
        self.log_dir = Path(settings.LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.loggers: Dict[str, logging.Logger] = {}
        self.default_level = getattr(logging, settings.LOG_LEVEL.upper())
        self.log_queue = asyncio.Queue()
        
        # Configuration du logging de base
        logging.basicConfig(
            level=self.default_level,
            format=settings.LOG_FORMAT
        )
        
        # Démarrage du worker de traitement des logs
        asyncio.create_task(self._process_log_queue())

    async def _process_log_queue(self):
        """Traite la file d'attente des logs."""
        while True:
            try:
                log_entry = await self.log_queue.get()
                await self._write_log_entry(log_entry)
            except Exception as e:
                print(f"Erreur traitement log: {e}")
            finally:
                self.log_queue.task_done()

    async def _write_log_entry(self, entry: Dict):
        """Écrit une entrée de log dans le fichier approprié."""
        logger_name = entry.get("name", "default")
        log_file = self.log_dir / f"{logger_name}.log"
        
        try:
            async with aiofiles.open(log_file, "a") as f:
                await f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Erreur écriture log: {e}")

    def get_logger(
        self,
        name: str,
        level: Optional[str] = None,
        extra_fields: Optional[Dict] = None
    ) -> logging.Logger:
        """
        Crée ou récupère un logger configuré.
        
        Args:
            name: Nom du logger
            level: Niveau de log optionnel
            extra_fields: Champs supplémentaires à inclure
            
        Returns:
            Logger configuré
        """
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level or self.default_level)
        logger.handlers.clear()

        # Handler pour fichier principal
        main_handler = AsyncRotatingFileHandler(
            filename=self.log_dir / f"{name}.log",
            maxBytes=settings.LOG_FILE_MAX_BYTES,
            backupCount=settings.LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        main_handler.setFormatter(CustomJsonFormatter(
            extra_fields=extra_fields or {}
        ))
        logger.addHandler(main_handler)

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

def log_exception(logger: Optional[logging.Logger] = None):
    """
    Décorateur pour logger les exceptions.
    
    Usage:
        @log_exception(logger)
        async def my_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                log = logger or logging.getLogger(func.__module__)
                log.exception(
                    f"Exception dans {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "arguments": str(args),
                        "kwargs": str(kwargs)
                    }
                )
                raise
        return wrapper
    return decorator

# Instance singleton du gestionnaire
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
