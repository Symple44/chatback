# core/database/base.py

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool
from typing import Optional, Dict, Any
import asyncio
from datetime import datetime

from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger("database")

# Création de la classe de base pour les modèles SQLAlchemy
Base = declarative_base()

class DatabaseSessionManager:
    def __init__(self, database_url: str):
        """
        Initialise le gestionnaire de session de base de données.
        """
        self.engine = create_async_engine(
            database_url,
            echo=settings.DEBUG,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True,
            poolclass=AsyncAdaptedQueuePool
        )
        
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False
        )
        
        self._last_health_check = None
        self._health_check_interval = 30  # seconds
        self.health_check_lock = asyncio.Lock()

    async def create_all(self):
        """Crée toutes les tables définies dans les modèles."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                logger.info("Tables de base de données créées avec succès")
                metrics.increment_counter("database_schema_updates")
        except Exception as e:
            logger.error(f"Erreur lors de la création des tables: {e}")
            metrics.increment_counter("database_schema_errors")
            raise

    async def drop_all(self):
        """Supprime toutes les tables."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                logger.warning("Toutes les tables ont été supprimées")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression des tables: {e}")
            raise

    async def get_session(self) -> AsyncSession:
        """Retourne une nouvelle session de base de données."""
        return self.session_factory()

    async def close(self):
        """Ferme toutes les connexions."""
        try:
            await self.engine.dispose()
            logger.info("Connexions à la base de données fermées")
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture des connexions: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état de la connexion à la base de données.
        
        Returns:
            Dict contenant l'état de santé
        """
        async with self.health_check_lock:
            # Éviter les vérifications trop fréquentes
            if (self._last_health_check and 
                (datetime.utcnow() - self._last_health_check).total_seconds() < self._health_check_interval):
                return {"status": "cached"}

            try:
                async with self.session_factory() as session:
                    await session.execute(text("SELECT 1"))
                    pool_status = await self.get_pool_status()
                    
                    self._last_health_check = datetime.utcnow()
                    
                    return {
                        "status": True,
                        "timestamp": self._last_health_check.isoformat(),
                        "pool": pool_status
                    }
            except Exception as e:
                logger.error(f"Erreur de connexion à la base de données: {e}")
                return {
                    "status": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }

    async def get_pool_status(self) -> Dict[str, int]:
        """
        Retourne le statut du pool de connexions.
        
        Returns:
            Dict contenant les métriques du pool
        """
        try:
            return {
                "size": self.engine.pool.size(),
                "checked_out": self.engine.pool.checkedout(),
                "overflow": self.engine.pool.overflow(),
                "checkedin": self.engine.pool.checkedin()
            }
        except Exception as e:
            logger.error(f"Erreur récupération statut pool: {e}")
            return {}

    async def vacuum_analyze(self):
        """Exécute VACUUM ANALYZE sur la base de données."""
        try:
            async with self.engine.begin() as conn:
                await conn.execute("VACUUM ANALYZE")
            logger.info("VACUUM ANALYZE exécuté avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de VACUUM ANALYZE: {e}")
            raise

    async def cleanup_expired_sessions(self, days: int = 30):
        """
        Nettoie les sessions expirées.
        
        Args:
            days: Nombre de jours avant expiration
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    f"""
                    DELETE FROM chat_sessions 
                    WHERE updated_at < NOW() - INTERVAL '{days} days'
                    """
                )
                await session.commit()
                deleted_count = result.rowcount
                logger.info(f"{deleted_count} sessions expirées nettoyées")
                metrics.increment_counter("database_cleanups")
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des sessions: {e}")
            metrics.increment_counter("database_cleanup_errors")
            raise

# Singleton pour le gestionnaire de session
_session_manager: Optional[DatabaseSessionManager] = None

def get_session_manager(database_url: Optional[str] = None) -> DatabaseSessionManager:
    global _session_manager
    if _session_manager is None:
        if database_url is None:
            raise ValueError("database_url est requis pour la première initialisation")
        _session_manager = DatabaseSessionManager(database_url)
        logger.info("Nouveau gestionnaire de session créé")
    return _session_manager

async def cleanup_database():
    """Nettoie les ressources de la base de données."""
    global _session_manager
    if _session_manager is not None:
        await _session_manager.close()
        _session_manager = None
        logger.info("Ressources de la base de données nettoyées")

# Context manager pour les sessions
class DatabaseSession:
    """Context manager pour gérer les sessions de base de données."""
    
    def __init__(self):
        self.session_manager = get_session_manager()
    
    async def __aenter__(self) -> AsyncSession:
        self.session = await self.session_manager.get_session()
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.session.rollback()
        await self.session.close()
