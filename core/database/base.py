# core/database/base.py

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine
from typing import Optional

# Création de la classe de base pour les modèles SQLAlchemy
Base = declarative_base()

class DatabaseSessionManager:
    def __init__(self, database_url: str):
        """
        Initialise le gestionnaire de session de base de données.
        
        Args:
            database_url: URL de connexion à la base de données
        """
        self.engine = create_async_engine(
            database_url,
            echo=False,  # Mettre à True pour voir les requêtes SQL
            future=True,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800
        )
        
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False
        )

    async def create_all(self):
        """Crée toutes les tables définies dans les modèles."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_all(self):
        """Supprime toutes les tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def get_session(self) -> AsyncSession:
        """Retourne une nouvelle session de base de données."""
        return self.session_factory()

    async def close(self):
        """Ferme toutes les connexions."""
        await self.engine.dispose()

    async def health_check(self) -> bool:
        """Vérifie l'état de la connexion à la base de données."""
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Erreur de connexion à la base de données: {e}")
            return False

    async def get_pool_status(self) -> dict:
        """Retourne le statut du pool de connexions."""
        return {
            "pool_size": self.engine.pool.size(),
            "checked_out": self.engine.pool.checkedin(),
            "overflow": self.engine.pool.overflow(),
            "checkedout": self.engine.pool.checkedout()
        }

# Singleton pour le gestionnaire de session
_session_manager: Optional[DatabaseSessionManager] = None

def get_session_manager(database_url: Optional[str] = None) -> DatabaseSessionManager:
    """
    Retourne l'instance du gestionnaire de session.
    En crée une nouvelle si elle n'existe pas.
    
    Args:
        database_url: URL de connexion à la base de données (nécessaire uniquement à la première création)
    
    Returns:
        DatabaseSessionManager: Instance du gestionnaire de session
    """
    global _session_manager
    if _session_manager is None:
        if database_url is None:
            raise ValueError("database_url est requis pour la première initialisation")
        _session_manager = DatabaseSessionManager(database_url)
    return _session_manager

async def cleanup_database():
    """Nettoie les ressources de la base de données."""
    global _session_manager
    if _session_manager is not None:
        await _session_manager.close()
        _session_manager = None
