from sqlalchemy import (
    text,
    Table,
    Column,
    String,
    DateTime,
    JSON,
    MetaData,
    Index,
    create_engine
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import json
import uuid
from typing import Optional, Dict
from core.utils.logger import get_logger

logger = get_logger("session_manager")

class SessionManager:
    def __init__(self, database_url: str):
        """Initialise le gestionnaire de sessions."""
        try:
            self.engine = create_async_engine(
                database_url,
                echo=False,
                future=True
            )
            
            self.async_session = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self.metadata = MetaData()
            
            # Définition de la table chat_sessions
            self.chat_sessions = Table(
                'chat_sessions',
                self.metadata,
                Column('id', String, primary_key=True, default=lambda: str(uuid.uuid4())),
                Column('user_id', String, nullable=False),
                Column('session_id', String, unique=True, nullable=False),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
                Column('session_context', JSON, default=dict),
                
                # Définition des index
                Index('idx_user_id', 'user_id'),
                Index('idx_session_id', 'session_id', unique=True),
                Index('idx_updated_at', 'updated_at')
            )
            
            logger.info("SessionManager initialisé")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de SessionManager: {e}")
            raise

    async def initialize(self):
        """Initialise la base de données et crée les tables si nécessaire."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(self.metadata.create_all)
                logger.info("Tables de session initialisées")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des tables: {e}")
            raise

    async def update_session_context(self, session_id: str, new_context: Dict) -> bool:
        """Met à jour le contexte d'une session."""
        async with self.async_session() as session:
            try:
                # Récupérer le contexte actuel
                result = await session.execute(
                    text("SELECT session_context FROM chat_sessions WHERE session_id = :session_id"),
                    {"session_id": session_id}
                )
                row = result.fetchone()
                
                if row:
                    # Charger le contexte existant
                    current_context = (
                        row[0] if isinstance(row[0], dict) else json.loads(row[0])
                    )
                  
                    # Fusionner l'ancien et le nouveau contexte
                    merged_context = {**current_context, **new_context}
                    
                    # Mise à jour avec le contexte fusionné
                    await session.execute(
                        text("""
                        UPDATE chat_sessions 
                        SET session_context = :context,
                            updated_at = :updated_at
                        WHERE session_id = :session_id
                        """),
                        {
                            "session_id": session_id,
                            "context": json.dumps(merged_context),
                            "updated_at": datetime.utcnow()
                        }
                    )
                    await session.commit()
                    logger.debug(f"Contexte mis à jour pour la session {session_id}")
                    return True
                    
                logger.warning(f"Session non trouvée: {session_id}")
                return False
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur lors de la mise à jour du contexte: {e}")
                return False

    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Récupère une session existante."""
        async with self.async_session() as session:
            try:
                result = await session.execute(
                    text("SELECT * FROM chat_sessions WHERE session_id = :session_id"),
                    {"session_id": session_id}
                )
                row = result.fetchone()
                
                if row:
                    return {
                        "session_id": row.session_id,
                        "context": row.session_context or {}
                    }
                return None
                
            except Exception as e:
                logger.error(f"Erreur lors de la récupération de la session: {e}")
                return None

    async def create_session(self, user_id: str) -> Dict:
        """Crée une nouvelle session."""
        async with self.async_session() as session:
            try:
                session_id = str(uuid.uuid4())
                current_time = datetime.utcnow()
                record_id = str(uuid.uuid4())  # Génération de l'ID unique
             
                # Préparation du contexte initial
                session_context = {
                    "created_at": current_time.isoformat(),
                    "history": []
                }
             
                # Requête SQL avec placeholders nommés
                query = text("""
                    INSERT INTO chat_sessions 
                        (id, user_id, session_id, session_context, created_at, updated_at)
                    VALUES 
                        (:id, :user_id, :session_id, :session_context, :created_at, :updated_at)
                    RETURNING session_id, session_context
                """)
             
                params = {
                    "id": record_id,  # Ajout de l'ID généré
                    "user_id": user_id,
                    "session_id": session_id,
                    "session_context": json.dumps(session_context),
                    "created_at": current_time,
                    "updated_at": current_time,
                }

                result = await session.execute(query, params)
                await session.commit()
                
                # Récupération du résultat
                row = result.fetchone()
                if row:
                    context = row[1] if isinstance(row[1], dict) else json.loads(row[1])
                    created_session = {
                        "session_id": row[0],
                        "context": context
                    }
                    logger.info(f"Nouvelle session créée: {session_id}")
                    return created_session
                
                raise ValueError("La session n'a pas pu être créée")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur lors de la création de la session: {e}")
                raise

    async def cleanup_old_sessions(self, days: int = 30) -> int:
        """Nettoie les anciennes sessions."""
        async with self.async_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                result = await session.execute(
                    text("""
                    DELETE FROM chat_sessions 
                    WHERE updated_at < :cutoff_date 
                    RETURNING id
                    """),
                    {"cutoff_date": cutoff_date}
                )
                deleted = len(result.fetchall())
                await session.commit()
                logger.info(f"{deleted} anciennes sessions supprimées")
                return deleted
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur lors du nettoyage des sessions: {e}")
                return 0