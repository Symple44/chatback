from sqlalchemy import (
    Column, String, DateTime, JSON, Text, Integer, text,
    select, update, delete
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import uuid
from typing import Optional, Dict, List
import logging

from core.config import settings
from core.utils.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    additional_data = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ChatRequest(Base):
    __tablename__ = 'chat_requests'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), nullable=False)
    user_query = Column(Text, nullable=False)
    chatbot_response = Column(Text, nullable=False)
    resolution_method = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    session_id = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    session_context = Column(JSON, nullable=True, server_default=text('\'{}\''))

class DatabaseManager:
    def __init__(self, database_url: str):
        """Initialise le gestionnaire de base de données."""
        try:
            self.engine = create_async_engine(
                database_url,
                echo=False,
                future=True,
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW
            )
            
            self.async_session = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("Connexion à la base de données initialisée")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la base de données: {e}")
            raise

    async def initialize_tables(self):
        """Initialise les tables de la base de données."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Tables initialisées")

    async def save_chat_request(
        self,
        user_id: str,
        query: str,
        response: str,
        session_id: Optional[str] = None,
        additional_data: Optional[Dict] = None
    ) -> bool:
        """Sauvegarde une requête de chat."""
        async with self.async_session() as session:
            try:
                # Sauvegarde dans chat_history
                chat_history = ChatHistory(
                    user_id=user_id,
                    session_id=session_id or str(uuid.uuid4()),
                    query=query,
                    response=response,
                    additional_data=additional_data or {}
                )
                
                # Sauvegarde dans chat_requests
                chat_request = ChatRequest(
                    user_id=user_id,
                    user_query=query,
                    chatbot_response=response,
                    resolution_method="model"
                )
                
                session.add(chat_history)
                session.add(chat_request)
                await session.commit()
                
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur lors de la sauvegarde du chat: {e}")
                return False

    async def get_chat_history(
    self,
    user_id: str,
    session_id: Optional[str] = None,
    limit: int = 50
) -> List[Dict]:
        """Récupère l'historique des conversations."""
        async with self.async_session() as session:
            try:
                query = select(ChatHistory)\
                    .filter(ChatHistory.user_id == user_id)
                
                if session_id:
                    query = query.filter(ChatHistory.session_id == session_id)
                
                query = query.order_by(ChatHistory.timestamp.desc()).limit(limit)
                result = await session.execute(query)
                history = result.scalars().all()
                
                return [{
                    "id": str(entry.id),
                    "query": entry.query,
                    "response": entry.response,
                    # Ajout d'une vérification pour le timestamp
                    "timestamp": entry.timestamp.isoformat() if entry.timestamp else datetime.utcnow().isoformat(),
                    "additional_data": entry.additional_data or {}  # Ajout d'une valeur par défaut
                } for entry in history]
                
            except Exception as e:
                logger.error(f"Erreur lors de la récupération de l'historique: {e}")
                return []

    async def get_session_history(self, session_id: str) -> List[Dict]:
        """Récupère l'historique d'une session spécifique."""
        async with self.async_session() as session:
            try:
                query = select(ChatHistory)\
                    .filter(ChatHistory.session_id == session_id)\
                    .order_by(ChatHistory.timestamp)
                
                result = await session.execute(query)
                history = result.scalars().all()
                
                return [{
                    "id": str(entry.id),
                    "query": entry.query,
                    "response": entry.response,
                    "timestamp": entry.timestamp.isoformat(),
                    "additional_data": entry.additional_data
                } for entry in history]
                
            except Exception as e:
                logger.error(f"Erreur lors de la récupération de l'historique de session: {e}")
                return []

    async def create_session(self, user_id: str) -> Dict:
        """Crée une nouvelle session."""
        async with self.async_session() as session:
            try:
                # Générer l'id de session
                session_id = str(uuid.uuid4())
                logger.info(f"Création session pour user {user_id}: {session_id}")
      
                # Créer l'instance de ChatSession
                new_session = ChatSession(
                    id=str(uuid.uuid4()),  # ID unique pour la session
                    user_id=user_id,
                    session_id=session_id,
                    session_context={},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                # Ajouter et commiter
                session.add(new_session)
                await session.commit()
                
                logger.info(f"Session créée avec succès: {session_id}")
      
                return {
                    "session_id": session_id,
                    "created_at": new_session.created_at.isoformat()
                }
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur lors de la création de la session: {e}")
                raise

    async def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Récupère les informations d'une session."""
        async with self.async_session() as session:
            try:
                result = await session.execute(
                    select(ChatSession)
                    .filter(ChatSession.session_id == session_id)
                )
                
                session_info = result.scalar_one_or_none()
                if session_info:
                    return {
                        "id": str(session_info.id),
                        "user_id": session_info.user_id,
                        "session_id": session_info.session_id,
                        "created_at": session_info.created_at.isoformat(),
                        "updated_at": session_info.updated_at.isoformat(),
                        "context": session_info.session_context
                    }
                return None
                
            except Exception as e:
                logger.error(f"Erreur lors de la récupération de la session: {e}")
                return None

    async def update_session_context(
        self,
        session_id: str,
        context: Dict
    ) -> bool:
        """Met à jour le contexte d'une session."""
        async with self.async_session() as session:
            try:
                stmt = (
                    update(ChatSession)
                    .where(ChatSession.session_id == session_id)
                    .values(session_context=context)
                )
                
                result = await session.execute(stmt)
                await session.commit()
                
                return result.rowcount > 0
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur lors de la mise à jour du contexte: {e}")
                return False

    async def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Nettoie les anciennes sessions."""
        async with self.async_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days_old)
                
                # Suppression des anciennes sessions
                delete_query = delete(ChatSession).where(
                    ChatSession.updated_at < cutoff_date
                )
                result = await session.execute(delete_query)
                await session.commit()
                
                return result.rowcount
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur lors du nettoyage des sessions: {e}")
                return 0

    async def check_health(self) -> bool:
        """Vérifie l'état de la connexion à la base de données."""
        try:
            async with self.async_session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Erreur de connexion à la base de données: {e}")
            return False