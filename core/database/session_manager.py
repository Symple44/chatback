from sqlalchemy import (
    select,
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
from sqlalchemy.orm import sessionmaker, selectinload, joinedload
from datetime import datetime, timedelta
import json
import uuid
from typing import Optional, Dict, List, Any
from core.utils.logger import get_logger
from core.database.models import ChatSession, ChatHistory
from core.config.config import settings

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
                # Récupérer la session et son contexte actuel
                result = await session.execute(
                    text("SELECT session_context FROM chat_sessions WHERE session_id = :session_id"),
                    {"session_id": session_id}
                )
                row = result.fetchone()
                
                if row:
                    # Charger le contexte existant
                    current_context = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                    
                    # Construction du nouveau contexte
                    history = current_context.get("history", [])
                    if "query" in new_context and "response" in new_context:
                        history.append({
                            "query": new_context["query"],
                            "response": new_context["response"],
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        # Garder les 5 dernières interactions
                        history = history[-5:]
                    
                    # Mise à jour du contexte
                    merged_context = {
                        **current_context,
                        "history": history,
                        "last_updated": datetime.utcnow().isoformat(),
                        "metadata": {
                            **current_context.get("metadata", {}),
                            **new_context.get("metadata", {})
                        },
                        "stats": {
                            "total_interactions": len(history),
                            "last_interaction": datetime.utcnow().isoformat()
                        }
                    }
                    
                    # Mise à jour en base
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
        """Récupère une session existante avec son contexte enrichi."""
        async with self.async_session() as session:
            try:
                result = await session.execute(
                    text("""
                    SELECT s.*, 
                           (SELECT json_agg(
                               json_build_object(
                                   'query', h.query,
                                   'response', h.response,
                                   'timestamp', h.created_at
                               )
                           ) 
                            FROM (
                                SELECT *
                                FROM chat_history
                                WHERE session_id = s.session_id
                                ORDER BY created_at DESC
                                LIMIT 5
                            ) h
                           ) as recent_history
                    FROM chat_sessions s
                    WHERE s.session_id = :session_id
                    """),
                    {"session_id": session_id}
                )
                row = result.fetchone()
                
                if row:
                    context = row.session_context or {}
                    if row.recent_history:
                        context["history"] = list(reversed(row.recent_history))
                    
                    return {
                        "session_id": row.session_id,
                        "context": context
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
                    "user_id": str(user_id),
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

    async def get_or_create_session(
        self,
        session_id: Optional[str],
        user_id: str,
        metadata: Dict
    ) -> ChatSession:
        """
        Récupère ou crée une session de chat avec contexte enrichi.
        """
        async with self.async_session() as session:
            if session_id:
                stmt = select(ChatSession).where(ChatSession.session_id == str(session_id))
                result = await session.execute(stmt)
                chat_session = result.scalar_one_or_none()
                
                if chat_session:
                    # Mise à jour des métadonnées existantes
                    if metadata:
                        chat_session.session_metadata = {
                            **chat_session.session_metadata,
                            **metadata,
                            "last_updated": datetime.utcnow().isoformat()
                        }
                    chat_session.updated_at = datetime.utcnow()
                    await session.commit()
                    return chat_session

            # Création d'une nouvelle session avec métadonnées enrichies
            new_session = ChatSession(
                user_id=user_id,
                session_id=str(uuid.uuid4()),
                session_context={
                    "created_at": datetime.utcnow().isoformat(),
                    "source": metadata.get("source", "api"),
                    "history": [],
                    "preferences": metadata.get("preferences", {
                        "language": "fr",
                        "model": settings.MODEL_NAME,
                        "max_history": 5
                    })
                },
                session_metadata={  # Ajout des métadonnées
                    **metadata,
                    "created_from": "chat_api",
                    "user_id": str(user_id),
                    "creation_timestamp": datetime.utcnow().isoformat(),
                    "system_info": {
                        "model_version": settings.VERSION,
                        "api_version": "1.0"
                    }
                },
                is_active=True
            )

            session.add(new_session)
            await session.commit()
            await session.refresh(new_session)
            return new_session
                
    async def get_session_history(self, session_id: str, limit: int = 5) -> List[ChatHistory]:
        """Récupère l'historique récent d'une session."""
        async with self.async_session() as session:
            result = await session.execute(
                text("""
                SELECT * FROM chat_history 
                WHERE session_id = :session_id
                ORDER BY created_at DESC 
                LIMIT :limit
                """),
                {"session_id": session_id, "limit": limit}
            )
            return list(reversed(result.fetchall()))

    async def get_chat_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Récupère l'historique des conversations pour un utilisateur."""
        async with self.async_session() as session:
            try:
                # Utiliser text() pour éviter les problèmes de type
                query = text("""
                    SELECT id, session_id, query, response, created_at, 
                           confidence_score, tokens_used, processing_time
                    FROM chat_history 
                    WHERE user_id = :user_id 
                    ORDER BY created_at DESC 
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {
                    'user_id': str(user_id), 
                    'limit': limit
                })
                
                return [
                    {
                        'id': str(row.id),
                        'session_id': row.session_id,
                        'query': row.query,
                        'response': row.response,
                        'created_at': row.created_at,
                        'confidence_score': row.confidence_score,
                        'tokens_used': row.tokens_used,
                        'processing_time': row.processing_time
                    }
                    for row in result
                ]
            except Exception as e:
                logger.error(f"Erreur récupération historique: {e}")
                return []
                
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Calcule les statistiques d'une session."""
        async with self.async_session() as session:
            try:
                # Statistiques des messages
                result = await session.execute(
                    text("""
                    SELECT 
                        COUNT(*) as total_messages,
                        AVG(processing_time) as avg_processing_time,
                        AVG(confidence_score) as avg_confidence
                    FROM chat_history 
                    WHERE session_id = :session_id
                    """),
                    {"session_id": session_id}
                )
                stats = result.fetchone()._asdict()

                # Durée de la session
                result = await session.execute(
                    text("""
                    SELECT 
                        created_at, 
                        updated_at,
                        is_active 
                    FROM chat_sessions 
                    WHERE session_id = :session_id
                    """),
                    {"session_id": session_id}
                )
                session_data = result.fetchone()
                if session_data:
                    duration = session_data.updated_at - session_data.created_at
                    stats.update({
                        'duration_hours': duration.total_seconds() / 3600,
                        'is_active': session_data.is_active,
                        'last_activity': session_data.updated_at.isoformat()
                    })
                
                return stats
            
            except Exception as e:
                logger.error(f"Erreur calcul statistiques session: {e}")
                return {}

    async def cleanup_old_sessions(self, days: int = 30, user_id: Optional[str] = None) -> int:
        """Nettoie les anciennes sessions inactives."""
        async with self.async_session() as session:
            try:
                # Construction de la requête de base
                query = """
                UPDATE chat_sessions 
                SET 
                    is_active = false,
                    session_context = jsonb_set(
                        session_context::jsonb,
                        '{metadata,deactivated_at}',
                        to_jsonb(:deactivation_time::text),
                        true
                    ),
                    session_context = jsonb_set(
                        session_context::jsonb,
                        '{metadata,deactivation_reason}',
                        '"automatic_cleanup"',
                        true
                    )
                WHERE 
                    updated_at < :cutoff_date 
                    AND is_active = true
                """
                
                params = {
                    "cutoff_date": datetime.utcnow() - timedelta(days=days),
                    "deactivation_time": datetime.utcnow().isoformat()
                }

                # Ajout du filtre utilisateur si spécifié
                if user_id:
                    query += " AND user_id = :user_id"
                    params["user_id"] = user_id

                result = await session.execute(text(query), params)
                await session.commit()
                
                deactivated_count = result.rowcount
                logger.info(f"Nettoyage: {deactivated_count} sessions désactivées")
                return deactivated_count

            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur nettoyage sessions: {e}")
                return 0
