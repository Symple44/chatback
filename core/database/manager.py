# core/database/manager.py
from typing import List, Dict, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import uuid
import os
from sqlalchemy import select, and_, or_, func, desc, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from .models import (
    User, ChatSession, ChatHistory, ReferencedDocument, 
    MessageEmbedding, UsageMetric, SystemConfig, ErrorLog
)
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config import settings

logger = get_logger(__name__)

class DatabaseManager:
    def __init__(self, session_factory):
        """Initialise le gestionnaire de base de données."""
        self.session_factory = session_factory

    async def create_user(self, user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Crée un nouvel utilisateur.
        
        Args:
            user_data: Données de l'utilisateur
            
        Returns:
            Données de l'utilisateur créé ou None en cas d'erreur
        """
        async with self.session_factory() as session:
            try:
                # Vérification de l'unicité
                existing = await session.execute(
                    select(User).where(
                        or_(
                            User.email == user_data["email"],
                            User.username == user_data["username"]
                        )
                    )
                )
                if existing.scalar_one_or_none():
                    raise ValueError("Email ou username déjà utilisé")

                # Création de l'utilisateur
                user = User(
                    id=uuid.uuid4(),
                    email=user_data["email"],
                    username=user_data["username"],
                    full_name=user_data["full_name"],
                    metadata=user_data.get("metadata", {})
                )
                session.add(user)
                await session.commit()
                await session.refresh(user)
                
                metrics.increment_counter("database_user_creations")
                return user.to_dict()

            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur création utilisateur: {e}")
                metrics.increment_counter("database_user_creation_errors")
                return None
    
    async def save_chat_interaction(
        self,
        session_id: str,
        user_id: str,
        query: str,
        response: str,
        query_vector: List[float],
        response_vector: List[float],
        confidence_score: float,
        tokens_used: int,
        processing_time: float,
        referenced_docs: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Sauvegarde une interaction de chat complète.
        
        Args:
            session_id: ID de la session
            user_id: ID de l'utilisateur
            query: Question de l'utilisateur
            response: Réponse du système
            query_vector: Vecteur de la question
            response_vector: Vecteur de la réponse
            confidence_score: Score de confiance
            tokens_used: Nombre de tokens utilisés
            processing_time: Temps de traitement
            referenced_docs: Documents référencés
            metadata: Métadonnées additionnelles
            
        Returns:
            ID de l'historique créé ou None en cas d'erreur
        """
        async with self.session_factory() as session:
            try:
                # Création de l'historique
                chat_history = ChatHistory(
                    id=uuid.uuid4(),
                    session_id=session_id,
                    user_id=user_id,
                    query=query,
                    response=response,
                    query_vector=query_vector,
                    response_vector=response_vector,
                    confidence_score=confidence_score,
                    tokens_used=tokens_used,
                    processing_time=processing_time,
                    metadata=metadata or {}
                )
                session.add(chat_history)

                # Ajout des documents référencés
                if referenced_docs:
                    for doc in referenced_docs:
                        ref_doc = ReferencedDocument(
                            id=uuid.uuid4(),
                            chat_history_id=chat_history.id,
                            document_name=doc["name"],
                            page_number=doc.get("page"),
                            relevance_score=doc.get("score", 0.0),
                            content_snippet=doc.get("snippet"),
                            metadata=doc.get("metadata", {})
                        )
                        session.add(ref_doc)

                # Création des embeddings
                for vec_type, vector in [("query", query_vector), ("response", response_vector)]:
                    if vector:
                        embedding = MessageEmbedding(
                            id=uuid.uuid4(),
                            message_id=chat_history.id,
                            embedding_type=vec_type,
                            vector=vector,
                            model_version=settings.VERSION,
                            metadata={"timestamp": datetime.utcnow().isoformat()}
                        )
                        session.add(embedding)

                await session.commit()
                metrics.increment_counter("database_chat_saves")
                return str(chat_history.id)

            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur sauvegarde chat: {e}")
                metrics.increment_counter("database_chat_save_errors")
                return None

    async def find_similar_questions(
        self,
        vector: List[float],
        threshold: float = 0.8,
        limit: int = 5,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        try:
            async with self.session_factory() as session:
                vector_str = "[" + ",".join(map(str, vector)) + "]"
                
                query = text("""
                    WITH scored_vectors AS (
                        SELECT 
                            ch.query,
                            ch.response,
                            1 - (ch.query_vector <-> cast(:vector as vector)) as similarity
                        FROM chat_history ch
                        WHERE 1 - (ch.query_vector <-> cast(:vector as vector)) > :threshold
                        ORDER BY similarity DESC
                        LIMIT :limit
                    )
                    SELECT * FROM scored_vectors
                """)
                
                result = await session.execute(
                    query,
                    {
                        "vector": vector_str,
                        "threshold": threshold,
                        "limit": limit
                    }
                )
                
                return [
                    {
                        "query": row.query,
                        "response": row.response,
                        "similarity": row.similarity,
                        "timestamp": datetime.utcnow(),
                        "user_id": None,
                        "session_id": None
                    } 
                    for row in result
                ]

        except Exception as e:
            logger.error(f"Erreur recherche similarité: {e}")
            return []

    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Récupère les statistiques complètes d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Statistiques de l'utilisateur
        """
        async with self.session_factory() as session:
            try:
                # Statistiques des sessions
                session_stats = await session.execute(
                    select(
                        func.count(ChatSession.id).label("total_sessions"),
                        func.count(ChatSession.id).filter(
                            ChatSession.is_active == True
                        ).label("active_sessions")
                    ).where(ChatSession.user_id == user_id)
                )
                session_stats = session_stats.first()

                # Statistiques des messages
                message_stats = await session.execute(
                    select(
                        func.count(ChatHistory.id).label("total_messages"),
                        func.avg(ChatHistory.processing_time).label("avg_processing_time"),
                        func.avg(ChatHistory.confidence_score).label("avg_confidence"),
                        func.sum(ChatHistory.tokens_used).label("total_tokens"),
                        func.sum(ChatHistory.cost).label("total_cost")
                    ).where(ChatHistory.user_id == user_id)
                )
                message_stats = message_stats.first()

                return {
                    "sessions": {
                        "total": session_stats.total_sessions,
                        "active": session_stats.active_sessions
                    },
                    "messages": {
                        "total": message_stats.total_messages,
                        "avg_processing_time": round(message_stats.avg_processing_time, 3) if message_stats.avg_processing_time else 0,
                        "avg_confidence": round(message_stats.avg_confidence, 3) if message_stats.avg_confidence else 0,
                        "total_tokens": message_stats.total_tokens or 0,
                        "total_cost": round(message_stats.total_cost or 0, 4)
                    },
                    "last_updated": datetime.utcnow().isoformat()
                }

            except Exception as e:
                logger.error(f"Erreur récupération statistiques: {e}")
                return {}

    async def cleanup_old_data(self, days: int = 30) -> Tuple[int, int, int]:
        """
        Nettoie les anciennes données.
        
        Args:
            days: Âge maximum des données en jours
            
        Returns:
            Tuple contenant le nombre d'éléments supprimés (sessions, messages, métriques)
        """
        async with self.session_factory() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Nettoyage des anciennes sessions
                sessions = await session.execute(
                    select(ChatSession).where(
                        and_(
                            ChatSession.updated_at < cutoff_date,
                            ChatSession.is_active == False
                        )
                    )
                )
                deleted_sessions = len(sessions.all())
                
                # Nettoyage de l'historique
                history = await session.execute(
                    select(ChatHistory).where(
                        ChatHistory.created_at < cutoff_date
                    )
                )
                deleted_history = len(history.all())
                
                # Nettoyage des métriques
                metrics = await session.execute(
                    select(UsageMetric).where(
                        UsageMetric.created_at < cutoff_date
                    )
                )
                deleted_metrics = len(metrics.all())
                
                await session.commit()
                logger.info(f"Nettoyage terminé: {deleted_sessions} sessions, {deleted_history} messages, {deleted_metrics} métriques")
                
                return deleted_sessions, deleted_history, deleted_metrics

            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur nettoyage données: {e}")
                return 0, 0, 0

    async def get_system_config(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur de configuration système.
        
        Args:
            key: Clé de configuration
            
        Returns:
            Valeur de configuration ou None
        """
        async with self.session_factory() as session:
            try:
                result = await session.execute(
                    select(SystemConfig)
                    .where(
                        and_(
                            SystemConfig.key == key,
                            SystemConfig.is_active == True
                        )
                    )
                )
                config = result.scalar_one_or_none()
                return config.value if config else None

            except Exception as e:
                logger.error(f"Erreur récupération configuration: {e}")
                return None

    async def set_system_config(
        self,
        key: str,
        value: Any,
        description: Optional[str] = None
    ) -> bool:
        """
        Définit une valeur de configuration système.
        
        Args:
            key: Clé de configuration
            value: Valeur à définir
            description: Description optionnelle
            
        Returns:
            Succès de l'opération
        """
        async with self.session_factory() as session:
            try:
                config = await session.execute(
                    select(SystemConfig).where(SystemConfig.key == key)
                )
                config = config.scalar_one_or_none()

                if config:
                    config.value = value
                    config.description = description or config.description
                    config.updated_at = datetime.utcnow()
                else:
                    config = SystemConfig(
                        key=key,
                        value=value,
                        description=description
                    )
                    session.add(config)

                await session.commit()
                return True

            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur définition configuration: {e}")
                return False
    
    async def initialize_user_resources(self, user_id: str) -> bool:
        """Initialise les ressources pour un nouvel utilisateur."""
        try:
            # Création du répertoire utilisateur
            user_dir = f"data/users/{user_id}"
            os.makedirs(user_dir, exist_ok=True)
            
            # Configuration des préférences par défaut
            async with self.session_factory() as session:
                user = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = user.scalar_one_or_none()
                if user:
                    if not user.user_metadata:
                        user.user_metadata = {}
                    user.user_metadata.update({
                        "preferences": {
                            "theme": "light",
                            "language": "fr",
                            "notifications": True
                        },
                        "resources": {
                            "user_dir": user_dir
                        }
                    })
                    await session.commit()
                    
                logger.info(f"Ressources initialisées pour l'utilisateur {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des ressources: {e}")
            return False

    async def calculate_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Calcule les statistiques complètes d'un utilisateur."""
        async with self.session_factory() as session:
            try:
                # Statistiques des sessions
                session_stats = await session.execute(
                    select(
                        func.count(ChatSession.id).label('total_sessions'),
                        func.count(ChatSession.id).filter(
                            ChatSession.is_active == True
                        ).label('active_sessions')
                    ).where(ChatSession.user_id == user_id)
                )
                session_stats = session_stats.first()

                # Statistiques des messages
                message_stats = await session.execute(
                    select(
                        func.count(ChatHistory.id).label('total_messages'),
                        func.avg(ChatHistory.processing_time).label('avg_processing_time'),
                        func.avg(ChatHistory.confidence_score).label('avg_confidence')
                    ).where(ChatHistory.user_id == user_id)
                )
                message_stats = message_stats.first()

                # Récupération des informations utilisateur
                user = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = user.scalar_one_or_none()

                if user:
                    return {
                        "sessions": {
                            "total": session_stats.total_sessions,
                            "active": session_stats.active_sessions
                        },
                        "messages": {
                            "total": message_stats.total_messages,
                            "avg_processing_time": round(message_stats.avg_processing_time, 3) if message_stats.avg_processing_time else 0,
                            "avg_confidence": round(message_stats.avg_confidence, 3) if message_stats.avg_confidence else 0
                        },
                        "account": {
                            "age_days": (datetime.utcnow() - user.created_at).days,
                            "last_login": user.last_login.isoformat() if user.last_login else None,
                            "is_active": user.is_active
                        }
                    }
                return {}
                
            except Exception as e:
                logger.error(f"Erreur lors du calcul des statistiques: {e}")
                return {}

    async def cleanup_user_data(self, user_id: str) -> bool:
        """Nettoie toutes les données associées à un utilisateur."""
        async with self.session_factory() as session:
            try:
                # Suppression des sessions
                await session.execute(
                    delete(ChatSession).where(ChatSession.user_id == user_id)
                )
                
                # Suppression de l'historique
                await session.execute(
                    delete(ChatHistory).where(ChatHistory.user_id == user_id)
                )
                
                # Suppression du répertoire utilisateur
                user_dir = f"data/users/{user_id}"
                if os.path.exists(user_dir):
                    import shutil
                    shutil.rmtree(user_dir)
                
                await session.commit()    
                logger.info(f"Données utilisateur nettoyées: {user_id}")
                return True
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur lors du nettoyage des données: {e}")
                return False
                
    async def log_error(
        self,
        error_type: str,
        error_message: str,
        endpoint: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Log une erreur dans la base de données."""
        try:
            async with self.session_factory() as session:
                error_log = ErrorLog(
                    error_type=error_type,
                    error_message=error_message,
                    component=endpoint,
                    user_id=user_id,
                    error_metadata=metadata or {}
                )
                session.add(error_log)
                await session.commit()
        except Exception as e:
            logger.error(f"Erreur lors du log d'erreur: {e}")
            
