# core/database/models.py
from sqlalchemy import Column, String, DateTime, Float, Integer, Boolean, ForeignKey, Text, Index, func, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship, validates
from sqlalchemy.schema import CheckConstraint
from sqlalchemy.ext.hybrid import hybrid_property
from datetime import datetime
import uuid
import re

from .base import Base
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger(__name__)

class User(Base):
    """Modèle utilisateur."""
    __tablename__ = 'users'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))
    metadata = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)

    # Relations
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    chat_history = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")

    # Index
    __table_args__ = (
        Index('idx_user_email', email),
        Index('idx_user_username', username),
        Index('idx_user_active', is_active),
        Index('idx_user_metadata', metadata, postgresql_using='gin'),
        CheckConstraint('length(username) >= 3', name='username_length_check')
    )

    @validates('email')
    def validate_email(self, key, email):
        """Valide le format de l'email."""
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            raise ValueError("Format d'email invalide")
        return email.lower()

    @validates('username')
    def validate_username(self, key, username):
        """Valide le format du nom d'utilisateur."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", username):
            raise ValueError("Le nom d'utilisateur ne peut contenir que des lettres, chiffres, tirets et underscores")
        return username

class ChatSession(Base):
    """Modèle de session de chat."""
    __tablename__ = 'chat_sessions'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(UUID, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    session_id = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    context = Column(JSONB, default={})
    metadata = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)

    # Relations
    user = relationship("User", back_populates="sessions")
    chat_history = relationship("ChatHistory", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_session_user', user_id),
        Index('idx_session_active', is_active),
        Index('idx_session_updated', updated_at.desc()),
        Index('idx_session_context', context, postgresql_using='gin'),
        Index('idx_session_metadata', metadata, postgresql_using='gin')
    )

    @hybrid_property
    def duration(self):
        """Calcule la durée de la session."""
        if not self.created_at:
            return 0
        end_time = self.updated_at or datetime.utcnow()
        return (end_time - self.created_at).total_seconds()

class ChatHistory(Base):
    """Modèle d'historique des chats."""
    __tablename__ = 'chat_history'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(255), ForeignKey('chat_sessions.session_id', ondelete='CASCADE'), nullable=False)
    user_id = Column(UUID, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    query_vector = Column(ARRAY(Float(precision=6)), nullable=True)
    response_vector = Column(ARRAY(Float(precision=6)), nullable=True)
    confidence_score = Column(Float, nullable=False, default=0.0)
    tokens_used = Column(Integer, nullable=False, default=0)
    processing_time = Column(Float, nullable=False, default=0.0)
    cost = Column(Float, default=0.0)
    metadata = Column(JSONB, nullable=False, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    user = relationship("User", back_populates="chat_history")
    session = relationship("ChatSession", back_populates="chat_history")
    referenced_documents = relationship("ReferencedDocument", back_populates="chat_history", cascade="all, delete-orphan")
    embeddings = relationship("MessageEmbedding", back_populates="message", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_chat_history_session', session_id, created_at.desc()),
        Index('idx_chat_history_user', user_id, created_at.desc()),
        Index('idx_chat_history_confidence', confidence_score.desc()),
        Index('idx_chat_history_metadata', metadata, postgresql_using='gin'),
        Index('idx_chat_history_query_text', func.to_tsvector('french', query), postgresql_using='gin'),
        Index('idx_chat_history_response_text', func.to_tsvector('french', response), postgresql_using='gin'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='confidence_score_check'),
        CheckConstraint('tokens_used >= 0', name='tokens_used_check'),
        CheckConstraint('processing_time >= 0', name='processing_time_check')
    )

    @validates('query_vector', 'response_vector')
    def validate_vector(self, key, vector):
        """Valide les dimensions des vecteurs."""
        if vector is not None and len(vector) != settings.ELASTICSEARCH_EMBEDDING_DIM:
            raise ValueError(f"Dimension du vecteur incorrecte. Attendu: {settings.ELASTICSEARCH_EMBEDDING_DIM}")
        return vector

class ReferencedDocument(Base):
    """Modèle pour les documents référencés."""
    __tablename__ = 'referenced_documents'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_history_id = Column(UUID, ForeignKey('chat_history.id', ondelete='CASCADE'), nullable=False)
    document_name = Column(String(255), nullable=False)
    page_number = Column(Integer)
    relevance_score = Column(Float)
    content_snippet = Column(Text)
    metadata = Column(JSONB, default={})
    vector_id = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    chat_history = relationship("ChatHistory", back_populates="referenced_documents")

    __table_args__ = (
        Index('idx_ref_doc_chat_history', chat_history_id),
        Index('idx_ref_doc_relevance', relevance_score.desc()),
        Index('idx_ref_doc_metadata', metadata, postgresql_using='gin'),
        CheckConstraint('relevance_score >= 0 AND relevance_score <= 1', name='relevance_score_check')
    )

class MessageEmbedding(Base):
    """Modèle pour les embeddings de messages."""
    __tablename__ = 'message_embeddings'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id = Column(UUID, ForeignKey('chat_history.id', ondelete='CASCADE'), nullable=False)
    embedding_type = Column(String(50), nullable=False)  # 'query' ou 'response'
    vector = Column(ARRAY(Float(precision=6)), nullable=False)
    model_version = Column(String(100))
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    message = relationship("ChatHistory", back_populates="embeddings")

    __table_args__ = (
        Index('idx_embedding_message', message_id),
        Index('idx_embedding_type', embedding_type),
        Index('idx_embedding_metadata', metadata, postgresql_using='gin')
    )

    @validates('vector')
    def validate_vector(self, key, vector):
        """Valide les dimensions du vecteur."""
        if len(vector) != settings.ELASTICSEARCH_EMBEDDING_DIM:
            raise ValueError(f"Dimension du vecteur incorrecte. Attendu: {settings.ELASTICSEARCH_EMBEDDING_DIM}")
        return vector

class UsageMetric(Base):
    """Modèle pour les métriques d'utilisation."""
    __tablename__ = 'usage_metrics'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(UUID, ForeignKey('users.id', ondelete='CASCADE'))
    session_id = Column(String(255), ForeignKey('chat_sessions.session_id', ondelete='CASCADE'))
    endpoint_path = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    response_time = Column(Float)
    status_code = Column(Integer)
    error_message = Column(Text)
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_metric_user', user_id),
        Index('idx_metric_session', session_id),
        Index('idx_metric_created', created_at.desc()),
        Index('idx_metric_status', status_code),
        Index('idx_metric_endpoint', endpoint_path),
        CheckConstraint('response_time >= 0', name='response_time_check')
    )

class SystemConfig(Base):
    """Modèle pour la configuration système."""
    __tablename__ = 'system_config'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String(255), unique=True, nullable=False)
    value = Column(JSONB, nullable=False)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    metadata = Column(JSONB, default={})

    __table_args__ = (
        Index('idx_config_key', key),
        Index('idx_config_active', is_active),
        Index('idx_config_metadata', metadata, postgresql_using='gin')
    )

    @classmethod
    async def get_value(cls, key: str, default=None):
        """Récupère une valeur de configuration."""
        from sqlalchemy.future import select
        from .base import get_session_manager
        
        async with get_session_manager().get_session() as session:
            result = await session.execute(
                select(cls).where(cls.key == key, cls.is_active == True)
            )
            config = result.scalar_one_or_none()
            return config.value if config else default
        }
