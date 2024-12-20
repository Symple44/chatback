# core/database/models.py
from sqlalchemy import (
    Column, String, DateTime, Float, Integer, Boolean, 
    ForeignKey, Text, Index, func, UUID, JSON, ARRAY,
    UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import relationship, validates, declarative_base
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
    user_metadata = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)

    # Relations
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    chat_history = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")
    usage_metrics = relationship("UsageMetric", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")

    # Index
    __table_args__ = (
        Index('idx_user_email', email),
        Index('idx_user_username', username),
        Index('idx_user_active', is_active),
        Index('idx_user_metadata', user_metadata, postgresql_using='gin'),
        CheckConstraint('length(username) >= 3', name='username_length_check'),
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

    def to_dict(self):
        """Convertit l'objet en dictionnaire."""
        return {
            "id": str(self.id),
            "email": self.email,
            "username": self.username,
            "full_name": self.full_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "metadata": self.user_metadata,
            "is_active": self.is_active
        }

class ChatSession(Base):
    """Modèle de session de chat."""
    __tablename__ = 'chat_sessions'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(UUID, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    session_id = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    session_context = Column(JSONB, default={})
    session_metadata  = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)

    # Relations
    user = relationship("User", back_populates="sessions")
    chat_history = relationship("ChatHistory", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_session_user', user_id),
        Index('idx_session_active', is_active),
        Index('idx_session_updated', updated_at.desc()),
        Index('idx_session_context', session_context, postgresql_using='gin'),
        Index('idx_session_metadata', session_metadata, postgresql_using='gin'),
    )

    @hybrid_property
    def duration(self):
        """Calcule la durée de la session."""
        if not self.created_at:
            return 0
        end_time = self.updated_at or datetime.utcnow()
        return (end_time - self.created_at).total_seconds()

    def to_dict(self):
        """Convertit l'objet en dictionnaire."""
        return {
            "session_id": self.session_id,
            "user_id": str(self.user_id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "session_context": self.session_context,
            "metadata": self.session_metadata,
            "is_active": self.is_active,
            "duration": self.duration
        }

class ChatHistory(Base):
    """Modèle d'historique des chats."""
    __tablename__ = 'chat_history'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(255), ForeignKey('chat_sessions.session_id', ondelete='CASCADE'), nullable=False)
    user_id = Column(UUID, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    query_vector = Column(ARRAY(Float))
    response_vector = Column(ARRAY(Float))
    confidence_score = Column(Float, nullable=False, default=0.0)
    tokens_used = Column(Integer, nullable=False, default=0)
    processing_time = Column(Float, nullable=False, default=0.0)
    cost = Column(Float, default=0.0)
    chat_metadata  = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    user = relationship("User", back_populates="chat_history")
    session = relationship("ChatSession", back_populates="chat_history")
    referenced_documents = relationship("ReferencedDocument", back_populates="chat_history", cascade="all, delete-orphan")
    metrics = relationship("MessageMetrics", back_populates="message", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_history_session', session_id),
        Index('idx_history_user', user_id),
        Index('idx_history_created', created_at.desc()),
        Index('idx_history_metadata', chat_metadata, postgresql_using='gin'),
        Index('idx_history_vector', query_vector, postgresql_using='gin'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='confidence_score_check'),
        CheckConstraint('tokens_used >= 0', name='tokens_used_check'),
        CheckConstraint('processing_time >= 0', name='processing_time_check'),
    )

    def to_dict(self):
        """Convertit l'objet en dictionnaire."""
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "user_id": str(self.user_id),
            "query": self.query,
            "response": self.response,
            "confidence_score": self.confidence_score,
            "tokens_used": self.tokens_used,
            "processing_time": self.processing_time,
            "cost": self.cost,
            "metadata": self.chat_metadata,
            "created_at": self.created_at.isoformat()
        }

class ReferencedDocument(Base):
    """Modèle pour les documents référencés."""
    __tablename__ = 'referenced_documents'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_history_id = Column(UUID, ForeignKey('chat_history.id', ondelete='CASCADE'), nullable=False)
    document_name = Column(String(255), nullable=False)
    page_number = Column(Integer)
    relevance_score = Column(Float)
    content_snippet = Column(Text)
    doc_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    chat_history = relationship("ChatHistory", back_populates="referenced_documents")

    __table_args__ = (
        Index('idx_ref_doc_chat', chat_history_id),
        Index('idx_ref_doc_relevance', relevance_score.desc()),
        Index('idx_ref_doc_metadata', doc_metadata, postgresql_using='gin'),
        CheckConstraint('relevance_score >= 0 AND relevance_score <= 1', name='relevance_score_check'),
    )

    def to_dict(self):
        """Convertit l'objet en dictionnaire."""
        return {
            "id": str(self.id),
            "chat_history_id": str(self.chat_history_id),
            "document_name": self.document_name,
            "page_number": self.page_number,
            "relevance_score": self.relevance_score,
            "content_snippet": self.content_snippet,
            "metadata": self.doc_metadata,
            "created_at": self.created_at.isoformat()
        }

class MessageMetrics(Base):
    """Modèle pour les métriques des messages."""
    __tablename__ = 'message_metrics'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id = Column(UUID, ForeignKey('chat_history.id', ondelete='CASCADE'), unique=True)
    tokens_prompt = Column(Integer, default=0)
    tokens_completion = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    prompt_cost = Column(Float, default=0.0)
    completion_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    model_name = Column(String(100))
    metrics_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    message = relationship("ChatHistory", back_populates="metrics")

    __table_args__ = (
        Index('idx_metrics_message', message_id),
        Index('idx_metrics_model', model_name),
        Index('idx_metrics_metadata', metrics_metadata, postgresql_using='gin'),
        CheckConstraint('tokens_prompt >= 0', name='tokens_prompt_check'),
        CheckConstraint('tokens_completion >= 0', name='tokens_completion_check'),
        CheckConstraint('total_tokens >= 0', name='total_tokens_check'),
        CheckConstraint('prompt_cost >= 0', name='prompt_cost_check'),
        CheckConstraint('completion_cost >= 0', name='completion_cost_check'),
        CheckConstraint('total_cost >= 0', name='total_cost_check')
    )

    def to_dict(self):
        """Convertit l'objet en dictionnaire."""
        return {
            "id": str(self.id),
            "message_id": str(self.message_id),
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "total_tokens": self.total_tokens,
            "prompt_cost": self.prompt_cost,
            "completion_cost": self.completion_cost,
            "total_cost": self.total_cost,
            "model_name": self.model_name,
            "metadata": self.metrics_metadata,
            "created_at": self.created_at.isoformat()
        }

class UsageMetric(Base):
    """Modèle pour les métriques d'utilisation."""
    __tablename__ = 'usage_metrics'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(UUID, ForeignKey('users.id', ondelete='CASCADE'))
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer)
    response_time = Column(Float)
    request_size = Column(Integer)
    response_size = Column(Integer)
    user_agent = Column(String(255))
    ip_address = Column(String(45))
    error_message = Column(Text)
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    user = relationship("User", back_populates="usage_metrics")

    __table_args__ = (
        Index('idx_usage_user', user_id),
        Index('idx_usage_endpoint', endpoint),
        Index('idx_usage_created', created_at.desc()),
        Index('idx_usage_metadata', metadata, postgresql_using='gin'),
        CheckConstraint('response_time >= 0', name='response_time_check'),
    )

class APIKey(Base):
    """Modèle pour les clés API."""
    __tablename__ = 'api_keys'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(UUID, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    key_hash = Column(String(255), nullable=False)
    name = Column(String(100))
    scopes = Column(ARRAY(String))
    expires_at = Column(DateTime(timezone=True))
    last_used_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)
    metadata = Column(JSONB, default={})

    # Relations
    user = relationship("User", back_populates="api_keys")

    __table_args__ = (
        Index('idx_api_key_user', user_id),
        Index('idx_api_key_hash', key_hash),
        Index('idx_api_key_active', is_active),
    )

    @property
    def is_expired(self):
        """Vérifie si la clé API est expirée."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

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
        Index('idx_config_metadata', metadata, postgresql_using='gin'),
    )

class DocumentIndex(Base):
    """Modèle pour l'index des documents."""
    __tablename__ = 'document_index'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_name = Column(String(255), nullable=False)
    document_type = Column(String(50), nullable=False)
    document_hash = Column(String(255), nullable=False)
    content_vector = Column(ARRAY(Float))
    metadata = Column(JSONB, default={})
    indexed_at = Column(DateTime(timezone=True), server_default=func.now())
    last_accessed = Column(DateTime(timezone=True))
    status = Column(String(50), default='active')

    __table_args__ = (
        Index('idx_doc_name', document_name),
        Index('idx_doc_type', document_type),
        Index('idx_doc_hash', document_hash),
        Index('idx_doc_status', status),
        Index('idx_doc_metadata', metadata, postgresql_using='gin'),
        UniqueConstraint('document_name', 'document_hash', name='uq_doc_name_hash')
    )

    def to_dict(self):
        return {
            "id": str(self.id),
            "document_name": self.document_name,
            "document_type": self.document_type,
            "document_hash": self.document_hash,
            "metadata": self.metadata,
            "indexed_at": self.indexed_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "status": self.status
        }

class VectorEmbedding(Base):
    """Modèle pour les embeddings vectoriels."""
    __tablename__ = 'vector_embeddings'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id = Column(UUID, nullable=False)  # ID du document ou message source
    source_type = Column(String(50), nullable=False)  # 'document' ou 'message'
    embedding = Column(ARRAY(Float), nullable=False)
    model_name = Column(String(100), nullable=False)
    dimensions = Column(Integer, nullable=False)
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_vector_source', source_id, source_type),
        Index('idx_vector_model', model_name),
        Index('idx_vector_metadata', metadata, postgresql_using='gin'),
        CheckConstraint('dimensions > 0', name='dimensions_check')
    )

class UserPreference(Base):
    """Modèle pour les préférences utilisateur."""
    __tablename__ = 'user_preferences'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(UUID, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    language = Column(String(5), default='fr')
    theme = Column(String(20), default='light')
    notifications_enabled = Column(Boolean, default=True)
    email_notifications = Column(Boolean, default=True)
    custom_settings = Column(JSONB, default={})
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index('idx_pref_user', user_id),
        CheckConstraint(
            "language IN ('fr', 'en', 'es', 'de', 'it')", 
            name='language_check'
        ),
        CheckConstraint(
            "theme IN ('light', 'dark', 'system')", 
            name='theme_check'
        )
    )

class DocumentChunk(Base):
    """Modèle pour les chunks de documents."""
    __tablename__ = 'document_chunks'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(UUID, ForeignKey('document_index.id', ondelete='CASCADE'), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_size = Column(Integer, nullable=False)
    embedding = Column(ARRAY(Float))
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_chunk_doc', document_id),
        Index('idx_chunk_index', chunk_index),
        Index('idx_chunk_metadata', metadata, postgresql_using='gin'),
        CheckConstraint('chunk_size > 0', name='chunk_size_check')
    )

class ErrorLog(Base):
    """Modèle pour les logs d'erreurs."""
    __tablename__ = 'error_logs'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    error_type = Column(String(100), nullable=False)
    error_message = Column(Text, nullable=False)
    stack_trace = Column(Text)
    component = Column(String(100))
    severity = Column(String(20), default='error')
    user_id = Column(UUID, ForeignKey('users.id', ondelete='SET NULL'))
    session_id = Column(String(255))
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_error_type', error_type),
        Index('idx_error_severity', severity),
        Index('idx_error_component', component),
        Index('idx_error_created', created_at.desc()),
        CheckConstraint(
            "severity IN ('debug', 'info', 'warning', 'error', 'critical')", 
            name='severity_check'
        )
    )

class BackgroundJob(Base):
    """Modèle pour les tâches en arrière-plan."""
    __tablename__ = 'background_jobs'

    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_type = Column(String(100), nullable=False)
    status = Column(String(50), default='pending')
    progress = Column(Float, default=0.0)
    total_items = Column(Integer)
    processed_items = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    metadata = Column(JSONB, default={})
    error_details = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_job_type', job_type),
        Index('idx_job_status', status),
        Index('idx_job_created', created_at.desc()),
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')", 
            name='status_check'
        ),
        CheckConstraint('progress >= 0 AND progress <= 100', name='progress_check')
    )

    @property
    def duration(self):
        """Calcule la durée du job en secondes."""
        if not self.started_at:
            return 0
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()

    @property
    def progress_rate(self):
        """Calcule le taux de progression."""
        if not self.total_items:
            return 0
        return (self.processed_items / self.total_items) * 100

# Fonction utilitaire pour les triggers
def create_trigger_function():
    return """
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ language 'plpgsql';
    """

# Création des triggers pour updated_at
def create_updated_at_triggers():
    triggers = []
    for table in [
        'chat_sessions', 'user_preferences', 'system_config'
    ]:
        trigger = f"""
        DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table};
        CREATE TRIGGER update_{table}_updated_at
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """
        triggers.append(trigger)
    return '\n'.join(triggers)
