# core/llm/model.py
from sqlalchemy import (
    Column, String, DateTime, Float, Integer, Boolean, 
    ForeignKey, Text, Index, func, UUID, JSON, ARRAY
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, validates
from sqlalchemy.schema import CheckConstraint
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
            "metadata": self.metadata,
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
    metadata = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)

    # Relations
    user = relationship("User", back_populates="sessions")
    chat_history = relationship("ChatHistory", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_session_user', user_id),
        Index('idx_session_active', is_active),
        Index('idx_session_updated', updated_at.desc()),
    )

    def to_dict(self):
        """Convertit l'objet en dictionnaire."""
        return {
            "session_id": self.session_id,
            "user_id": str(self.user_id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "session_context": self.session_context,
            "metadata": self.metadata,
            "is_active": self.is_active
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
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    user = relationship("User", back_populates="chat_history")
    session = relationship("ChatSession", back_populates="chat_history")
    referenced_documents = relationship("ReferencedDocument", back_populates="chat_history", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_history_session', session_id),
        Index('idx_history_user', user_id),
        Index('idx_history_created', created_at.desc()),
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
            "metadata": self.metadata,
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
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    chat_history = relationship("ChatHistory", back_populates="referenced_documents")

    __table_args__ = (
        Index('idx_ref_doc_chat', chat_history_id),
        Index('idx_ref_doc_relevance', relevance_score.desc()),
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
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
