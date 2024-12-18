# core/database/models.py
from sqlalchemy import Column, String, DateTime, Float, Integer, Boolean, ForeignKey, Text, Index, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from core.config import settings
from .base import Base

class User(Base):
    __tablename__ = 'users'
    
    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))
    preferences = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)

    # Relations
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    chat_history = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_user_email', email),
        Index('idx_user_username', username),
        Index('idx_user_active', is_active),
    )

    def to_dict(self):
        return {
            'id': str(self.id),
            'email': self.email,
            'username': self.username,
            'full_name': self.full_name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    
    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(UUID, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    session_id = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    session_context = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)
    session_metadata = Column(JSONB, default={})

    # Relations
    user = relationship("User", back_populates="sessions")
    chat_history = relationship("ChatHistory", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_session_user', user_id),
        Index('idx_session_active', is_active),
        Index('idx_session_updated', updated_at.desc()),
        Index('idx_session_context', session_context, postgresql_using='gin'),
    )

    def to_dict(self):
        return {
            'id': str(self.id),
            'user_id': str(self.user_id),
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_active': self.is_active,
            'session_metadata': self.session_metadata
        }

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    
    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(255), ForeignKey('chat_sessions.session_id', ondelete='CASCADE'), nullable=False)
    user_id = Column(UUID, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    query_vector = Column(Vector(settings.ELASTICSEARCH_EMBEDDING_DIM))
    response_vector = Column(Vector(settings.ELASTICSEARCH_EMBEDDING_DIM))
    confidence_score = Column(Float, nullable=False, default=0.0)
    tokens_used = Column(Integer, nullable=False, default=0)
    processing_time = Column(Float, nullable=False, default=0.0)
    additional_data = Column(JSONB, nullable=False, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    user = relationship("User", back_populates="chat_history")
    session = relationship("ChatSession", back_populates="chat_history")
    referenced_documents = relationship("ReferencedDocument", back_populates="chat_history", cascade="all, delete-orphan")

    __table_args__ = (
        Index(
            'idx_chat_history_query_vector',
            query_vector,
            postgresql_using='ivfflat',
            postgresql_with={'lists': '100'},
            postgresql_ops={'query_vector': 'vector_cosine_ops'}
        ),
        Index(
            'idx_chat_history_response_vector',
            response_vector,
            postgresql_using='ivfflat',
            postgresql_with={'lists': '100'},
            postgresql_ops={'response_vector': 'vector_cosine_ops'}
        ),
        Index('idx_chat_history_session_created', session_id, created_at.desc()),
        Index('idx_chat_history_user_created', user_id, created_at.desc()),
        Index('idx_chat_history_confidence', confidence_score.desc()),
        Index('idx_chat_history_additional_data', additional_data, postgresql_using='gin'),
        Index('idx_chat_history_query_text', query, postgresql_using='gin'),
        Index('idx_chat_history_response_text', response, postgresql_using='gin'),
    )

    def to_dict(self):
        return {
            'id': str(self.id),
            'session_id': self.session_id,
            'user_id': str(self.user_id),
            'query': self.query,
            'response': self.response,
            'confidence_score': round(self.confidence_score, 4),
            'tokens_used': self.tokens_used,
            'processing_time': round(self.processing_time, 4),
            'additional_data': self.additional_data,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class ReferencedDocument(Base):
    __tablename__ = 'referenced_documents'
    
    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_history_id = Column(UUID, ForeignKey('chat_history.id', ondelete='CASCADE'), nullable=False)
    document_name = Column(String(255), nullable=False)
    page_number = Column(Integer)
    relevance_score = Column(Float)
    content_snippet = Column(Text)
    metadata = Column(JSONB, default={})

    # Relations
    chat_history = relationship("ChatHistory", back_populates="referenced_documents")

    __table_args__ = (
        Index('idx_ref_doc_chat_history', chat_history_id),
        Index('idx_ref_doc_relevance', relevance_score.desc()),
        Index('idx_ref_doc_metadata', metadata, postgresql_using='gin'),
    )

    def to_dict(self):
        return {
            'id': str(self.id),
            'chat_history_id': str(self.chat_history_id),
            'document_name': self.document_name,
            'page_number': self.page_number,
            'relevance_score': round(self.relevance_score, 4) if self.relevance_score else None,
            'content_snippet': self.content_snippet,
            'metadata': self.metadata
        }

class UsageMetric(Base):
    __tablename__ = 'usage_metrics'
    
    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(UUID, ForeignKey('users.id', ondelete='CASCADE'))
    session_id = Column(String(255), ForeignKey('chat_sessions.session_id', ondelete='CASCADE'))
    endpoint_path = Column(String(255), nullable=False)
    response_time = Column(Float)
    status_code = Column(Integer)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_metric_user', user_id),
        Index('idx_metric_session', session_id),
        Index('idx_metric_created', created_at.desc()),
        Index('idx_metric_status', status_code),
        Index('idx_metric_endpoint', endpoint_path),
    )

    def to_dict(self):
        return {
            'id': str(self.id),
            'user_id': str(self.user_id) if self.user_id else None,
            'session_id': self.session_id,
            'endpoint_path': self.endpoint_path,
            'response_time': round(self.response_time, 4) if self.response_time else None,
            'status_code': self.status_code,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }