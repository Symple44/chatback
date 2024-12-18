# core/database/models.py
from sqlalchemy import Column, String, DateTime, Float, Integer, Boolean, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY, FLOAT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from typing import List
import uuid
from datetime import datetime

Base = declarative_base()

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

    sessions = relationship("ChatSession", back_populates="user")
    chat_history = relationship("ChatHistory", back_populates="user")

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    
    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(UUID, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    session_id = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    session_context = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)
    metadata = Column(JSONB, default={})

    user = relationship("User", back_populates="sessions")
    chat_history = relationship("ChatHistory", back_populates="session")

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    
    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(255), ForeignKey('chat_sessions.session_id', ondelete='CASCADE'))
    user_id = Column(UUID, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    query_vector = Column(ARRAY(FLOAT))  # Pour stocker les vecteurs
    response_vector = Column(ARRAY(FLOAT))
    confidence_score = Column(Float)
    tokens_used = Column(Integer)
    processing_time = Column(Float)
    additional_data = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="chat_history")
    session = relationship("ChatSession", back_populates="chat_history")
    referenced_documents = relationship("ReferencedDocument", back_populates="chat_history")

class ReferencedDocument(Base):
    __tablename__ = 'referenced_documents'
    
    id = Column(UUID, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_history_id = Column(UUID, ForeignKey('chat_history.id', ondelete='CASCADE'))
    document_name = Column(String(255), nullable=False)
    page_number = Column(Integer)
    relevance_score = Column(Float)
    content_snippet = Column(Text)
    metadata = Column(JSONB, default={})

    chat_history = relationship("ChatHistory", back_populates="referenced_documents")

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
