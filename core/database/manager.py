# core/database/manager.py
from typing import List, Dict, Optional, Any
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import numpy as np

from .models import User, ChatSession, ChatHistory, ReferencedDocument, UsageMetric
from core.utils.logger import get_logger

logger = get_logger(__name__)

class DatabaseManager:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def create_user(self, user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with self.session_factory() as session:
            try:
                user = User(
                    email=user_data["email"],
                    username=user_data["username"],
                    full_name=user_data["full_name"],
                    preferences=user_data.get("preferences", {})
                )
                session.add(user)
                await session.commit()
                await session.refresh(user)
                return {
                    "id": user.id,
                    "email": user.email,
                    "username": user.username,
                    "full_name": user.full_name,
                    "created_at": user.created_at
                }
            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur création utilisateur: {e}")
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
        additional_data: Optional[Dict] = None
    ) -> Optional[str]:
        async with self.session_factory() as session:
            try:
                # Création de l'historique
                chat_history = ChatHistory(
                    session_id=session_id,
                    user_id=user_id,
                    query=query,
                    response=response,
                    query_vector=query_vector,
                    response_vector=response_vector,
                    confidence_score=confidence_score,
                    tokens_used=tokens_used,
                    processing_time=processing_time,
                    additional_data=additional_data or {}
                )
                session.add(chat_history)
                await session.flush()

                # Ajout des documents référencés
                if referenced_docs:
                    for doc in referenced_docs:
                        ref_doc = ReferencedDocument(
                            chat_history_id=chat_history.id,
                            document_name=doc["name"],
                            page_number=doc.get("page"),
                            relevance_score=doc.get("score", 0.0),
                            content_snippet=doc.get("snippet"),
                            metadata=doc.get("metadata", {})
                        )
                        session.add(ref_doc)

                await session.commit()
                return str(chat_history.id)

            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur sauvegarde chat: {e}")
                return None

    async def find_similar_questions(self, vector: List[float], threshold: float = 0.8, limit: int = 5) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            try:
                # Modification de la requête pour utiliser la fonction cosine_similarity
                query = text("""
                    WITH vector_matches AS (
                        SELECT 
                            ch.id,
                            ch.query,
                            ch.response,
                            ch.confidence_score,
                            ch.created_at,
                            u.username,
                            1 - (ch.query_vector <=> :vector::vector) as similarity
                        FROM chat_history ch
                        JOIN users u ON ch.user_id = u.id
                        WHERE ch.created_at > NOW() - INTERVAL '30 days'
                    )
                    SELECT *
                    FROM vector_matches
                    WHERE similarity > :threshold
                    ORDER BY similarity DESC, created_at DESC
                    LIMIT :limit
                """)
    
                result = await session.execute(
                    query,
                    {
                        "vector": f"[{','.join(map(str, vector))}]",
                        "threshold": threshold,
                        "limit": limit
                    }
                )

                return [
                    {
                        "query": row.query,
                        "response": row.response,
                        "similarity": float(row.similarity),
                        "confidence": float(row.confidence_score),
                        "created_at": row.created_at.isoformat(),
                        "username": row.username
                    }
                    for row in result
                ]

            except Exception as e:
                logger.error(f"Erreur recherche questions similaires: {e}")
                return []

    async def log_usage(
        self,
        user_id: str,
        session_id: str,
        endpoint: str,
        response_time: float,
        status_code: int,
        error_message: Optional[str] = None
    ) -> None:
        async with self.session_factory() as session:
            try:
                metric = UsageMetric(
                    user_id=user_id,
                    session_id=session_id,
                    endpoint_path=endpoint,
                    response_time=response_time,
                    status_code=status_code,
                    error_message=error_message
                )
                session.add(metric)
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Erreur log usage: {e}")
