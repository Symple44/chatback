# api/routes/chat_routes.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Optional
from datetime import datetime
import uuid

from ..models.requests import ChatRequest
from ..models.responses import ChatResponse
from ..dependencies import get_components
from core.utils.logger import get_logger
from core.database.base import get_session_manager

logger = get_logger("chat_routes")
router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def process_chat_message(request: ChatRequest, components=Depends(get_components)):
    start_time = datetime.utcnow()
    try:
        # Recherche de documents similaires
        query_vector = components.model.create_embedding(request.query)
        relevant_docs = await components.es_client.search_documents(
            query=request.query,
            vector=query_vector,
            metadata={"application": request.application} if request.application else None
        )

        # Génération de la réponse
        response_text = await components.model.generate_response(
            query=request.query,
            context_docs=relevant_docs,
            conversation_history=request.context.get("history", []),
            language=request.language
        )

        # Création du vecteur de la réponse
        response_vector = components.model.create_embedding(response_text)

        # Sauvegarde dans la base de données
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        async with get_session_manager().get_session() as session:
            chat_id = await components.db.save_chat_interaction(
                session_id=request.session_id or str(uuid.uuid4()),
                user_id=request.user_id,
                query=request.query,
                response=response_text,
                query_vector=query_vector,
                response_vector=response_vector,
                confidence_score=max((doc.get('score', 0) for doc in relevant_docs), default=0),
                tokens_used=len(request.query.split()) + len(response_text.split()),
                processing_time=processing_time,
                referenced_docs=[{
                    "name": doc.get('title', 'Unknown'),
                    "page": doc.get('metadata', {}).get('page', 1),
                    "score": doc.get('score', 0),
                    "snippet": doc.get('content', '')[:500]
                } for doc in relevant_docs[:3]]
            )

        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            conversation_id=chat_id,
            documents_used=relevant_docs[:3],
            confidence_score=max((doc.get('score', 0) for doc in relevant_docs), default=0),
            processing_time=processing_time,
            application=request.application
        )

    except Exception as e:
        logger.error(f"Erreur traitement message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/similar")
async def get_similar_questions(
    query: str,
    threshold: float = 0.8,
    limit: int = 5,
    components=Depends(get_components)
):
    """Trouve des questions similaires basées sur la similarité vectorielle."""
    try:
        query_vector = components.model.create_embedding(query)
        similar = await components.db.find_similar_questions(
            vector=query_vector,
            threshold=threshold,
            limit=limit
        )
        return similar
    except Exception as e:
        logger.error(f"Erreur recherche questions similaires: {e}")
        raise HTTPException(status_code=500, detail=str(e))
