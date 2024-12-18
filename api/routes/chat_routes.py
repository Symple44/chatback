# api/routes/chat_routes.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
from datetime import datetime
import uuid
import os
from ..models.requests import ChatRequest
from ..models.responses import ChatResponse
from ..dependencies import get_components
from core.utils.logger import get_logger

logger = get_logger("chat_routes")
router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def process_chat_message(request: ChatRequest, components=Depends(get_components)):
    try:
        start_time = datetime.utcnow()
        logger.info(f"Nouvelle requête reçue: {request.dict()}")

        # Recherche de documents
        relevant_docs = await components.es_client.search_documents(
            query=request.query,
            metadata={"application": request.application} if request.application else None
        )
        logger.info(f"Documents trouvés: {len(relevant_docs)}")

        fragments = []
        if relevant_docs:
            first_doc = relevant_docs[0]
            doc_path = first_doc.get('metadata', {}).get('pdf_path')
            
            if doc_path and os.path.exists(doc_path):
                fragments = await components.doc_extractor.extract_relevant_fragments(
                    doc_path=doc_path,
                    keywords=request.query.split()
                )
                # S'assurer que tous les champs requis sont présents
                formatted_fragments = [{
                    "text": f.text,
                    "page_num": f.page_num if hasattr(f, 'page_num') else 1,  # Valeur par défaut si manquant
                    "images": f.images if hasattr(f, 'images') else [],
                    "confidence": f.confidence if hasattr(f, 'confidence') else 0.0,
                    "source": f.source if hasattr(f, 'source') else doc_path,
                    "context_before": f.context_before if hasattr(f, 'context_before') else "",
                    "context_after": f.context_after if hasattr(f, 'context_after') else ""
                } for f in fragments]

        response_text = await components.model.generate_response(
            query=request.query,
            context_docs=relevant_docs,
            conversation_history=request.context.get("history", []),
            language=request.language
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return ChatResponse(
            response=response_text,
            session_id=request.session_id or str(uuid.uuid4()),
            conversation_id=str(uuid.uuid4()),
            documents_used=relevant_docs[:3],
            confidence_score=max((doc.get("score", 0) for doc in relevant_docs), default=0),
            fragments=formatted_fragments if fragments else None,  # Utiliser les fragments formatés
            processing_time=processing_time,
            application=request.application
        )

    except Exception as e:
        logger.error(f"Erreur traitement requête: {e}")
        raise HTTPException(status_code=500, detail=str(e))
