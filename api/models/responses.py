# api/models/responses.py
from pydantic import BaseModel, Field, UUID4, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentReference(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    page: Optional[int] = Field(None, ge=1)
    score: float = Field(..., ge=0.0, le=1.0)
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default={})

    @validator('score')
    def validate_score(cls, v):
        return round(v, 4)

class ImageInfo(BaseModel):
    data: str  # Base64 encoded image
    mime_type: str
    width: Optional[int] = Field(None, gt=0)
    height: Optional[int] = Field(None, gt=0)
    position: Dict[str, float] = Field(default_factory=dict)

class DocumentFragment(BaseModel):
    text: str
    page_num: int = Field(..., gt=0)
    images: List[ImageInfo] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: str
    context_before: str = ""
    context_after: str = ""

class SimilarQuestion(BaseModel):
    query: str
    response: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime

class ChatResponse(BaseModel):
    response: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=32, max_length=64)
    conversation_id: UUID4
    documents_used: List[DocumentReference] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time: float = Field(..., ge=0.0)
    application: Optional[str] = None
    fragments: Optional[List[DocumentFragment]] = None
    similar_questions: Optional[List[SimilarQuestion]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Ajout de champs pour le suivi et le contexte
    conversation_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Contexte de la conversation actuelle"
    )
    
    recommended_actions: Optional[List[str]] = Field(
        default=None,
        description="Actions recommandées basées sur la réponse"
    )

    @validator('confidence_score', 'processing_time')
    def round_floats(cls, v):
        return round(v, 4) if v is not None else v

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Pour créer une commande, suivez ces étapes...",
                "session_id": "session-123456789",
                "conversation_id": "987fcdeb-51d3-12d3-a456-426614174000",
                "documents_used": [
                    {
                        "title": "Guide des commandes",
                        "page": 1,
                        "score": 0.95,
                        "content": "Extrait du document...",
                        "metadata": {
                            "source": "documentation",
                            "last_updated": "2024-01-15"
                        }
                    }
                ],
                "confidence_score": 0.95,
                "processing_time": 1.23,
                "application": "cmmanager",
                "fragments": [
                    {
                        "text": "Extrait pertinent du document",
                        "page_num": 1,
                        "confidence": 0.9,
                        "source": "Guide.pdf"
                    }
                ],
                "similar_questions": [
                    {
                        "query": "Comment faire une commande ?",
                        "response": "Pour faire une commande...",
                        "similarity": 0.92,
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                ],
                "metadata": {
                    "model_version": "1.0",
                    "tokens_used": 150
                }
            }
        }
        from_attributes = True
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID4: lambda v: str(v)
        }

class ErrorResponse(BaseModel):
    detail: str
    error_code: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    path: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Une erreur est survenue lors du traitement",
                "error_code": "PROCESSING_ERROR",
                "timestamp": "2024-01-15T10:30:00Z",
                "path": "/api/chat"
            }
        }
