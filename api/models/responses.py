# api/models/responses.py
from pydantic import BaseModel, Field, UUID4, EmailStr, validator, constr
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import numpy as np

class DocumentReference(BaseModel):
    """Référence à un document source."""
    title: str = Field(..., min_length=1, max_length=255)
    page: Optional[int] = Field(None, ge=1)
    score: float = Field(..., ge=0.0, le=1.0)
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    vector_id: Optional[str] = None
    last_updated: Optional[datetime] = None

    @validator('score')
    def round_score(cls, v: float) -> float:
        return round(v, 4)

class ImageInfo(BaseModel):
    """Information sur une image."""
    data: str  # Base64 encoded image
    mime_type: str
    width: Optional[int] = Field(None, gt=0)
    height: Optional[int] = Field(None, gt=0)
    position: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentFragment(BaseModel):
    """Fragment de document avec contexte."""
    text: str = Field(..., min_length=1)
    page_num: int = Field(..., gt=0)
    images: List[ImageInfo] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: str
    context_before: str = ""
    context_after: str = ""
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SimilarQuestion(BaseModel):
    """Question similaire trouvée."""
    query: str
    response: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime
    user_id: Optional[UUID4] = None
    session_id: Optional[UUID4] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('similarity')
    def round_similarity(cls, v: float) -> float:
        return round(v, 4)

class SessionResponse(BaseModel):
    """Réponse de session."""
    session_id: UUID4 = Field(..., description="Identifiant unique de la session")
    user_id: UUID4
    created_at: datetime
    updated_at: datetime
    context: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    stats: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "987fcdeb-51d3-12d3-a456-426614174000",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "context": {
                    "history": [],
                    "preferences": {}
                },
                "is_active": True,
                "metadata": {
                    "source": "web_interface",
                    "user_agent": "Mozilla/5.0..."
                },
                "stats": {
                    "total_messages": 0,
                    "avg_response_time": 0
                }
            }
        }
        from_attributes = True

class ChatResponse(BaseModel):
    """Réponse complète du chat."""
    response: str = Field(..., min_length=1)
    session_id: UUID4
    conversation_id: UUID4
    documents: List[DocumentReference] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time: float = Field(..., ge=0.0)
    tokens_used: int = Field(..., ge=0)
    cost: float = Field(default=0.0, ge=0.0)
    application: Optional[str] = None
    fragments: Optional[List[DocumentFragment]] = None
    similar_questions: Optional[List[SimilarQuestion]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    query_vector: Optional[List[float]] = None
    response_vector: Optional[List[float]] = None
    context_used: Optional[Dict[str, Any]] = None
    suggested_actions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @validator('confidence_score', 'processing_time')
    def round_floats(cls, v: float) -> float:
        return round(v, 4)

    @validator('query_vector', 'response_vector')
    def validate_vector(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            if len(v) != 384:  # Dimension du vecteur
                raise ValueError("Dimension du vecteur incorrecte")
            return [round(x, 6) for x in v]
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Pour créer une commande, suivez ces étapes...",
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "conversation_id": "987fcdeb-51d3-12d3-a456-426614174000",
                "documents": [{
                    "title": "Guide des commandes",
                    "page": 1,
                    "score": 0.95,
                    "content": "Extrait pertinent..."
                }],
                "confidence_score": 0.95,
                "processing_time": 1.23,
                "tokens_used": 150,
                "application": "cmmanager"
            }
        }
        from_attributes = True

class UserResponse(BaseModel):
    """Réponse utilisateur."""
    id: UUID4
    email: EmailStr
    username: constr(min_length=3, max_length=50)
    full_name: constr(min_length=3, max_length=100)
    created_at: datetime
    last_login: Optional[datetime]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = Field(default=True)
    stats: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True

class ErrorResponse(BaseModel):
    """Réponse d'erreur."""
    detail: str
    error_code: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    path: Optional[str] = None
    request_id: Optional[UUID4] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Erreur de traitement de la requête",
                "error_code": "PROCESSING_ERROR",
                "timestamp": "2024-01-15T10:30:00Z",
                "path": "/api/chat",
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "metadata": {
                    "component": "model_inference",
                    "error_type": "timeout"
                }
            }
        }

class VectorStats(BaseModel):
    """Statistiques des vecteurs."""
    total_vectors: int = Field(..., ge=0)
    dimension: int = Field(..., eq=384)
    avg_processing_time: float
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HealthCheckResponse(BaseModel):
    """Réponse du health check."""
    status: str = Field(..., regex="^(healthy|unhealthy|degraded)$")
    components: Dict[str, bool]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics: Dict[str, Any] = Field(default_factory=dict)
