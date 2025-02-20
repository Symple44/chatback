# api/models/responses.py
from pydantic import BaseModel, Field, UUID4, EmailStr, validator, constr, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from core.search.strategies import SearchMethod

class UserResponse(BaseModel):
    """Réponse utilisateur."""
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "email": "user@example.com",
                "username": "john_doe"
            }
        }
    )
    
    id: UUID4
    email: EmailStr
    username: constr(min_length=3, max_length=50)
    full_name: constr(min_length=3, max_length=100)
    created_at: datetime
    last_login: Optional[datetime]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = Field(default=True)
    stats: Dict[str, Any] = Field(default_factory=dict)

class DocumentReference(BaseModel):
    """Référence à un document source."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Guide utilisateur",
                "page": 1,
                "score": 0.95,
                "content": "Extrait du document..."
            }
        }
    )
    
    title: str = Field(..., min_length=1, max_length=255)
    page: Optional[int] = Field(default=0, ge=1)  
    score: float = Field(..., ge=0.0)
    content: str = Field(default="Pas de contenu disponible")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    vector_id: str = Field(default_factory=lambda: str(uuid.uuid4()))  
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    @validator('title')
    def ensure_title(cls, v: str) -> str:
        """S'assure qu'un titre est toujours présent."""
        if not v.strip():
            return "Document sans titre"
        return v.strip()
    
    @validator('score')
    def normalize_score(cls, v: float) -> float:
        """Normalise le score si nécessaire."""
        if v > 1.0:
            return (v - 1.0) / 1.0
        return v

class ImageInfo(BaseModel):
    """Information sur une image."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": "base64_encoded_image_data",
                "mime_type": "image/jpeg",
                "width": 800,
                "height": 600
            }
        }
    )
    
    data: str  # Base64 encoded image
    mime_type: str
    width: Optional[int] = Field(None, gt=0)
    height: Optional[int] = Field(None, gt=0)
    position: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentFragment(BaseModel):
    """Fragment de document avec contexte."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Extrait du document...",
                "page_num": 1,
                "confidence": 0.95,
                "source": "document.pdf"
            }
        }
    )
    
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
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Comment configurer...",
                "response": "Pour configurer...",
                "similarity": 0.92
            }
        }
    )
    
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
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "created_at": "2024-12-21T10:00:00Z"
            }
        }
    )
    
    session_id: UUID4 = Field(..., description="Identifiant unique de la session")
    user_id: UUID4
    created_at: datetime
    updated_at: datetime
    context: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    stats: Dict[str, Any] = Field(default_factory=dict)

class SearchResult(BaseModel):
    """Résultat de recherche enrichi."""
    content: str = Field(..., description="Contenu trouvé")
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_type: str = Field(default="document")
    relevance_details: Optional[Dict[str, Any]] = None

class SearchMetadata(BaseModel):
    """Métadonnées de recherche."""
    method: SearchMethod
    params: Dict[str, Any]
    stats: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total_found": 0,
            "processed": 0,
            "processing_time": 0.0
        }
    )
    timing: Dict[str, float] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    """Réponse complète du chat."""
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "response": "Voici la réponse...",
                "confidence_score": 0.95,
                "processing_time": 0.234
            }
        }
    )
    
    response: str = Field(..., min_length=1)
    session_id: UUID4
    conversation_id: UUID4
    # Résultats de recherche enrichis
    context_docs: List[SearchResult] = Field(
        default_factory=list,
        description="Documents de contexte trouvés"
    )
    # Métadonnées enrichies
    search_metadata: Optional[SearchMetadata] = None
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time: float = Field(..., ge=0.0)
    tokens_used: int = Field(..., ge=0)
    tokens_details: Optional[Dict[str, int]] = Field(default_factory=dict)
    application: str = Field(default="chat_api") 
    fragments: List[DocumentFragment] = Field(default_factory=list)  
    similar_questions: List[SimilarQuestion] = Field(default_factory=list)  
    metadata: Dict[str, Any] = Field(default_factory=lambda: {
        "source": "model",
        "timestamp": datetime.utcnow().isoformat(),
        "session_context": {},
        "raw_response": None,  
        "raw_prompt": None    
    })
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    query_vector: List[float] = Field(default_factory=list) 
    response_vector: List[float] = Field(default_factory=list)  
    context_used: Dict[str, Any] = Field(default_factory=dict)  
    suggested_actions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @validator('confidence_score', 'processing_time')
    def round_floats(cls, v: float) -> float:
        return round(v, 4)

class ErrorResponse(BaseModel):
    """Réponse d'erreur."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Une erreur est survenue",
                "error_code": "INTERNAL_ERROR",
                "timestamp": "2024-12-21T10:00:00Z"
            }
        }
    )
    
    detail: str
    error_code: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    path: Optional[str] = None
    request_id: Optional[UUID4] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VectorStats(BaseModel):
    """Statistiques des vecteurs."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_vectors": 1000,
                "dimension": 384,
                "avg_processing_time": 0.05
            }
        }
    )
    
    total_vectors: int = Field(..., ge=0)
    dimension: int = Field(..., eq=384)
    avg_processing_time: float
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HealthCheckResponse(BaseModel):
    """Réponse du health check."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "components": {"database": True, "cache": True}
            }
        }
    )
    
    status: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Dict[str, bool] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)

class SearchMetrics(BaseModel):
    """Métriques de recherche enrichies."""
    total_searches: int = Field(default=0, description="Nombre total de recherches")
    success_rate: float = Field(default=0.0, description="Taux de succès")
    average_time: float = Field(default=0.0, description="Temps moyen de traitement")
    cache_hit_rate: float = Field(default=0.0, description="Taux d'utilisation du cache")
    processing_time: float = Field(..., description="Temps de traitement")
    results_count: int = Field(..., description="Nombre de résultats")
    cache_hit: bool = Field(..., description="Utilisation du cache")
    memory_usage: Dict[str, float] = Field(..., description="Utilisation mémoire")
    query_vector_time: float = Field(..., description="Temps de création du vecteur")

class SearchDebugInfo(BaseModel):
    """Informations de debug pour la recherche."""
    raw_config: Dict[str, Any] = Field(..., description="Configuration brute")
    search_strategy: str = Field(..., description="Stratégie utilisée")
    cache_status: Dict[str, Any] = Field(..., description="Statut du cache")
    memory_details: Dict[str, Any] = Field(..., description="Détails mémoire")
    request_headers: Dict[str, str] = Field(..., description="En-têtes de la requête")
    timing_breakdown: Dict[str, float] = Field(..., description="Détail des temps")

class SearchTestResponse(BaseModel):
    """Réponse du test de recherche."""
    test_id: UUID4 = Field(..., description="Identifiant unique du test")
    success: bool = Field(..., description="Succès du test")
    method: SearchMethod = Field(..., description="Méthode de recherche utilisée")
    configuration_used: Dict[str, Any] = Field(..., description="Configuration appliquée")
    results: List[DocumentReference] = Field(..., description="Résultats de recherche")
    metrics: SearchMetrics = Field(..., description="Métriques de performance")
    debug_info: Optional[SearchDebugInfo] = Field(None, description="Informations de debug")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "test_id": "123e4567-e89b-12d3-a456-426614174000",
                "success": True,
                "method": "rag",
                "configuration_used": {
                    "max_docs": 5,
                    "min_score": 0.3,
                    "vector_weight": 0.7,
                    "semantic_weight": 0.3
                },
                "results": [
                    {
                        "title": "Document 1",
                        "score": 0.95,
                        "content": "Extrait du contenu..."
                    }
                ],
                "metrics": {
                    "processing_time": 0.234,
                    "results_count": 1,
                    "cache_hit": False,
                    "memory_usage": {
                        "gpu": 1024,
                        "ram": 2048
                    },
                    "query_vector_time": 0.023
                }
            }
        }

class SearchError(BaseModel):
    """Erreur détaillée de recherche."""
    error_code: str = Field(..., description="Code d'erreur")
    detail: str = Field(..., description="Description de l'erreur")
    metadata: Dict[str, Any] = Field(..., description="Métadonnées de l'erreur")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "VALIDATION_ERROR",
                "detail": "Paramètre invalide",
                "metadata": {
                    "field": "max_docs",
                    "value": 25,
                    "constraint": "max_value: 20"
                },
                "timestamp": "2024-02-18T12:00:00Z"
            }
        }

class SearchValidationError(SearchError):
    """Erreur de validation spécifique à la recherche."""
    def __init__(self, field: str, value: Any, constraint: str):
        super().__init__(
            error_code="VALIDATION_ERROR",
            detail=f"Validation error for {field}: {constraint}",
            metadata={
                "field": field,
                "value": value,
                "constraint": constraint
            }
        )

class SearchLimits(BaseModel):
    """Limites de configuration de recherche."""
    max_docs: int = Field(default=20, ge=1, le=100)
    min_score: float = Field(default=0.1, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    semantic_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    
    class Config:
        frozen = True  # Rend l'objet immuable

SEARCH_LIMITS = SearchLimits()

__all__ = [
    'ChatResponse',
    'ErrorResponse',
    'DocumentReference',
    'SimilarQuestion',
    'VectorStats',
    'SearchMetadata',
    'SearchMetrics',
    'SearchTestResponse',
    'SearchDebugInfo',
    'SearchError'
]