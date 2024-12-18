# api/models/requests.py
from pydantic import BaseModel, Field, EmailStr, validator, UUID4
from typing import Optional, Dict, List, Union, Any
from datetime import datetime

class MessageData(BaseModel):
    content: str = Field(..., min_length=1, max_length=4096)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict] = Field(default={})

class ChatContext(BaseModel):
    history: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Union[str, int, float, bool]] = Field(default={})
    preferences: Dict[str, Union[str, int, float, bool]] = Field(default={})

class ChatRequest(BaseModel):
    user_id: UUID4 = Field(..., description="Identifiant unique de l'utilisateur")
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question ou requête de l'utilisateur"
    )
    session_id: Optional[str] = Field(
        None,
        min_length=32,
        max_length=64,
        description="Identifiant de session optionnel"
    )
    context: Optional[ChatContext] = Field(
        default=ChatContext(),
        description="Contexte de la conversation"
    )
    language: str = Field(
        default="fr",
        min_length=2,
        max_length=5,
        description="Code de langue (fr, en, etc.)"
    )
    application: Optional[str] = Field(
        None,
        max_length=50,
        description="Nom de l'application source"
    )
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("La requête ne peut pas être vide")
        return v.strip()
    
    @validator('language')
    def validate_language(cls, v):
        allowed_languages = {'fr', 'en', 'es', 'de', 'it'}
        if v not in allowed_languages:
            raise ValueError(f"Langue non supportée. Langues autorisées: {allowed_languages}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "query": "Comment créer une commande ?",
                "session_id": "session-123456789",
                "context": {
                    "history": [
                        {"role": "user", "content": "Question précédente"},
                        {"role": "assistant", "content": "Réponse précédente"}
                    ],
                    "metadata": {"source": "web_interface"},
                    "preferences": {"response_length": "detailed"}
                },
                "language": "fr",
                "application": "cmmanager"
            }
        }
        from_attributes = True

class SessionCreate(BaseModel):
    user_id: UUID4 = Field(..., description="Identifiant de l'utilisateur")
    metadata: Optional[Dict[str, Any]] = Field(default={})

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "metadata": {
                    "source": "web_app",
                    "user_agent": "Mozilla/5.0..."
                }
            }
        }
        from_attributes = True
