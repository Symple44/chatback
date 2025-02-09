# api/models/requests.py
from pydantic import BaseModel, Field, EmailStr, validator, UUID4, constr, ConfigDict
from typing import Optional, Dict, List, Union, Any
from datetime import datetime
import re
from enum import Enum

class MessageData(BaseModel):
    """Modèle pour les données de message."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Bonjour!",
                "timestamp": "2024-12-21T10:00:00Z",
                "metadata": {"source": "chat"}
            }
        }
    )
    
    content: str = Field(..., min_length=1, max_length=4096)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('content')
    def validate_content(cls, v):
        """Valide le contenu du message."""
        if not v.strip():
            raise ValueError("Le contenu ne peut pas être vide ou contenir uniquement des espaces")
        return v.strip()

class ChatContext(BaseModel):
    """Modèle pour le contexte de chat."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "history": [
                    {"role": "user", "content": "Bonjour"},
                    {"role": "assistant", "content": "Bonjour! Comment puis-je vous aider?"}
                ],
                "metadata": {"source": "web"}
            }
        }
    )
    
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        max_length=50,
        description="Historique limité aux 50 derniers messages"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)
    last_interaction: Optional[datetime] = None
    source_documents: List[str] = Field(default_factory=list)

class BusinessType(str, Enum):
    STEEL = "steel"
    WOOD = "wood"
    ALUMINUM = "aluminum"
    GENERIC = "generic"
    
class ChatRequest(BaseModel):
    """Modèle pour les requêtes de chat."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "session_id": "987fcdeb-51d3-a456-426614174000",
                "query": "Comment puis-je vous aider?",
                "business": "steel",
                "language": "fr",
                "vector_search": True  # Ajout dans l'exemple
            }
        }
    )
    
    user_id: UUID4 = Field(..., description="Identifiant unique de l'utilisateur")
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question ou requête de l'utilisateur"
    )
    session_id: Optional[UUID4] = Field(
        None,
        description="Identifiant de session au format UUID"
    )
    context: Optional[ChatContext] = Field(
        default=None,
        description="Contexte de la conversation"
    )
    business: BusinessType = Field(
        default=None,
        description="Type de métier (steel, generic)"
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
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées au format JSONB"
    )
    vector_search: bool = Field(
        default=True,
        description="Active ou désactive la recherche vectorielle"
    )

    @validator('query')
    def validate_query(cls, v):
        """Valide et nettoie la requête."""
        v = v.strip()
        if not v:
            raise ValueError("La requête ne peut pas être vide")
        if len(v.split()) < 2:
            raise ValueError("La requête doit contenir au moins deux mots")
        return v

    @validator('language')
    def validate_language(cls, v):
        """Valide le code de langue."""
        allowed_languages = {'fr', 'en', 'es', 'de', 'it'}
        if v.lower() not in allowed_languages:
            raise ValueError(f"Langue non supportée. Langues autorisées: {allowed_languages}")
        return v.lower()

    @validator('metadata')
    def validate_metadata(cls, v):
        """Valide les métadonnées."""
        max_depth = 3
        def check_depth(obj, current_depth=0):
            if current_depth > max_depth:
                raise ValueError(f"Profondeur maximale des métadonnées dépassée ({max_depth})")
            if isinstance(obj, dict):
                return all(check_depth(value, current_depth + 1) for value in obj.values())
            return True
        
        if not check_depth(v):
            raise ValueError("Structure des métadonnées trop profonde")
        return v

class SessionCreate(BaseModel):
    """Modèle pour la création de session."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "metadata": {"source": "web"}
            }
        }
    )
    
    user_id: UUID4 = Field(..., description="Identifiant de l'utilisateur")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées de la session"
    )
    initial_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Contexte initial de la session"
    )
    settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Paramètres de la session"
    )

    @validator('metadata', 'settings')
    def validate_json_fields(cls, v):
        """Valide les champs JSON."""
        if len(str(v)) > 10000:  # Limite la taille des champs JSON
            raise ValueError("Taille des données JSON trop importante")
        return v

class UserCreate(BaseModel):
    """Modèle pour la création d'utilisateur."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "user@example.com",
                "username": "john_doe",
                "full_name": "John Doe",
                "metadata": {
                    "preferences": {"theme": "dark"}
                }
            }
        }
    )
    
    email: EmailStr = Field(..., description="Email de l'utilisateur")
    username: constr(min_length=3, max_length=50, strip_whitespace=True) = Field(
        ...,
        description="Nom d'utilisateur"
    )
    full_name: constr(min_length=3, max_length=100, strip_whitespace=True) = Field(
        ...,
        description="Nom complet"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('username')
    def validate_username(cls, v):
        """Valide le format du nom d'utilisateur."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Le nom d'utilisateur ne peut contenir que des lettres, chiffres, tirets et underscores")
        return v

    @validator('full_name')
    def validate_full_name(cls, v):
        """Valide le format du nom complet."""
        if not re.match(r'^[a-zA-ZÀ-ÿ\s-]+$', v):
            raise ValueError("Le nom complet ne peut contenir que des lettres et tirets")
        return v
