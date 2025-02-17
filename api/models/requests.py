# api/models/requests.py
from pydantic import BaseModel, Field, EmailStr, validator, UUID4, constr, ConfigDict
from typing import Optional, Dict, List, Union, Any
from datetime import datetime
import re
from enum import Enum
from core.search.strategies import SearchMethod

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
    
class SearchConfig(BaseModel):
    """Configuration de la recherche."""
    method: SearchMethod = Field(
        default=SearchMethod.RAG,
        description="Méthode de recherche à utiliser"
    )
    params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_docs": 5,
            "min_score": 0.3
        },
        description="Paramètres spécifiques à la méthode"
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filtres de métadonnées pour la recherche"
    )

    @validator("params")
    def validate_params(cls, v, values):
        """Valide les paramètres selon la méthode."""
        method = values.get("method", SearchMethod.RAG)
        
        # Paramètres requis par méthode
        required_params = {
            SearchMethod.RAG: {"max_docs", "min_score"},
            SearchMethod.HYBRID: {"max_docs", "min_score", "rerank_top_k"},
            SearchMethod.SEMANTIC: {"max_docs", "min_score", "use_concepts"}
        }
        
        if method != SearchMethod.DISABLED:
            missing = required_params.get(method, set()) - set(v.keys())
            if missing:
                raise ValueError(f"Paramètres manquants pour {method}: {missing}")
                
        return v

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
    """Modèle de requête de chat."""
    user_id: str = Field(..., description="Identifiant de l'utilisateur")
    query: str = Field(..., min_length=1, max_length=4096)
    session_id: Optional[str] = Field(default=None)
    search_config: Optional[SearchConfig] = Field(
        default=None,
        description="Configuration de la recherche"
    )
    language: str = Field(default="fr")
    application: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('search_config')
    def setup_search_config(cls, v, values):
        """Configure la recherche avec des valeurs par défaut."""
        if v is None:
            return SearchConfig()
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
