# api/models/requests.py
from pydantic import BaseModel, Field, EmailStr, validator, UUID4, constr, ConfigDict
from typing import Optional, Dict, List, Union, Any
from datetime import datetime
import re
from enum import Enum
from core.search.strategies import SearchMethod
from core.config.config import settings
from core.config.chat_config import BusinessType

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

class SearchParamsBase(BaseModel):
    """Modèle de base pour les paramètres de recherche."""
    max_docs: int = Field(
        default=settings.MAX_RELEVANT_DOCS,
        ge=1,
        le=20,
        description="Nombre maximum de documents à retourner"
    )
    min_score: float = Field(
        default=settings.CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Score minimum de confiance"
    )

class RAGParams(SearchParamsBase):
    """Paramètres spécifiques pour la recherche RAG."""
    vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Poids pour le score vectoriel"
    )
    semantic_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Poids pour le score sémantique"
    )

class HybridParams(SearchParamsBase):
    """Paramètres spécifiques pour la recherche hybride."""
    rerank_top_k: int = Field(
        default=10,
        ge=1,
        description="Nombre de documents à reranker"
    )
    combine_method: str = Field(
        default="weighted_sum",
        pattern="^(weighted_sum|max|average)$",
        description="Méthode de combinaison des scores"
    )

class SemanticParams(SearchParamsBase):
    """Paramètres spécifiques pour la recherche sémantique."""
    use_concepts: bool = Field(
        default=True,
        description="Utiliser l'extraction de concepts"
    )
    boost_exact_matches: bool = Field(
        default=True,
        description="Booster les correspondances exactes"
    )

class SearchConfig(BaseModel):
    """Configuration complète de la recherche."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "method": "rag",
                "params": {
                    "max_docs": 5,
                    "min_score": 0.3,
                    "vector_weight": 0.7,
                    "semantic_weight": 0.3
                }
            }
        }
    )

    method: SearchMethod = Field(
        default=SearchMethod.RAG,
        description="Méthode de recherche à utiliser"
    )
    params: Union[RAGParams, HybridParams, SemanticParams, Dict[str, Any]] = Field(
        default_factory=RAGParams,
        description="Paramètres de la recherche"
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filtres de métadonnées"
    )

    @validator('params')
    def validate_params(cls, v, values):
        """Valide les paramètres selon la méthode."""
        method = values.get('method', SearchMethod.RAG)
        
        # Si params est déjà un dictionnaire, le convertir en classe appropriée
        if isinstance(v, dict):
            if method == SearchMethod.RAG:
                v = RAGParams(**v)
            elif method == SearchMethod.HYBRID:
                v = HybridParams(**v)
            elif method == SearchMethod.SEMANTIC:
                v = SemanticParams(**v)
            elif method == SearchMethod.DISABLED:
                v = SearchParamsBase(**v)
                
        # Vérifier que le type de paramètres correspond à la méthode
        param_class_mapping = {
            SearchMethod.RAG: RAGParams,
            SearchMethod.HYBRID: HybridParams,
            SearchMethod.SEMANTIC: SemanticParams,
            SearchMethod.DISABLED: SearchParamsBase
        }
        
        expected_class = param_class_mapping.get(method)
        if not isinstance(v, expected_class):
            raise ValueError(f"Type de paramètres incorrect pour {method}. Attendu: {expected_class.__name__}")
            
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
    
class ChatRequest(BaseModel):
    """Modèle pour la requête de chat."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Comment créer une commande d'achat ?",
                "user_id": "123",
                "business": "generic",
                "search_config": {
                    "method": "rag",
                    "params": {
                        "max_docs": 5,
                        "min_score": 0.3,
                        "vector_weight": 0.7,
                        "semantic_weight": 0.3
                    }
                }
            }
        }
    )

    query: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Question de l'utilisateur"
    )
    user_id: str = Field(..., description="Identifiant de l'utilisateur")
    business: BusinessType = Field(
        default=BusinessType.GENERIC,
        description="Type de processeur métier à utiliser"
    )
    session_id: Optional[str] = Field(default=None)
    search_config: Optional[SearchConfig] = Field(
        default=None,
        description="Configuration de la recherche"
    )
    language: str = Field(default="fr")
    application: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('search_config')
    def setup_search_config(cls, v):
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
