# api/models/requests.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime

class MessageData(BaseModel):
    content: str
    timestamp: datetime

class AssistantMessageData(MessageData):
    documents_used: Optional[List[Dict]] = None
    confidence_score: Optional[float] = None

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Identifiant de l'utilisateur")
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="Identifiant de session optionnel")
    context: Optional[Dict] = Field(default={}, description="Contexte additionnel")
    language: Optional[str] = Field("fr", description="Langue de la réponse")
    application: Optional[str] = Field(None, description="Nom de l'application")

class ChatHistoryCreate(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    user_message: MessageData
    assistant_message: AssistantMessageData