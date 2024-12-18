from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ImageInfo(BaseModel):
    data: str  # Base64 encoded image
    mime_type: str
    width: Optional[int]
    height: Optional[int]
    position: Dict[str, float]
    
class DocumentFragment(BaseModel):
    text: str
    page_num: int
    images: List[Dict[str, Any]]
    confidence: float
    source: str
    context_before: str
    context_after: str

class AlternativeSuggestion(BaseModel):
    topic: str
    document_count: int
    examples: List[str]
    confidence: float

class ChatResponse(BaseModel):
    response: str
    session_id: str
    conversation_id: str
    documents_used: List[Dict[str, Any]]
    confidence_score: float
    fragments: Optional[List[DocumentFragment]] = None
    suggestions: Optional[List[AlternativeSuggestion]] = None
    processing_time: float
    application: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None