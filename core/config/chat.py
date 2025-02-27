# core/config/chat.py
import os
from enum import Enum
from pydantic import BaseModel

class BusinessType(str, Enum):
    """Types de processeurs métier."""
    STEEL = "steel"
    WOOD = "wood"
    ALUMINUM = "aluminum"
    GENERIC = "generic"

class ChatConfig(BaseModel):
    """Configuration pour le chat."""
    
    # Paramètres du processeur de chat
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "5"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    MIN_RELEVANT_DOCS: int = int(os.getenv("MIN_RELEVANT_DOCS", "1"))
    MAX_THEMES: int = int(os.getenv("MAX_THEMES", "3"))
    MAX_CLARIFICATION_ATTEMPTS: int = int(os.getenv("MAX_CLARIFICATION_ATTEMPTS", "3"))
    MIN_QUERY_LENGTH: int = int(os.getenv("MIN_QUERY_LENGTH", "2"))
    MAX_CONTEXT_DOCS: int = int(os.getenv("MAX_CONTEXT_DOCS", "6"))