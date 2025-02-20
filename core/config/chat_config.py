# core/chat/chat_config.py
from enum import Enum

class BusinessType(str, Enum):
    STEEL = "steel"
    WOOD = "wood"
    ALUMINUM = "aluminum"
    GENERIC = "generic"