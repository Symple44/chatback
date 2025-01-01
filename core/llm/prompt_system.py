# core/llm/prompt_system.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import re
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("prompt_system")

class Message(BaseModel):
    role: str = Field(..., description="Role du message (system, user, assistant)")
    content: str = Field(..., description="Contenu du message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)

class PromptSystem:
   def __init__(self):
       self.chat_template = settings.CHAT_TEMPLATE
       self.system_template = settings.SYSTEM_PROMPT

   def build_chat_prompt(
       self,
       messages: List[Dict],
       context: str,
       query: str,
       lang: str = "fr"
   ) -> str:
       # Préparation du contexte système
       system = self.system_template.format(app_name=settings.APP_NAME)

       # Construction de l'historique 
       history = []
       if messages:
           for msg in messages[-5:]:  
               role = msg.get("role", "user")
               content = msg.get("content", "")
               history.append(f"{role}: {content}")

       # Construction du prompt
       prompt = self.chat_template.format(
           system=system,
           query=query or "",
           context=context or "",
           context_summary=self._extract_key_points(context) if context else "",
           history="\n\n".join(history),
           timestamp=datetime.utcnow().isoformat(),
           language=lang,
           response=""
       )

       return prompt

   def _extract_key_points(self, context: str) -> str:
       # Extraction des points clés        
       patterns = [
           (r"Pour (créer|modifier|supprimer).*?[.]\n", 0.9), 
           (r"^\d+\.\s+.*?[.]\n", 0.8),
           (r"(?:Attention|Important)\s*:.*?[.]\n", 0.7),
           (r"^[-•]\s+.*?[.]\n", 0.6)
       ]

       points = []
       for pattern, score in patterns:
           matches = re.finditer(pattern, context, re.MULTILINE)
           for match in matches:
               text = match.group(0).strip()
               if text and len(text) > 20:
                   points.append((text, score))

       points.sort(key=lambda x: x[1], reverse=True)
       return "\n• " + "\n• ".join(p[0] for p in points[:5])