# core/llm/prompt_system.py
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("prompt_system")

class Message(BaseModel):
    """Modèle pour les messages de conversation."""
    role: str = Field(..., description="Role du message (system, user, assistant)")
    content: str = Field(..., description="Contenu du message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)

    @validator('role')
    def validate_role(cls, v):
        valid_roles = {"system", "user", "assistant", "context"}
        if v not in valid_roles:
            raise ValueError(f"Role invalide. Valeurs acceptées: {valid_roles}")
        return v

class PromptSystem:
    """Gestionnaire de prompts pour le modèle."""
    
    def __init__(self):
        """Initialise le système de prompts."""
        self.system_roles = settings.SYSTEM_ROLES
        self.response_types = settings.RESPONSE_TYPES

    async def build_chat_prompt(
        self,
        query: str,
        context_docs: List[Dict],
        context_summary: Optional[Union[str, Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
        context_info: Optional[Dict] = None,
        language: str = "fr",
        response_type: str = "comprehensive"
    ) -> str:
        """Construit un prompt complet avec la nouvelle structure."""
        try:
            # Construction du prompt
            messages = []
            
            # 1. Message système
            system_content = [
                f"Je suis {settings.APP_NAME}, votre assistant IA spécialisé.",
                "Je fournis des réponses précises basées sur la documentation disponible."
            ]
            messages.append({
                "role": "system",
                "content": "\n".join(system_content)
            })

            # 2. Message de contexte (comme système également)
            if context_docs or context_summary:
                context_content = []
                
                # Ajout du résumé s'il existe
                if context_summary:
                    if isinstance(context_summary, dict):
                        context_content.append(context_summary.get("structured_summary", ""))
                    else:
                        context_content.append(str(context_summary))
                
                # Ajout des sources pertinentes
                if context_docs:
                    context_content.append("\nSources pertinentes:")
                    for doc in context_docs:
                        title = doc.get("title", "")
                        score = doc.get("score", 0)
                        context_content.append(f"• {title} (pertinence: {score:.2f})")
                
                messages.append({
                    "role": "system",
                    "content": "\n".join(context_content)
                })

            # 3. Ajout de l'historique des conversations
            if conversation_history:
                for msg in conversation_history[-5:]:  # Limité aux 5 derniers messages
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })

            # 4. Ajout des instructions spécifiques (comme système)
            instructions = self._build_instructions(response_type, context_info)
            if instructions:
                messages.append({
                    "role": "system",
                    "content": instructions
                })

            # 5. Ajout de la question de l'utilisateur
            messages.append({
                "role": "user",
                "content": query
            })

            return messages

        except Exception as e:
            logger.error(f"Erreur construction prompt: {e}")
            # Format minimal en cas d'erreur
            return [
                {"role": "system", "content": f"Je suis {settings.APP_NAME}, votre assistant."},
                {"role": "user", "content": query}
            ]

    def _prepare_context(
        self,
        docs: List[Dict],
        summary: Optional[Union[str, Dict]]
    ) -> str:
        """
        Prépare le contexte en utilisant principalement le résumé du summarizer
        plutôt que les documents complets.
        """
        context_parts = []

        # Utilisation prioritaire du résumé s'il existe
        if summary:
            if isinstance(summary, dict):
                # Utilisation du résumé structuré
                structured_summary = summary.get("structured_summary", "")
                raw_summary = summary.get("raw_summary", "")
                themes = summary.get("themes", [])
                key_points = summary.get("key_points", [])
                
                # Construction du contexte structuré
                if structured_summary:
                    context_parts.append(structured_summary)
                else:
                    if themes:
                        context_parts.append("Thèmes principaux:")
                        context_parts.extend(f"• {theme}" for theme in themes)
                        context_parts.append("")
                    
                    if raw_summary:
                        context_parts.append("Résumé des documents:")
                        context_parts.append(raw_summary)
                        context_parts.append("")
                    
                    if key_points:
                        context_parts.append("Points clés:")
                        context_parts.extend(f"• {point}" for point in key_points)
            else:
                context_parts.append(str(summary))
            
            # Ajout éventuel des sources sans leur contenu complet
            if docs:
                context_parts.append("\nSources consultées:")
                for i, doc in enumerate(docs, 1):
                    title = doc.get("title", f"Document {i}")
                    score = doc.get("score", 0)
                    context_parts.append(f"• {title} (pertinence: {score:.2f})")
        else:
            # Fallback si pas de résumé : utilisation limitée des documents
            if docs:
                context_parts.append("Extraits pertinents:")
                for i, doc in enumerate(docs[:2], 1):  # Limite à 2 documents
                    title = doc.get("title", f"Document {i}")
                    # On ne prend que les 100 premiers caractères du contenu
                    content = doc.get("content", "").strip()[:100] + "..."
                    score = doc.get("score", 0)
                    context_parts.append(f"\n[Source: {title} (pertinence: {score:.2f})]\n{content}")
            else:
                context_parts.append("\nAucun document pertinent trouvé.")

        return "\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"Erreur construction prompt: {e}")
        # Format minimal en cas d'erreur
        return [
            {"role": "system", "content": f"Je suis {settings.APP_NAME}, votre assistant."},
            {"role": "user", "content": query}
        ]
        
    def _build_instructions(
        self,
        response_type: str,
        context_info: Optional[Dict]
    ) -> str:
        """Construit les instructions pour le modèle."""
        instructions = []
        
        # Instructions basées sur le type de réponse
        response_config = self.response_types.get(response_type, self.response_types["comprehensive"])
        instructions.append(response_config["description"])
        
        # Instructions basées sur le contexte
        if context_info:
            if context_info.get("needs_clarification"):
                instructions.append("Demandez des précisions si nécessaire")
            if context_info.get("confidence", 0) < 0.5:
                instructions.append("Indiquez si des informations supplémentaires sont requises")
        
        return "\n".join(instructions)
