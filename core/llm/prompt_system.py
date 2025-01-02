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
        self.chat_template = settings.CHAT_TEMPLATE
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
            # System message
            system = self.system_roles["system"].format(app_name=settings.APP_NAME)

            # Contexte
            context_content = self._prepare_context(context_docs, context_summary)

            # Analyse du contexte
            themes = self._format_themes(context_info.get("themes", []))
            clarifications = self._format_clarifications(context_info.get("clarifications", []))
            questions = self._format_questions(context_info.get("suggested_questions", []))

            # Historique
            history = self._format_conversation_history(conversation_history)

            # Instructions
            instructions = self._build_instructions(response_type, context_info)

            # Construction du prompt final avec les balises de rôle
            return self.chat_template.format(
                system=system,
                context=context_content,
                themes=themes,
                clarifications=clarifications,
                questions=questions,
                history=history,
                instructions=instructions,
                query=query,
                response=""  # La réponse sera générée par le modèle
            )

        except Exception as e:
            logger.error(f"Erreur construction prompt: {e}")
            return self._build_fallback_prompt(query)

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

    def _format_conversation_history(
        self,
        history: Optional[List[Dict]]
    ) -> str:
        """Formate l'historique avec les rôles."""
        if not history:
            return "Pas d'historique disponible."

        formatted_messages = []
        for msg in history[-settings.MAX_HISTORY_MESSAGES:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            role_prefix = self.system_roles.get(role, "")
            formatted_messages.append(f"<|{role}|>\n{content}\n</|{role}|>")

        return "\n".join(formatted_messages)

    def _build_instructions(
        self,
        response_type: str,
        context_info: Optional[Dict]
    ) -> str:
        """Construit les instructions avec le nouveau format."""
        instructions = []

        # Instructions de base selon le type de réponse
        response_config = self.response_types.get(response_type, self.response_types["comprehensive"])
        instructions.append(response_config["description"])

        # Instructions basées sur le contexte
        if context_info:
            if context_info.get("needs_clarification"):
                instructions.append("• Demandez des précisions si nécessaire")
            if context_info.get("context_confidence", 0) < 0.5:
                instructions.append("• Indiquez si des informations supplémentaires sont requises")
            if context_info.get("ambiguous_themes", False):
                instructions.append("• Précisez le thème principal de la réponse")

        return "\n".join(instructions)

    def _format_themes(self, themes: List[str]) -> str:
        """Formate la liste des thèmes."""
        if not themes:
            return "• Aucun thème spécifique identifié"
        return "\n".join(f"• {theme}" for theme in themes)

    def _format_clarifications(self, clarifications: List[str]) -> str:
        """Formate les points nécessitant clarification."""
        if not clarifications:
            return "• Aucune clarification nécessaire"
        return "\n".join(f"• {clarif}" for clarif in clarifications)

    def _format_questions(self, questions: List[str]) -> str:
        """Formate les questions suggérées."""
        if not questions:
            return "• Aucune question supplémentaire"
        return "\n".join(f"• {q}" for q in questions)

    def _build_fallback_prompt(self, query: str) -> str:
        """Construit un prompt minimal en cas d'erreur."""
        return f"""<|system|>
{self.system_roles['system'].format(app_name=settings.APP_NAME)}
</|system|>

<|user|>
{query}
</|user|>

<|assistant|>
Je vais essayer de vous aider avec cette demande.
</|assistant|>"""