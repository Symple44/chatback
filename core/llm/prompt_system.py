# core/llm/prompt_system.py
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("prompt_system")

class Message(BaseModel):
    """Modèle pour les messages de conversation."""
    role: str = Field(..., description="Role du message (system, user, assistant, tool)")
    content: str = Field(..., description="Contenu du message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)

    @validator('role')
    def validate_role(cls, v):
        valid_roles = {"system", "user", "assistant", "context", "tool"}
        if v not in valid_roles:
            raise ValueError(f"Role invalide. Valeurs acceptées: {valid_roles}")
        return v

class PromptSystem:
    """Gestionnaire de prompts pour le modèle."""
    
    def __init__(self):
        """Initialise le système de prompts."""
        self.system_roles = settings.SYSTEM_ROLES
        self.response_types = settings.RESPONSE_TYPES
        self.max_context_length = settings.MAX_CONTEXT_LENGTH
        self.language_indicators = {
            "fr": "Répondez en français",
            "en": "Respond in English",
            "es": "Responda en español",
            "de": "Antworten Sie auf Deutsch"
        }

    async def build_chat_prompt(
        self,
        query: str,
        context_docs: List[Dict],
        context_summary: Optional[Union[str, Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
        context_info: Optional[Dict] = None,
        prompt_prefix: Optional[str] = None,
        language: str = "fr",
        response_type: str = "comprehensive"
    ) -> List[Dict]:
        """
        Construit un prompt complet avec la nouvelle structure.
        Compatible avec Mistral et Llama.
        """
        try:
            messages = []
            is_mistral = "mistral" in settings.MODEL_NAME.lower()
            
            # 1. Message système avec informations de base et prompt_prefix
            system_content = [
                f"Je suis {settings.APP_NAME}, votre assistant IA spécialisé dans le logiciel 2CM Manager pour le métier de la charpente métallique, serrurerie et métallerie.",
                "Je fournis des réponses précises basées sur la documentation disponible."
            ]

            # Ajout du prefix personnalisé
            if prompt_prefix:
                system_content.append(prompt_prefix)

            # 2. Ajout du contexte
            context_content = await self._build_context_section(
                context_docs,
                context_summary,
                context_info
            )
            if context_content:
                system_content.extend(context_content)

            # 3. Instructions spécifiques au type de réponse
            instructions = self._build_instructions(response_type, context_info)
            if instructions:
                system_content.append(instructions)

            # 4. Construction du message système final
            messages.append({
                "role": "system",
                "content": "\n".join(system_content)
            })

            # 5. Ajout de l'historique avec gestion spécifique selon le modèle
            if conversation_history:
                history_messages = await self._build_conversation_history(
                    conversation_history,
                    is_mistral
                )
                messages.extend(history_messages)

            # 6. Ajout de la question actuelle
            messages.append({
                "role": "user",
                "content": query
            })

            # 7. Validation finale selon le modèle
            if is_mistral:
                messages = self._validate_mistral_format(messages)
            else:
                messages = self._validate_llama_format(messages)

            return messages

        except Exception as e:
            logger.error(f"Erreur construction prompt: {e}")
            # Format minimal en cas d'erreur
            return [
                {"role": "system", "content": f"Je suis {settings.APP_NAME}, votre assistant."},
                {"role": "user", "content": query}
            ]

    async def _build_context_section(
        self,
        context_docs: List[Dict],
        context_summary: Optional[Union[str, Dict]],
        context_info: Optional[Dict]
    ) -> List[str]:
        """Construit la section de contexte du prompt."""
        context_content = []

        # Ajout du résumé du contexte
        if context_summary:
            if isinstance(context_summary, dict):
                summary_text = context_summary.get("structured_summary", "")
                if summary_text:
                    context_content.append(f"\nRésumé du contexte:\n{summary_text}")
                
                # Ajout des thèmes identifiés
                if themes := context_summary.get("themes", []):
                    context_content.append(f"\nThèmes principaux: {', '.join(themes)}")
            else:
                context_content.append(f"\nContexte:\n{str(context_summary)}")

        # Ajout des documents pertinents
        if context_docs:
            context_content.append("\nSources pertinentes:")
            for doc in context_docs[:3]:  # Limite aux 3 plus pertinents
                title = doc.get("title", "Document")
                content = doc.get("content", "")[:200]  # Limite de longueur
                score = doc.get("score", 0)
                context_content.append(f"• {title} (pertinence: {score:.2f}):\n{content}...")

        # Ajout des informations de contexte additionnelles
        if context_info and context_info.get("needs_clarification"):
            context_content.append("\nPoints nécessitant clarification:")
            if points := context_info.get("clarification_points", []):
                context_content.extend([f"• {point}" for point in points])

        return context_content

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
        instructions.append(response_config.get("style", ""))
        
        # Instructions basées sur le contexte
        if context_info:
            if context_info.get("needs_clarification"):
                instructions.append("Si nécessaire, demandez des clarifications.")
            if context_info.get("confidence", 0) < 0.5:
                instructions.append("En cas d'incertitude, précisez les informations manquantes.")
        
        return "\n".join(instructions)

    async def _build_conversation_history(
        self,
        history: List[Dict],
        is_mistral: bool
    ) -> List[Dict]:
        """Construit l'historique de conversation au format approprié."""
        history_messages = []
        max_messages = 5  # Limite aux 5 derniers messages
        
        filtered_history = [
            msg for msg in history[-max_messages:]
            if msg.get("role") in ["user", "assistant"] and msg.get("content")
        ]

        if is_mistral:
            # Pour Mistral : assurer l'alternance user/assistant
            current_role = "user"
            for msg in filtered_history:
                if msg["role"] == current_role:
                    history_messages.append(msg)
                    current_role = "assistant" if current_role == "user" else "user"
        else:
            # Pour Llama : pas besoin d'alternance stricte
            history_messages.extend(filtered_history)

        return history_messages

    def _validate_mistral_format(self, messages: List[Dict]) -> List[Dict]:
        """Valide et corrige le format pour Mistral."""
        validated = []
        has_system = False

        for msg in messages:
            if msg["role"] == "system":
                if not has_system:
                    validated.append(msg)
                    has_system = True
            elif msg["role"] in ["user", "assistant"]:
                validated.append(msg)

        # S'assurer que le dernier message est de l'utilisateur
        if validated and validated[-1]["role"] != "user":
            validated.pop()

        return validated

    def _validate_llama_format(self, messages: List[Dict]) -> List[Dict]:
        """Valide et corrige le format pour Llama."""
        # Llama est plus flexible, on vérifie juste la présence des rôles valides
        return [
            msg for msg in messages
            if msg.get("role") in ["system", "user", "assistant"]
            and msg.get("content")
        ]