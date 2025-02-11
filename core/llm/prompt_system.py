# core/llm/prompt_system.py
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from core.config.config import settings
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
        Construit un prompt complet avec structure améliorée.
        Compatible avec Mistral et Llama.
        """
        try:
            messages = []
            is_mistral = "mistral" in settings.MODEL_NAME.lower()

            # 1. Message système avec format amélioré
            system_content = [
                "[SYSTEM_PROMPT]",
                f"Je suis {settings.APP_NAME}, un assistant spécialisé dans le logiciel 2CM Manager, dédié aux métiers de la charpente métallique, serrurerie et métallerie.",
                "\nPrincipes fondamentaux :",
                "- Je fournis des informations précises basées sur la documentation 2CM Manager",
                "- Je demande des clarifications si une question n'est pas claire",
                "- Je reconnais quand une information est manquante",
                "- Je me concentre sur des solutions pratiques et réalisables",
                "- J'utilise un langage professionnel mais accessible",
                "- Je m'adapte aux besoins spécifiques de chaque utilisateur",
                f"\nDate actuelle : {datetime.utcnow().strftime('%d/%m/%Y')}",
                "[/SYSTEM_PROMPT]"
            ]

            # 2. Ajout du contexte documentaire si disponible
            if context_docs:
                context_content = await self._build_context_section(
                    context_docs,
                    context_summary,
                    context_info
                )
                if context_content:
                    messages.append({
                        "role": "system",
                        "content": "\n".join([
                            "[CONTEXT]",
                            *context_content,
                            "[/CONTEXT]"
                        ])
                    })

            # 3. Instructions spécifiques au type de réponse
            if response_type in self.response_types:
                response_config = self.response_types[response_type]
                system_content.extend([
                    "\n[RESPONSE_TYPE]",
                    response_config["description"],
                    response_config.get("style", ""),
                    "[/RESPONSE_TYPE]"
                ])

            # 4. Construction du message système final
            messages.append({
                "role": "system",
                "content": "\n".join(system_content)
            })

            # 5. Ajout de l'historique de conversation
            if conversation_history:
                history_messages = await self._build_conversation_history(
                    conversation_history,
                    is_mistral
                )
                messages.extend(history_messages)

            # 6. Ajout de la question actuelle avec format clair
            messages.append({
                "role": "user",
                "content": f"[INST]{query}[/INST]"
            })

            # 7. Validation du format selon le modèle
            if is_mistral:
                messages = self._validate_mistral_format(messages)
            else:
                messages = self._validate_llama_format(messages)

            return messages

        except Exception as e:
            logger.error(f"Erreur construction prompt: {e}")
            return [
                {"role": "system", "content": f"[SYSTEM_PROMPT]You are {settings.APP_NAME}.[/SYSTEM_PROMPT]"},
                {"role": "user", "content": f"[INST]{query}[/INST]"}
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
        """
        Valide et formate correctement les messages pour Mistral.
        Maintient la structure claire tout en respectant les contraintes Mistral.
        """
        try:
            # Extraction des différentes parties
            system_messages = [msg for msg in messages if msg["role"] == "system"]
            context_messages = [msg for msg in messages if msg.get("content", "").startswith("[CONTEXT]")]
            chat_messages = [msg for msg in messages if msg["role"] not in ["system"] and not msg.get("content", "").startswith("[CONTEXT]")]

            if not chat_messages:
                chat_messages = [{
                    "role": "user",
                    "content": "[INST]Comment puis-je vous aider?[/INST]"
                }]

            # Construction du message formaté
            formatted_messages = []
            
            # Pour Mistral, on combine le contenu système dans le premier message utilisateur
            system_content = []
            if system_messages:
                # Fusion des messages système
                for msg in system_messages:
                    content = msg["content"]
                    # Éviter la duplication des balises si déjà présentes
                    if not content.startswith("[SYSTEM_PROMPT]"):
                        content = f"[SYSTEM_PROMPT]\n{content}\n[/SYSTEM_PROMPT]"
                    system_content.append(content)

            # Ajout du contexte s'il existe
            if context_messages:
                system_content.extend([msg["content"] for msg in context_messages])

            # Trouver le premier message utilisateur et y injecter le contenu système
            system_block = "\n\n".join(system_content)
            first_user_msg_found = False
            
            for msg in chat_messages:
                if msg["role"] == "user" and not first_user_msg_found:
                    # Assurons-nous que le message utilisateur a les balises [INST]
                    user_content = msg["content"]
                    if not user_content.startswith("[INST]"):
                        user_content = f"[INST]{user_content}[/INST]"
                        
                    # Combinaison du contenu système avec le message utilisateur
                    msg["content"] = f"{system_block}\n\n{user_content}"
                    first_user_msg_found = True
                formatted_messages.append(msg)

            # Vérifier que le dernier message est de l'utilisateur
            if formatted_messages[-1]["role"] != "user":
                formatted_messages = formatted_messages[:-1]

            # Validation finale des balises
            for i, msg in enumerate(formatted_messages):
                if msg["role"] == "user" and not msg["content"].strip().endswith("[/INST]"):
                    formatted_messages[i]["content"] = f"{msg['content']}[/INST]"

            return formatted_messages

        except Exception as e:
            logger.error(f"Erreur formatage Mistral: {e}")
            # Format minimal en cas d'erreur
            return [{
                "role": "user",
                "content": "[INST]Comment puis-je vous aider?[/INST]"
            }]

    def _validate_llama_format(self, messages: List[Dict]) -> List[Dict]:
        """Valide et corrige le format pour Llama."""
        # Llama est plus flexible, on vérifie juste la présence des rôles valides
        return [
            msg for msg in messages
            if msg.get("role") in ["system", "user", "assistant"]
            and msg.get("content")
        ]