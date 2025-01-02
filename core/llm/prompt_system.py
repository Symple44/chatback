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
        self.system_template = settings.SYSTEM_PROMPT

        # Configuration unifiée des rôles
        self.roles = {
            "system": {
                "prefix": "Tu es un assistant IA spécialisé dans l'aide technique et la documentation.",
                "format": "[System] {message}",
                "description": "Rôle système définissant le comportement global"
            },
            "user": {
                "prefix": "L'utilisateur demande :",
                "format": "[User] {message}",
                "description": "Questions et requêtes de l'utilisateur"
            },
            "assistant": {
                "prefix": "En tant qu'assistant technique, je vais t'aider.",
                "format": "[Assistant] {message}",
                "description": "Réponses de l'assistant"
            },
            "context": {
                "prefix": "Voici le contexte pertinent :",
                "format": "[Context] {message}",
                "description": "Informations contextuelles"
            }
        }

        # Types de réponses disponibles
        self.response_instructions = {
            "comprehensive": "Fournissez une réponse détaillée et complète.",
            "concise": "Fournissez une réponse courte et directe.",
            "technical": "Fournissez une réponse technique avec des détails d'implémentation."
        }

        # Template enrichi avec les rôles
        self.prompt_template = """
{system_role}
{system_prompt}

{context_role}
{context}

{history_role}
{history}

Instructions spécifiques:
{instructions}

{user_role}
{query}

{assistant_role}"""

    async def build_chat_prompt(
        self,
        query: str,
        context_docs: List[Dict],
        context_summary: Optional[Union[str, Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
        context_info: Optional[Dict] = None,
        language: str = "fr",
        response_type: str = "comprehensive",
        additional_instructions: Optional[str] = None,
        role: str = "assistant"
    ) -> str:
        """
        Construit un prompt complet avec gestion des rôles.
        
        Args:
            query: Question ou requête de l'utilisateur
            context_docs: Documents pertinents pour le contexte
            context_summary: Résumé optionnel du contexte
            conversation_history: Historique de la conversation
            context_info: Informations additionnelles sur le contexte
            language: Code de langue
            response_type: Type de réponse souhaité
            additional_instructions: Instructions supplémentaires
            role: Rôle pour la réponse
        """
        try:
            # Préparation du contexte avec rôle
            context_text = self._prepare_context(context_docs, context_summary, context_info)
            
            # Construction de l'historique avec rôles
            history_text = self._format_conversation_history(conversation_history)
            
            # Instructions avec contexte spécifique
            instructions = self._build_instructions(response_type, additional_instructions, context_info)
            
            # Application du template avec tous les rôles
            return self.prompt_template.format(
                system_role=self.roles["system"]["prefix"],
                system_prompt=self.system_template.format(app_name=settings.APP_NAME),
                context_role=self.roles["context"]["prefix"],
                context=context_text,
                history_role="Historique des échanges :",
                history=history_text,
                user_role=self.roles["user"]["prefix"],
                query=query,
                assistant_role=self._get_role_prefix(role),
                instructions=instructions,
                lang=language
            )

        except Exception as e:
            logger.error(f"Erreur construction prompt: {e}")
            return self._build_fallback_prompt(query, role)

    def create_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> Message:
        """Crée un message formaté pour la conversation."""
        return Message(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {"formatted": True}
        )

    def format_message(self, message: Union[Message, Dict]) -> str:
        """Formate un message selon son rôle."""
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content", "")
        else:
            role = message.role
            content = message.content

        role_info = self.roles.get(role, self.roles["user"])
        return role_info["format"].format(message=content)

    def _get_role_prefix(self, role: str) -> str:
        """Récupère le préfixe approprié pour un rôle."""
        role_info = self.roles.get(role, self.roles["assistant"])
        return role_info["prefix"]

    def _prepare_context(
        self,
        context_docs: List[Dict],
        context_summary: Optional[Union[str, Dict]],
        context_info: Optional[Dict] = None
    ) -> str:
        """Prépare le contexte pour le prompt."""
        context_parts = []
        
        if context_info:
            confidence = context_info.get("context_confidence", 0.0)
            themes = context_info.get("themes", [])
            if confidence > 0 or themes:
                context_parts.append("Analyse du contexte :")
                context_parts.append(f"- Niveau de confiance : {confidence:.2%}")
                if themes:
                    context_parts.append(f"- Thèmes identifiés : {', '.join(themes)}")
                context_parts.append("")

        if not context_docs:
            context_parts.append("Pas de contexte spécifique disponible.")
        else:
            if isinstance(context_summary, dict):
                context_text = context_summary.get("structured_summary", "")
                context_parts.append(context_text)
            elif isinstance(context_summary, str):
                context_parts.append(context_summary)
            else:
                context_parts.extend(self._format_context_docs(context_docs))

        return "\n".join(context_parts)

    def _format_context_docs(self, docs: List[Dict]) -> List[str]:
        """Formate une liste de documents en texte de contexte."""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.get("title", f"Document {i}")
            content = doc.get("content", "").strip()
            if content:
                score = doc.get("score", 0)
                context_parts.append(
                    f"[Source: {title} (pertinence: {score:.2f})]\n{content}"
                )
        return context_parts

    def _format_conversation_history(self, history: Optional[List[Dict]]) -> str:
        """Formate l'historique de conversation."""
        if not history:
            return "Pas d'historique disponible."

        return "\n".join(
            self.format_message(msg) for msg in history[-5:]
        )

    def _build_instructions(
        self,
        response_type: str,
        additional_instructions: Optional[str],
        context_info: Optional[Dict] = None
    ) -> str:
        """Construit les instructions pour le modèle."""
        if response_type not in self.response_instructions:
            logger.warning(f"Type de réponse inconnu: {response_type}")
            response_type = "comprehensive"
            
        instructions = [self.response_instructions[response_type]]
        
        if additional_instructions:
            instructions.append(additional_instructions)

        if context_info:
            if context_info.get("needs_clarification"):
                instructions.append("Demandez des clarifications si nécessaire.")
            if context_info.get("ambiguous_themes", False):
                instructions.append("Identifiez clairement le thème principal de la réponse.")
            if context_info.get("context_confidence", 0) < 0.5:
                instructions.append("Précisez si des informations supplémentaires sont nécessaires.")
            
        return "\n".join(instructions)

    def _build_fallback_prompt(self, query: str, role: str = "assistant") -> str:
        """Construit un prompt minimal en cas d'erreur."""
        role_info = self.roles.get(role, self.roles["assistant"])
        message = f"Question: {query}" + "\n" + "Réponse:"
        return role_info["format"].format(message=message)