# core/llm/prompt_system.py
from typing import List, Dict, Optional, Union, ClassVar
from pydantic import BaseModel, Field
from datetime import datetime
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("prompt_system")

class Message(BaseModel):
    role: str = Field(..., description="Role du message (system, user, assistant, function)")
    content: str = Field(..., description="Contenu du message")
    name: Optional[str] = Field(None, description="Nom optionnel pour les fonctions")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)

class ConversationContext(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    system_context: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)
    max_messages: int = Field(default=10)

class PromptSystem:
    SYSTEM_MESSAGES: ClassVar[Dict[str, str]] = {
        "welcome": "Bienvenue ! Comment puis-je vous aider ?",
        "error": "Désolé, une erreur est survenue.",
        "rate_limit": "Vous avez atteint la limite de requêtes.", 
        "maintenance": "Le système est en maintenance.",
        "context_missing": "Aucun contexte documentaire disponible.",
        "max_tokens": "La limite de tokens a été atteinte.",
        "invalid_request": "Requête invalide. Veuillez réessayer.",
        "processing": "Traitement en cours...",
        "completed": "Traitement terminé."
    }

    def __init__(self):
        """Initialise le système de prompts."""
        self.chat_template = settings.CHAT_TEMPLATE
        self.system_template = settings.SYSTEM_PROMPT
        self.max_context_length = settings.MAX_CONTEXT_LENGTH
        
        self.role_formats = {
            "system": "Système: {message}",
            "user": "Utilisateur: {message}",
            "assistant": "Assistant: {message}",
            "function": "Fonction ({name}): {message}"
        }
        
        self.role_instructions = {
            "system": "Instructions système :",
            "user": "Requête utilisateur :",
            "assistant": "Réponse assistant :",
            "function": "Résultat fonction {name} :"
        }

    def build_chat_prompt(
        self,
        messages: List[Message],
        context_docs: Optional[List[Dict]] = None,
        lang: str = "fr",
        instruction_format: bool = True
    ) -> str:
        """
        Construit le prompt complet pour le chat.
        
        Args:
            messages: Liste des messages
            context_docs: Documents de contexte
            lang: Langue de la réponse
            instruction_format: Ajouter les instructions formatées
            
        Returns:
            str: Le prompt formaté
        """
        system = self.system_template.format(app_name=settings.APP_NAME)
        context = self._format_context(context_docs) if context_docs else self.get_system_message("context_missing")
        query = next((m.content for m in reversed(messages) if m.role == "user"), "")
        
        return self.chat_template.format(
            system=system,
            query=query,
            context=context,
            context_summary=self._generate_context_summary(context_docs) if context_docs else "",
            response=""
        )

    def _format_context(self, context_docs: List[Dict]) -> str:
        """Formate le contexte documentaire."""
        doc_parts = []
        for doc in context_docs:
            title = doc.get('title', 'Document')
            content = doc.get('content', '').strip()
            page = doc.get('metadata', {}).get('page', 'N/A')
            confidence = doc.get('score', 0.0)
            
            if content:
                header = f"[Source: {title} (Page {page}, Pertinence: {confidence:.2f})]"
                doc_parts.append(f"{header}\n{content}")
                
        return "\n\n".join(doc_parts)

    def _generate_context_summary(self, context_docs: List[Dict]) -> str:
        """Génère un résumé du contexte."""
        if not context_docs:
            return ""
            
        n_docs = len(context_docs)
        avg_confidence = sum(doc.get('score', 0.0) for doc in context_docs) / n_docs if n_docs > 0 else 0
        
        return f"Basé sur {n_docs} documents pertinents (confiance moyenne: {avg_confidence:.2f})"

    def create_message(
        self,
        role: str,
        content: str,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Message:
        """Crée un nouveau message avec métadonnées."""
        return Message(
            role=role,
            content=content,
            name=name,
            metadata=metadata or {}
        )

    def format_system_prompt(self, lang: str = "fr") -> Message:
        """Crée le message système initial."""
        content = self.system_template.format(app_name=settings.APP_NAME)
        metadata = {"lang": lang, "type": "system_prompt"}
        return self.create_message("system", content, metadata=metadata)

    def get_system_message(self, key: str, **kwargs) -> str:
        """Récupère un message système avec formatage optionnel."""
        message = self.SYSTEM_MESSAGES.get(key, self.SYSTEM_MESSAGES["error"])
        return message.format(**kwargs) if kwargs else message

    def truncate_messages(
        self,
        messages: List[Message],
        max_messages: Optional[int] = None
    ) -> List[Message]:
        """Tronque la liste des messages en gardant les plus récents."""
        if not max_messages:
            max_messages = settings.MAX_HISTORY_MESSAGES
            
        # Garde toujours le message système
        system_msg = next((m for m in messages if m.role == "system"), None)
        history = [m for m in messages if m.role != "system"]
        
        truncated = history[-max_messages:]
        if system_msg:
            truncated.insert(0, system_msg)
            
        return truncated

    def format_message(self, message: Message) -> str:
        """Formate un message selon son rôle."""
        template = self.role_formats.get(message.role, "{message}")
        return template.format(
            message=message.content,
            name=message.name or message.role
        )

    def get_conversation_stats(self, messages: List[Message]) -> Dict:
        """Calcule des statistiques sur la conversation."""
        return {
            "total_messages": len(messages),
            "messages_per_role": {
                role: len([m for m in messages if m.role == role])
                for role in set(m.role for m in messages)
            },
            "avg_message_length": sum(len(m.content) for m in messages) / len(messages) if messages else 0,
            "last_message_time": max(m.timestamp for m in messages) if messages else None
        }
