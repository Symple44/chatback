# core/llm/prompt_system.py
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from core.config.config import settings
from core.utils.logger import get_logger
import re
import json

logger = get_logger("prompt_system")

class MessageRole:
    """Enumération des rôles de message."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    CONTEXT = "context"
    TOOL = "tool"

class MessageTag:
    """Gestion centralisée des balises de message."""
    SYSTEM_PROMPT = ("[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]")
    CONTEXT = ("[CONTEXT]", "[/CONTEXT]")
    RESPONSE_TYPE = ("[RESPONSE_TYPE]", "[/RESPONSE_TYPE]")
    CONVERSATION_HISTORY = ("[CONVERSATION_HISTORY]", "[/CONVERSATION_HISTORY]")
    INSTRUCTION = ("[INST]", "[/INST]")

class Message(BaseModel):
    """Modèle de message avec validation enrichie."""
    role: str = Field(..., description="Rôle du message")
    content: str = Field(..., description="Contenu du message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    language: Optional[str] = Field(default=None)

    @validator('role')
    def validate_role(cls, v):
        valid_roles = {
            MessageRole.SYSTEM, 
            MessageRole.USER, 
            MessageRole.ASSISTANT, 
            MessageRole.CONTEXT, 
            MessageRole.TOOL
        }
        if v not in valid_roles:
            raise ValueError(f"Rôle invalide. Valeurs acceptées: {valid_roles}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convertit le message en dictionnaire."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "language": self.language
        }

class PromptSystem:
    """Gestionnaire avancé de prompts pour modèles LLM."""
    
    def __init__(self, config=None):
        """
        Initialise le système de prompts avec configuration personnalisable.
        
        :param config: Configuration optionnelle pour personnaliser le comportement
        """
        self.config = config or {}
        self.system_roles = settings.SYSTEM_ROLES
        self.response_types = settings.RESPONSE_TYPES or {}
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
        context_docs: Optional[List[Dict]] = None,
        context_summary: Optional[Union[str, Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
        context_info: Optional[Dict] = None,
        prompt_prefix: Optional[str] = None,
        language: str = "fr",
        response_type: str = "comprehensive"
    ) -> List[Dict]:
        """
        Construit un prompt complet avec une structure modulaire et flexible.
        
        :param query: La requête utilisateur
        :param context_docs: Documents de contexte
        :param context_summary: Résumé du contexte
        :param conversation_history: Historique de conversation
        :param context_info: Informations supplémentaires sur le contexte
        :param prompt_prefix: Préfixe personnalisé pour le prompt
        :param language: Langue de réponse
        :param response_type: Type de réponse souhaité
        :return: Liste de messages formatés
        """
        try:
            messages = []
            is_mistral = "mistral" in str(settings.MODEL_NAME).lower()

            # 1. Message système principal
            system_content = self._generate_system_message(
                response_type, 
                language, 
                prompt_prefix
            )
            messages.append({
                "role": MessageRole.SYSTEM,
                "content": system_content
            })

            # 2. Gestion du contexte documentaire
            if context_docs:
                context_message = await self._build_context_section(
                    context_docs, 
                    context_summary, 
                    context_info
                )
                if context_message:
                    messages.append({
                        "role": MessageRole.CONTEXT,
                        "content": context_message
                    })

            # 3. Historique de conversation
            if conversation_history:
                history_messages = await self._build_conversation_history(
                    conversation_history, 
                    is_mistral
                )
                messages.extend(history_messages)

            # 4. Message utilisateur
            messages.append({
                "role": MessageRole.USER,
                "content": query
            })

            # 5. Validation et formatage final
            return (
                self._validate_mistral_format(messages) 
                if is_mistral 
                else self._validate_llama_format(messages)
            )

        except Exception as e:
            logger.error(f"Erreur lors de la construction du prompt: {e}", exc_info=True)
            return self._fallback_prompt(query)

    def _generate_system_message(
        self, 
        response_type: str, 
        language: str, 
        prompt_prefix: Optional[str] = None
    ) -> str:
        """
        Génère un message système personnalisé.
        
        :param response_type: Type de réponse
        :param language: Langue de réponse
        :param prompt_prefix: Préfixe personnalisé
        :return: Message système formaté
        """
        # Construction du message système
        system_parts = [
            f"{MessageTag.SYSTEM_PROMPT[0]}",
            f"Je suis {settings.APP_NAME}, un assistant spécialisé.",
            "\nPrincipes fondamentaux :",
            "- Je fournis des informations précises",
            "- Je demande des clarifications si nécessaire",
            "- Je reconnais mes limites",
            "- Je me concentre sur des solutions pratiques",
            f"\nDate actuelle : {datetime.utcnow().strftime('%d/%m/%Y')}",
            f"\nLangue : {self.language_indicators.get(language, 'Français')}",
        ]

        # Ajout des instructions de type de réponse
        if response_type in self.response_types:
            system_parts.extend([
                f"{MessageTag.RESPONSE_TYPE[0]}",
                self.response_types[response_type].get("description", ""),
                self.response_types[response_type].get("style", ""),
                f"{MessageTag.RESPONSE_TYPE[1]}"
            ])

        # Ajout d'un préfixe personnalisé si fourni
        if prompt_prefix:
            system_parts.append(f"\nInstructions spécifiques : {prompt_prefix}")

        system_parts.append(f"{MessageTag.SYSTEM_PROMPT[1]}")
        return "\n".join(system_parts)

    async def _build_context_section(
        self, 
        context_docs: List[Dict], 
        context_summary: Optional[Union[str, Dict]] = None,
        context_info: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Construit la section de contexte de manière robuste.
        
        :param context_docs: Documents de contexte
        :param context_summary: Résumé du contexte
        :param context_info: Informations supplémentaires
        :return: Section de contexte formatée
        """
        if not context_docs:
            return None

        context_parts = [f"{MessageTag.CONTEXT[0]}"]
        
        # Ajout du résumé de contexte
        if context_summary:
            context_parts.append("Résumé du contexte :")
            context_parts.append(
                json.dumps(context_summary) 
                if isinstance(context_summary, dict) 
                else str(context_summary)
            )

        # Ajout des documents de contexte
        context_parts.append("Documents de contexte :")
        for doc in context_docs[:self.max_context_length]:
            context_parts.append(json.dumps(doc))

        # Ajout d'informations supplémentaires
        if context_info:
            context_parts.append("Informations supplémentaires :")
            context_parts.append(json.dumps(context_info))

        context_parts.append(f"{MessageTag.CONTEXT[1]}")
        return "\n".join(context_parts)

    async def _build_conversation_history(
        self,
        history: List[Dict],
        is_mistral: bool
    ) -> List[Dict]:
        """
        Construit l'historique de conversation.
        Pour Mistral : assure l'alternance user/assistant et le bon formatage des balises.
        """
        try:
            formatted_history = []
            max_history = settings.MAX_HISTORY_MESSAGES  # Utiliser la configuration globale
            
            # Valider et trier l'historique par timestamp si présent
            valid_history = []
            for msg in history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    # S'assurer que le contenu est bien une chaîne
                    content = str(msg['content']).strip()
                    if content:  # Ignorer les messages vides
                        timestamp = msg.get('timestamp')
                        if timestamp:
                            if isinstance(timestamp, str):
                                try:
                                    timestamp = datetime.fromisoformat(timestamp)
                                except ValueError:
                                    timestamp = datetime.utcnow()
                        else:
                            timestamp = datetime.utcnow()
                            
                        valid_history.append({
                            'role': msg['role'],
                            'content': content,
                            'timestamp': timestamp
                        })

            # Trier par timestamp et prendre les messages les plus récents
            valid_history.sort(key=lambda x: x['timestamp'])
            recent_history = valid_history[-(max_history*2):]  # *2 car on a user + assistant

            # Ajouter l'en-tête de l'historique
            formatted_history.append({
                'role': 'system',
                'content': '[CONVERSATION_HISTORY]\nVoici les échanges précédents :'
            })

            if is_mistral:
                # Format spécifique pour Mistral
                formatted_history.append({
                    'role': 'system',
                    'content': f"{MessageTag.CONVERSATION_HISTORY[0]}Voici les échanges précédents :{MessageTag.CONVERSATION_HISTORY[1]}"
                })
                
                for msg in recent_history:
                    if msg['role'] == 'user':
                        formatted_history.append({
                            'role': 'user',
                            'content': f"[INST]{msg['content']}[/INST]"
                        })
                    elif msg['role'] == 'assistant':
                        formatted_history.append({
                            'role': 'assistant',
                            'content': msg['content']
                        })
            else:
                # Format standard pour les autres modèles
                for msg in recent_history:
                    if msg['role'] in ['user', 'assistant']:
                        prefix = "Utilisateur :" if msg['role'] == 'user' else "Assistant :"
                        formatted_history.append({
                            'role': msg['role'],
                            'content': f"{prefix} {msg['content']}"
                        })

            # Ajouter la balise de fin de l'historique
            formatted_history.append({
                'role': 'system',
                'content': '[/CONVERSATION_HISTORY]\n'
            })

            return formatted_history

        except Exception as e:
            logger.error(f"Erreur formatage historique: {e}", exc_info=True)
            return []

    def _fallback_prompt(self, query: str) -> List[Dict]:
        """
        Génère un prompt de secours en cas d'erreur.
        
        :param query: Requête utilisateur
        :return: Liste de messages de secours
        """
        return [{
            "role": MessageRole.SYSTEM,
            "content": f"{MessageTag.SYSTEM_PROMPT[0]}Assistant générique{MessageTag.SYSTEM_PROMPT[1]}"
        }, {
            "role": MessageRole.USER,
            "content": query
        }]

    def _validate_mistral_format(self, messages: List[Dict]) -> List[Dict]:
        """
        Validation et formatage spécifique pour Mistral.
        
        :param messages: Liste des messages à formater
        :return: Messages formatés pour Mistral
        """
        formatted_messages = []
        system_content = []
        user_messages = []
        conversation_history = []

        # Regroupement et nettoyage des messages
        for msg in messages:
            if msg['role'] in [MessageRole.SYSTEM, MessageRole.CONTEXT, "context"]:
                system_content.append(msg['content'])
            elif msg['role'] == MessageRole.USER:
                user_messages.append(msg['content'])
            elif 'CONVERSATION_HISTORY' in msg.get('content', ''):
                conversation_history.append(msg['content'])

        # Création du message système unifié
        if system_content:
            formatted_messages.append({
                "role": MessageRole.SYSTEM,
                "content": "\n\n".join(system_content)
            })

        # Ajout de l'historique de conversation
        if conversation_history:
            formatted_messages.append({
                "role": "system",
                "content": conversation_history[0]
            })

        # Formatage des messages utilisateur
        for user_msg in user_messages:
            formatted_messages.append({
                "role": MessageRole.USER,
                "content": f"[INST]{user_msg}[/INST]"
            })

        return formatted_messages or self._fallback_prompt("Comment puis-je vous aider ?")

    def _validate_llama_format(self, messages: List[Dict]) -> List[Dict]:
        """
        Validation et formatage pour les modèles de type Llama.
        
        :param messages: Liste des messages à formater
        :return: Messages formatés pour Llama
        """
        return [
            msg for msg in messages
            if msg.get("role") in [
                MessageRole.SYSTEM, 
                MessageRole.USER, 
                MessageRole.ASSISTANT
            ]
            and msg.get("content")
        ] or self._fallback_prompt("Comment puis-je vous aider ?")