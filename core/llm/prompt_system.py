# core/llm/prompt_system.py
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from core.config.config import settings
from core.utils.logger import get_logger
import re

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
                "content": query
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
                for i, msg in enumerate(recent_history):
                    content = msg['content']
                    role = msg['role']

                    # Nettoyer les balises existantes
                    content = re.sub(r'\[/?INST\]', '', content)

                    # Ajouter le préfixe selon le rôle
                    prefix = "Utilisateur :" if role == 'user' else "Assistant :"
                    content = f"{prefix} {content}"

                    if role == 'user':
                        content = f"[INST]{content}[/INST]"
                    elif role == 'assistant':
                        # S'assurer qu'il n'y a pas de balises INST dans la réponse
                        content = content.replace('[INST]', '').replace('[/INST]', '')

                    # Ajouter seulement si le rôle est valide
                    if role in ['user', 'assistant']:
                        formatted_history.append({
                            'role': role,
                            'content': content
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

    def _validate_mistral_format(self, messages: List[Dict]) -> List[Dict]:
        """
        Valide et formate correctement les messages pour Mistral.
        """
        try:
            # Extraction des différentes parties
            system_messages = [msg for msg in messages if msg["role"] == "system"]
            context_messages = [msg for msg in messages if msg.get("content", "").startswith("[CONTEXT]")]
            response_type_messages = [msg for msg in messages if msg.get("content", "").startswith("[RESPONSE_TYPE]")]
            chat_messages = [msg for msg in messages if msg["role"] not in ["system"] and 
                            not msg.get("content", "").startswith("[CONTEXT]") and
                            not msg.get("content", "").startswith("[RESPONSE_TYPE]")]

            # Construction du message système unifié
            system_content = []
            
            # Combiner tous les messages système
            for msg in system_messages:
                content = msg["content"]
                if "[SYSTEM_PROMPT]" not in content:
                    content = f"[SYSTEM_PROMPT]{content}[/SYSTEM_PROMPT]"
                system_content.append(content)

            # Ajout du contexte
            context_content = [msg["content"] for msg in context_messages]
            system_content.extend(context_content)

            # Ajout du type de réponse
            response_type_content = [msg["content"] for msg in response_type_messages]
            system_content.extend(response_type_content)

            # Formatage final des messages
            formatted_messages = []

            # Premier message utilisateur avec le contexte système
            if system_content and chat_messages:
                complete_system = "\n\n".join(system_content)
                
                formatted_messages.append({
                    "role": "system",
                    "content": complete_system
                })
                
                # Ajouter les messages utilisateur avec balises [INST]
                for msg in chat_messages:
                    if msg["role"] == "user":
                        formatted_messages.append({
                            "role": "user",
                            "content": f"[INST]{msg['content']}[/INST]"
                        })
                    else:
                        formatted_messages.append(msg)
            
            # Fallback si aucun message n'a été formaté
            if not formatted_messages:
                formatted_messages = [{
                    "role": "user",
                    "content": "[INST]Comment puis-je vous aider?[/INST]"
                }]

            return formatted_messages

        except Exception as e:
            logger.error(f"Erreur formatage Mistral: {e}")
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