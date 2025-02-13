# core/llm/prompt_system.py
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from core.config.config import settings
from core.config.models import (
    AVAILABLE_MODELS,
    EMBEDDING_MODELS,
    SUMMARIZER_MODELS
)
from core.utils.logger import get_logger

logger = get_logger("prompt_system")

class ModelType:
    """Types de modèles supportés."""
    MISTRAL = "mistral"
    LLAMA = "llama"
    T5 = "t5"
    MT5 = "mt5"
    FALCON = "falcon"
    MIXTRAL = "mixtral"

class MessageRole:
    """Rôles de message."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    CONTEXT = "context"
    TOOL = "tool"

class PromptTemplates:
    """Templates de prompts par type de modèle."""
    
    @staticmethod
    def format_mistral(role: str, content: str, is_first: bool = False) -> str:
        """Format pour modèles Mistral."""
        if is_first:
            # Premier message (toujours système)
            return f"<s>{content}</s>"
        elif role == MessageRole.SYSTEM:
            return f"{content}"
        elif role == MessageRole.USER:
            return f"[INST] {content} [/INST]"
        elif role == MessageRole.ASSISTANT:
            return content
        elif role == MessageRole.CONTEXT:
            return f"[CONTEXT]{content}[/CONTEXT]"
        return content
        
    @staticmethod
    def format_mixtral(role: str, content: str) -> str:
        """Format pour modèles Mixtral."""
        if role == MessageRole.SYSTEM:
            return f"<s><|system|>{content}</s>"
        elif role == MessageRole.USER:
            return f"<s><|user|>{content}</s>"
        elif role == MessageRole.ASSISTANT:
            return f"<s><|assistant|>{content}</s>"
        elif role == MessageRole.CONTEXT:
            return f"<s><|context|>{content}</s>"
        return content

    @staticmethod
    def format_llama(role: str, content: str) -> str:
        """Format pour modèles Llama."""
        if role == MessageRole.SYSTEM:
            return f"<<SYS>>{content}<</SYS>>"
        elif role == MessageRole.USER:
            return f"[INST]{content}[/INST]"
        elif role == MessageRole.CONTEXT:
            return f"[CTX]{content}[/CTX]"
        return content
        
    @staticmethod
    def format_t5(role: str, content: str) -> str:
        """Format pour modèles T5."""
        if role == MessageRole.SYSTEM:
            return f"system: {content}"
        elif role == MessageRole.USER:
            return f"question: {content}"
        elif role == MessageRole.ASSISTANT:
            return f"answer: {content}"
        elif role == MessageRole.CONTEXT:
            return f"context: {content}"
        return content
        
    @staticmethod
    def format_mt5(role: str, content: str) -> str:
        """Format pour modèles MT5."""
        return PromptTemplates.format_t5(role, content)
        
    @staticmethod
    def format_falcon(role: str, content: str) -> str:
        """Format pour modèles Falcon."""
        if role == MessageRole.SYSTEM:
            return f"System: {content}\n"
        elif role == MessageRole.USER:
            return f"User: {content}\n"
        elif role == MessageRole.ASSISTANT:
            return f"Assistant: {content}\n"
        elif role == MessageRole.CONTEXT:
            return f"Context: {content}\n"
        return content

class PromptSystem:
    """Gestionnaire de prompts pour les modèles LLM."""
    
    def __init__(self):
        self.system_roles = settings.SYSTEM_ROLES
        self.response_types = settings.RESPONSE_TYPES or {}
        self.max_context_length = settings.MAX_CONTEXT_LENGTH
        self.language_indicators = {
            "fr": "Répondez en français",
            "en": "Respond in English",
            "es": "Responda en español",
            "de": "Antworten Sie auf Deutsch"
        }
        
        # Mapping des modèles vers leurs templates
        self.model_templates = {
            ModelType.MISTRAL: PromptTemplates.format_mistral,
            ModelType.MIXTRAL: PromptTemplates.format_mixtral,
            ModelType.LLAMA: PromptTemplates.format_llama,
            ModelType.T5: PromptTemplates.format_t5,
            ModelType.MT5: PromptTemplates.format_mt5,
            ModelType.FALCON: PromptTemplates.format_falcon
        }

    def _detect_model_type(self, model_name: str) -> str:
        """Détecte le type de modèle à partir de son nom."""
        model_name = model_name.lower()
        if "mistral" in model_name:
            return ModelType.MISTRAL
        elif "mixtral" in model_name:
            return ModelType.MIXTRAL
        elif "llama" in model_name:
            return ModelType.LLAMA
        elif "mt5" in model_name:
            return ModelType.MT5
        elif "t5" in model_name:
            return ModelType.T5
        elif "falcon" in model_name:
            return ModelType.FALCON
        logger.warning(f"Type de modèle non reconnu pour {model_name}, utilisation du format Mistral par défaut")
        return ModelType.MISTRAL

    def _get_model_config(self, model_name: str) -> Optional[Dict]:
        """Récupère la configuration du modèle."""
        if model_name in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[model_name]
        elif model_name in EMBEDDING_MODELS:
            return EMBEDDING_MODELS[model_name]
        elif model_name in SUMMARIZER_MODELS:
            return SUMMARIZER_MODELS[model_name]
        return None

    def _get_template(self, model_name: str) -> callable:
        """Retourne le template approprié pour le modèle."""
        model_type = self._detect_model_type(model_name)
        return self.model_templates.get(model_type, PromptTemplates.format_mistral)

    def _build_system_message(
        self, 
        response_type: str,
        language: str,
        model_name: str,
        prompt_prefix: Optional[str] = None
    ) -> str:
        """Construit le message système avec formatage spécifique au modèle."""
        model_config = self._get_model_config(model_name)
        model_type = self._detect_model_type(model_name)
        
        parts = [
            f"Je suis {settings.APP_NAME}, un assistant spécialisé.",
            "\nPrincipes fondamentaux :",
            "- Je fournis des informations précises",
            "- Je demande des clarifications si nécessaire",
            "- Je reconnais mes limites",
            "- Je me concentre sur des solutions pratiques",
            f"\nDate actuelle : {datetime.utcnow().strftime('%d/%m/%Y')}",
            f"\nLangue : {self.language_indicators.get(language, 'Français')}"
        ]

        # Ajout des instructions spécifiques au modèle
        if model_config and "instructions" in model_config:
            parts.append(f"\nInstructions modèle :")
            parts.extend([f"- {instr}" for instr in model_config["instructions"]])

        # Ajout du type de réponse si spécifié
        if response_type in self.response_types:
            if model_type in [ModelType.T5, ModelType.MT5]:
                # Format spécifique pour T5/MT5
                parts.extend([
                    f"\nstyle: {self.response_types[response_type].get('description', '')}",
                    f"instructions: {self.response_types[response_type].get('style', '')}"
                ])
            else:
                parts.extend([
                    "\n[RESPONSE_TYPE]",
                    self.response_types[response_type].get("description", ""),
                    self.response_types[response_type].get("style", ""),
                    "[/RESPONSE_TYPE]"
                ])

        # Ajout du préfixe personnalisé
        if prompt_prefix:
            parts.append(f"\nInstructions spécifiques : {prompt_prefix}")

        return "\n".join(parts)

    async def _build_context_section(
        self,
        context_docs: List[Dict],
        model_name: str,
        include_metadata: bool = True
    ) -> Optional[str]:
        """Construit la section de contexte selon le modèle."""
        if not context_docs:
            return None

        model_type = self._detect_model_type(model_name)
        sorted_docs = sorted(
            context_docs,
            key=lambda x: float(x.get("score", 0)),
            reverse=True
        )

        context_parts = []

        # Format spécifique selon le type de modèle
        for doc in sorted_docs:
            if not doc.get("content"):
                continue

            doc_parts = []
            
            if model_type in [ModelType.T5, ModelType.MT5]:
                # Format plus concis pour T5/MT5
                doc_parts.append(f"text: {doc['content']}")
                if include_metadata and doc.get("metadata"):
                    doc_parts.append(f"source: {doc['metadata'].get('source', 'unknown')}")
            else:
                # Format standard pour autres modèles
                if include_metadata:
                    if doc.get("title"):
                        doc_parts.append(f"Document: {doc['title']}")
                    metadata = doc.get("metadata", {})
                    if metadata.get("page"):
                        doc_parts.append(f"Page: {metadata['page']}")
                    if metadata.get("section"):
                        doc_parts.append(f"Section: {metadata['section']}")
                    doc_parts.append(f"Pertinence: {doc.get('score', 0):.2f}")
                
                doc_parts.append(f"Contenu: {doc['content']}")
            
            context_parts.append("\n".join(doc_parts))

        return "\n\n".join(context_parts)

    async def _build_conversation_history(
        self,
        history: List[Dict],
        model_name: str,
        max_messages: int = 5
    ) -> str:
        """Construit l'historique de conversation selon le modèle."""
        if not history:
            return ""

        model_type = self._detect_model_type(model_name)
        recent_history = history[-max_messages*2:]
        history_parts = []

        for msg in recent_history:
            if not (msg.get("role") and msg.get("content")):
                continue

            if model_type in [ModelType.T5, ModelType.MT5]:
                # Format T5/MT5
                role = "question" if msg["role"] == "user" else "response"
                history_parts.append(f"{role}: {msg['content']}")
            elif model_type == ModelType.FALCON:
                # Format Falcon
                role = "User" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {msg['content']}")
            else:
                # Format standard
                role = "Utilisateur" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {msg['content']}")

        return "\n".join(history_parts)

    async def build_chat_prompt(
        self,
        query: str,
        context_docs: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
        language: str = "fr",
        response_type: str = "comprehensive",
        model_name: Optional[str] = None,
        prompt_prefix: Optional[str] = None,
        include_metadata: bool = True
    ) -> Union[List[Dict], str]:
        """
        Construit le prompt complet pour le chat.
        """
        try:
            model_name = model_name or settings.MODEL_NAME
            model_type = self._detect_model_type(model_name)
            template = self._get_template(model_name)
            
            if model_type == ModelType.MISTRAL:
                # Traitement spécial pour Mistral avec un format spécifique
                prompt_parts = []
                
                # Construction du message système
                system_content = self._build_system_message(
                    response_type=response_type,
                    language=language,
                    model_name=model_name,
                    prompt_prefix=prompt_prefix
                )
                prompt_parts.append(f"<s>[SYSTEM_PROMPT]{system_content}[/SYSTEM_PROMPT]</s>")
                
                # Ajout du contexte si présent
                if context_docs:
                    context_content = await self._build_context_section(
                        context_docs=context_docs,
                        model_name=model_name,
                        include_metadata=include_metadata
                    )
                    if context_content:
                        prompt_parts.append(f"<s>[CONTEXT]{context_content}[/CONTEXT]</s>")
                
                # Ajout de l'historique de conversation
                if conversation_history:
                    history_parts = []
                    history_pairs = []
                    
                    for i in range(0, len(conversation_history), 2):
                        if i + 1 < len(conversation_history):
                            user_msg = conversation_history[i]
                            assistant_msg = conversation_history[i + 1]
                            if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                                pair = f"<s>[INST] {user_msg['content'].strip()} [/INST]{assistant_msg['content'].strip()}</s>"
                                history_pairs.append(pair)
                    
                    if history_pairs:
                        history_content = " ".join(history_pairs)
                        prompt_parts.append(f"<s>[CONVERSATION_HISTORY]{history_content}[/CONVERSATION_HISTORY]</s>")
                
                # Ajout de la question utilisateur finale
                prompt_parts.append(f"<s>[INST] {query.strip()} [/INST]</s>")
                
                # Log pour debugging
                logger.debug("Context docs présents: %s", bool(context_docs))
                logger.debug("Nombre de messages dans l'historique: %s", len(conversation_history) if conversation_history else 0)
                logger.debug("Nombre de parties dans le prompt: %s", len(prompt_parts))
                
                return " ".join(prompt_parts)
                
            else:
                # Traitement standard pour les autres modèles
                messages = []
                
                # Message système
                system_content = self._build_system_message(
                    response_type=response_type,
                    language=language,
                    model_name=model_name,
                    prompt_prefix=prompt_prefix
                )
                messages.append({
                    "role": MessageRole.SYSTEM,
                    "content": template(MessageRole.SYSTEM, system_content)
                })

                # Contexte documentaire
                if context_docs:
                    context_content = await self._build_context_section(
                        context_docs=context_docs,
                        model_name=model_name,
                        include_metadata=include_metadata
                    )
                    if context_content:
                        messages.append({
                            "role": MessageRole.CONTEXT,
                            "content": template(MessageRole.CONTEXT, context_content)
                        })

                # Historique de conversation
                if conversation_history:
                    for msg in conversation_history[-settings.MAX_HISTORY_MESSAGES*2:]:
                        messages.append({
                            "role": msg["role"],
                            "content": template(msg["role"], msg["content"])
                        })

                # Question utilisateur
                messages.append({
                    "role": MessageRole.USER,
                    "content": template(MessageRole.USER, query)
                })

                return messages

        except Exception as e:
            logger.error(f"Erreur construction prompt: {e}")
            return [{
                "role": MessageRole.USER,
                "content": template(MessageRole.USER, query)
            }]