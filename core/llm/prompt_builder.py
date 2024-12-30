# core/llm/prompt_builder.py
from typing import List, Dict, Optional
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("prompt_builder")

class PromptBuilder:
    def build_prompt(
        self,
        query: str,
        context_docs: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
        language: str = "fr"
    ) -> List[Dict[str, str]]:
        """
        Construit le prompt au format messages avec rôles (system, user, assistant).
        """
        messages = []

        # Ajout du rôle système
        messages.append({
            "role": "system",
            "content": settings.SYSTEM_PROMPT
        })

        # Ajout du contexte documentaire
        if context_docs:
            context = self._build_context(context_docs)
            messages.append({
                "role": "system",
                "content": f"Contexte documentaire pertinent : {context}"
            })
        else:
            messages.append({
                "role": "system",
                "content": "Aucun contexte documentaire pertinent n'a été fourni."
            })

        # Ajout de l'historique de la conversation via une méthode séparée
        if conversation_history:
            history_messages = self._build_history(conversation_history)
            messages.extend(history_messages)

        # Ajout de la question de l'utilisateur
        messages.append({
            "role": "user",
            "content": query
        })

        return messages

    def _build_context(self, context_docs: List[Dict]) -> str:
        """
        Construit une description textuelle concise des documents.
        """
        context_parts = []
        for doc in context_docs:
            content = doc.get('content', '').strip()
            source = doc.get('title', 'Document')
            page = doc.get('metadata', {}).get('page', '?')
            if content:
                context_parts.append(f"[{source} (p.{page})] {content}")
        return " ".join(context_parts)

    def _build_history(self, conversation_history: List[Dict]) -> List[Dict[str, str]]:
        """
        Construit l'historique des conversations sous forme de messages.
        """
        history_messages = []
        for entry in conversation_history[-5:]:  # Limiter aux 5 derniers échanges
            if 'query' in entry and 'response' in entry:
                history_messages.append({
                    "role": "user",
                    "content": entry['query']
                })
                history_messages.append({
                    "role": "assistant",
                    "content": entry['response']
                })
        return history_messages

