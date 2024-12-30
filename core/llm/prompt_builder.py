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
    ) -> str:
        """Construit le prompt complet."""
        try:
            # Contexte des documents
            context = self._build_context(context_docs) if context_docs else ""
            
            # Historique de conversation
            history = self._build_history(conversation_history) if conversation_history else ""
            
            # Construction du prompt final
            prompt_parts = []
            
            if settings.SYSTEM_PROMPT:
                prompt_parts.append(f"Système: {settings.SYSTEM_PROMPT}")

            if context:
                prompt_parts.append(f"Contexte:\n{context}")

            if history:
                prompt_parts.append(f"Historique:\n{history}")

            prompt_parts.append(f"Question: {query}")
            prompt_parts.append("Réponse:")

            return "\n\n".join(prompt_parts)

        except Exception as e:
            logger.error(f"Erreur construction prompt: {e}")
            return query

    def _build_context(self, context_docs: List[Dict]) -> str:
        """Construit la section contexte du prompt."""
        context_parts = []
        for doc in context_docs:
            content = doc.get('content', '').strip()
            if content:
                source = doc.get('title', 'Document')
                page = doc.get('metadata', {}).get('page', '?')
                context_parts.append(f"[{source} (p.{page})]\n{content}")
        return "\n\n".join(context_parts)

    def _build_history(self, conversation_history: List[Dict]) -> str:
        """Construit la section historique du prompt."""
        history_parts = []
        for entry in conversation_history[-5:]:
            if isinstance(entry, dict) and 'query' in entry and 'response' in entry:
                history_parts.append(f"Q: {entry['query']}\nR: {entry['response']}")
        return "\n\n".join(history_parts)
