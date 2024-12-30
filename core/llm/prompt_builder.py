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
        """Construit le prompt pour la génération."""
        prompt_parts = []

        # Ajout du système prompt avec une directive explicite
        prompt_parts.append(f"Système: {settings.SYSTEM_PROMPT}")
        prompt_parts.append("Répondez de manière concise et pertinente en vous basant principalement sur les documents fournis.")

        # Contexte des documents
        if context_docs:
            prompt_parts.append(f"Contexte documentaire:\n{self._build_context(context_docs)}")
        else:
            prompt_parts.append("Aucun contexte documentaire pertinent n'a été fourni.")

        # Historique de la conversation
        if conversation_history:
            prompt_parts.append(f"Historique de la conversation (5 derniers échanges):\n{self._build_history(conversation_history)}")

        # Question utilisateur
        prompt_parts.append(f"Question utilisateur: {query}")

        # Préfixe de réponse
        prompt_parts.append("Réponse (maximum 3 paragraphes) :")

        return "\n\n".join(prompt_parts)

    def _build_context(self, context_docs: List[Dict]) -> str:
        """Construit la section contexte du prompt."""
        context_parts = []
        for doc in context_docs:
            content = doc.get('content', '').strip()
            source = doc.get('title', 'Document')
            page = doc.get('metadata', {}).get('page', '?')
            if content:
                context_parts.append(f"[Source: {source} (p.{page})]\n{content}")
        return "\n\n".join(context_parts)

    def _build_history(self, conversation_history: List[Dict]) -> str:
        """Construit la section historique du prompt."""
        history_parts = []
        for entry in conversation_history[-5:]:  # Limite aux 5 derniers échanges
            if 'query' in entry and 'response' in entry:
                history_parts.append(f"Q: {entry['query']}\nR: {entry['response']}")
        return "\n\n".join(history_parts)
