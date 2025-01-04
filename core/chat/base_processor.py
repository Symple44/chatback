# core/chat/base_processor.py
from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseProcessor(ABC):
    """Classe de base pour tous les processeurs."""
    
    def __init__(self, components):
        self.components = components

    @abstractmethod
    async def process_message(
        self,
        request: Dict,
        context: Optional[Dict] = None
    ) -> Dict:
        """Traite un message et retourne une réponse."""
        pass

    def _calculate_confidence(self, docs: list) -> float:
        """Calcule le score de confiance basé sur les documents."""
        if not docs:
            return 0.0
        return sum(doc.get('score', 0) for doc in docs) / len(docs)
