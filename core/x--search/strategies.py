# core/search/strategies.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class SearchMethod(str, Enum):
    """Types de méthodes de recherche disponibles."""
    DISABLED = "disabled"  # Pas de recherche de contexte
    RAG = "rag"           # Retrieval Augmented Generation classique
    HYBRID = "hybrid"     # Combinaison de RAG et recherche sémantique
    SEMANTIC = "semantic" # Recherche purement sémantique

@dataclass
class SearchResult:
    """Structure standardisée pour les résultats de recherche."""
    content: str
    score: float
    metadata: Dict[str, Any]
    source_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    timestamp: str = datetime.utcnow().isoformat()

class SearchStrategy(ABC):
    """Classe de base pour toutes les stratégies de recherche."""
    
    def __init__(self, components):
        """Initialise la stratégie avec les composants nécessaires."""
        self.components = components
        self.es_client = components.es_client
        self.model = components.model
        self._search_params = {}

    @abstractmethod
    async def search(
        self,
        query: str,
        metadata_filter: Optional[Dict] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Méthode abstraite pour la recherche.
        
        Args:
            query: Texte de la requête
            metadata_filter: Filtres optionnels
            **kwargs: Paramètres de recherche supplémentaires
            
        Returns:
            Liste des résultats de recherche
        """
        pass

    def configure(self, params: Dict[str, Any]) -> None:
        """Configure les paramètres de la stratégie."""
        self._search_params.update(params)

    async def create_embedding(self, text: str) -> Optional[List[float]]:
        """Crée un embedding pour le texte donné."""
        try:
            if not self.model:
                return None
            return await self.model.create_embedding(text)
        except Exception as e:
            logger.error(f"Erreur création embedding: {e}")
            return None

    def get_params(self) -> Dict[str, Any]:
        """Retourne les paramètres actuels de la stratégie."""
        return self._search_params.copy()

class DisabledSearch(SearchStrategy):
    """Stratégie représentant une recherche désactivée."""

    async def search(
        self,
        query: str,
        metadata_filter: Optional[Dict] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Retourne une liste vide car la recherche est désactivée."""
        return []

class SearchContext:
    """Contexte d'exécution pour une recherche."""
    
    def __init__(
        self,
        query: str,
        strategy: SearchStrategy,
        metadata_filter: Optional[Dict] = None,
        params: Optional[Dict] = None
    ):
        self.query = query
        self.strategy = strategy
        self.metadata_filter = metadata_filter or {}
        self.params = params or {}
        self.start_time = datetime.utcnow()
        self.results: List[SearchResult] = []
        self.error: Optional[Exception] = None

    async def execute(self) -> 'SearchContext':
        """Exécute la recherche avec la stratégie configurée."""
        try:
            # Configuration de la stratégie avec les paramètres
            self.strategy.configure(self.params)
            
            # Exécution de la recherche
            self.results = await self.strategy.search(
                query=self.query,
                metadata_filter=self.metadata_filter,
                **self.params
            )
            
            return self
            
        except Exception as e:
            self.error = e
            logger.error(f"Erreur exécution recherche: {e}")
            return self

    @property
    def duration(self) -> float:
        """Calcule la durée de la recherche."""
        return (datetime.utcnow() - self.start_time).total_seconds()

    @property
    def success(self) -> bool:
        """Indique si la recherche s'est bien déroulée."""
        return not self.error

    def to_dict(self) -> Dict[str, Any]:
        """Convertit le contexte en dictionnaire."""
        return {
            "query": self.query,
            "strategy": self.strategy.__class__.__name__,
            "params": self.params,
            "duration": self.duration,
            "success": self.success,
            "result_count": len(self.results),
            "error": str(self.error) if self.error else None,
            "timestamp": self.start_time.isoformat()
        }