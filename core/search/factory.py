# core/search/factory.py
from typing import Dict, Type
from core.utils.logger import get_logger
from .strategies import SearchStrategy, SearchMethod, DisabledSearch
from .implementations import (
    EnhancedRAGSearch,
    EnhancedHybridSearch,
    EnhancedSemanticSearch
)

logger = get_logger("search_factory")

class SearchStrategyFactory:
    """Factory pour la création des stratégies de recherche."""

    # Mapping des stratégies disponibles
    _strategies: Dict[SearchMethod, Type[SearchStrategy]] = {
        SearchMethod.DISABLED: DisabledSearch,
        SearchMethod.RAG: EnhancedRAGSearch,
        SearchMethod.HYBRID: EnhancedHybridSearch,
        SearchMethod.SEMANTIC: EnhancedSemanticSearch
    }

    @classmethod
    def create_strategy(
        cls,
        method: SearchMethod,
        components,
        **kwargs
    ) -> SearchStrategy:
        """
        Crée et retourne une instance de la stratégie demandée.
        
        Args:
            method: Méthode de recherche souhaitée
            components: Composants nécessaires
            **kwargs: Paramètres additionnels
            
        Returns:
            Instance de SearchStrategy
        """
        try:
            # Récupération de la classe de stratégie
            strategy_class = cls._strategies.get(method, DisabledSearch)
            
            # Création de l'instance
            strategy = strategy_class(components)
            
            # Configuration si des paramètres sont fournis
            if kwargs:
                strategy.configure(kwargs)
                
            logger.info(f"Stratégie {method.value} créée avec succès")
            return strategy
            
        except Exception as e:
            logger.error(f"Erreur création stratégie {method.value}: {e}")
            # Fallback vers la stratégie désactivée
            return DisabledSearch(components)