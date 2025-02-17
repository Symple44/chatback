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
        SearchMethod.SEMANTIC: EnhancedSemanticS-arch
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
            components: Composants nécessaires à la stratégie
            **kwargs: Paramètres supplémentaires pour la stratégie

        Returns:
            Instance de la stratégie de recherche
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

    @classmethod
    def register_strategy(
        cls,
        method: SearchMethod,
        strategy_class: Type[SearchStrategy]
    ) -> None:
        """
        Enregistre une nouvelle stratégie dans la factory.
        
        Args:
            method: Identifiant de la méthode
            strategy_class: Classe d'implémentation de la stratégie
        """
        if not issubclass(strategy_class, SearchStrategy):
            raise ValueError(f"La classe {strategy_class.__name__} doit hériter de SearchStrategy")
            
        cls._strategies[method] = strategy_class
        logger.info(f"Stratégie {method.value} enregistrée: {strategy_class.__name__}")

    @classmethod
    def get_available_methods(cls) -> List[SearchMethod]:
        """Retourne la liste des méthodes disponibles."""
        return list(cls._strategies.keys())

    @classmethod
    def get_strategy_class(cls, method: SearchMethod) -> Type[SearchStrategy]:
        """Retourne la classe de stratégie pour une méthode donnée."""
        return cls._strategies.get(method, DisabledSearch)