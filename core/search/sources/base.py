from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..interfaces import (
    SearchContext, 
    SearchResultBase,
    SearchSourceCapabilities,
    SourceType
)

class BaseDataSource(ABC):
    """Classe de base abstraite pour toutes les sources de données."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise la source de données.
        
        Args:
            config: Configuration optionnelle pour la source
        """
        self.config = config or {}
        self._initialized = False
        self._capabilities: List[SearchSourceCapabilities] = []

    @abstractmethod
    async def initialize(self) -> None:
        """Initialise la connexion et les ressources de la source."""
        pass

    @abstractmethod
    async def search(self, context: SearchContext) -> List[SearchResultBase]:
        """
        Effectue une recherche dans la source.
        
        Args:
            context: Contexte de recherche contenant la requête et les paramètres
            
        Returns:
            Liste des résultats de recherche
        """
        if not self._initialized:
            await self.initialize()

    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[SearchResultBase]:
        """
        Récupère un document par son ID.
        
        Args:
            id: Identifiant du document
            
        Returns:
            Document trouvé ou None
        """
        pass

    @abstractmethod
    def get_source_type(self) -> SourceType:
        """Retourne le type de la source de données."""
        pass

    def get_capabilities(self) -> List[SearchSourceCapabilities]:
        """Retourne les capacités supportées par la source."""
        return self._capabilities

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état de santé de la source.
        
        Returns:
            Dictionnaire contenant les informations de santé
        """
        pass

    async def cleanup(self) -> None:
        """Nettoie les ressources utilisées par la source."""
        self._initialized = False

    def validate_config(self) -> None:
        """Valide la configuration de la source."""
        pass

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Met à jour la configuration de la source.
        
        Args:
            config: Nouvelle configuration à fusionner
        """
        self.config.update(config)
        self.validate_config()

    async def _validate_connection(self) -> bool:
        """
        Valide la connexion à la source.
        
        Returns:
            True si la connexion est valide
        """
        try:
            health = await self.health_check()
            return health.get("status") == "healthy"
        except Exception:
            return False