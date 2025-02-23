# core/processing/chunking/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Chunk:
    """Représente un fragment de document."""
    content: str
    index: int
    metadata: Dict[str, Any]
    start_char: int
    end_char: int
    embedding: Optional[List[float]] = None

class ChunkingStrategy(ABC):
    """Stratégie de base pour le découpage de documents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise la stratégie de chunking.
        
        Args:
            config: Configuration optionnelle pour la stratégie
        """
        self.config = config or {}
        self.validate_config()

    @abstractmethod
    def validate_config(self) -> None:
        """Valide la configuration de la stratégie."""
        pass

    @abstractmethod
    def create_chunks(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Découpe le texte en chunks.
        
        Args:
            text: Texte à découper
            metadata: Métadonnées à inclure dans chaque chunk
            
        Returns:
            Liste des chunks créés
        """
        pass

    def _create_chunk_metadata(
        self,
        base_metadata: Dict[str, Any],
        chunk_specific: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Crée les métadonnées pour un chunk.
        
        Args:
            base_metadata: Métadonnées de base du document
            chunk_specific: Métadonnées spécifiques au chunk
            
        Returns:
            Métadonnées fusionnées
        """
        metadata = base_metadata.copy() if base_metadata else {}
        metadata.update(chunk_specific)
        metadata["chunking"] = {
            "strategy": self.__class__.__name__,
            "timestamp": datetime.utcnow().isoformat(),
            **self.config
        }
        return metadata

class ChunkingError(Exception):
    """Erreur lors du découpage en chunks."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)