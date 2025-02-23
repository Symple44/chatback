# core/processing/base_processor.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, BinaryIO
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class ProcessingStage(Enum):
    """Étapes de traitement d'un document."""
    EXTRACTION = "extraction"
    CLEANING = "cleaning"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"

class ProcessingError(Exception):
    """Erreur de traitement de document."""
    def __init__(self, message: str, stage: ProcessingStage, details: Optional[Dict] = None):
        self.message = message
        self.stage = stage
        self.details = details or {}
        super().__init__(self.message)

@dataclass
class ProcessingResult:
    """Résultat du traitement d'un document."""
    content: str
    metadata: Dict[str, Any]
    chunks: Optional[List[str]] = None
    embeddings: Optional[List[List[float]]] = None
    stage: ProcessingStage = ProcessingStage.EXTRACTION
    error: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime = datetime.utcnow()

class BaseProcessor(ABC):
    """Classe de base pour les processeurs de documents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise le processeur.
        
        Args:
            config: Configuration du processeur
        """
        self.config = config or {}
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Valide la configuration du processeur."""
        pass

    @abstractmethod
    async def process(
        self,
        content: Union[str, bytes, BinaryIO],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Traite le contenu.
        
        Args:
            content: Contenu à traiter
            metadata: Métadonnées optionnelles
            
        Returns:
            Résultat du traitement
        """
        pass

    @abstractmethod
    async def batch_process(
        self,
        contents: List[Union[str, bytes, BinaryIO]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[ProcessingResult]:
        """
        Traite un lot de contenus.
        
        Args:
            contents: Liste des contenus à traiter
            metadata: Liste des métadonnées optionnelles
            
        Returns:
            Liste des résultats de traitement
        """
        pass

    async def cleanup(self) -> None:
        """Nettoie les ressources utilisées."""
        pass

    def _merge_metadata(
        self,
        original: Optional[Dict[str, Any]],
        new: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fusionne les métadonnées."""
        result = original.copy() if original else {}
        result.update(new)
        result["last_modified"] = datetime.utcnow().isoformat()
        return result

    def _create_error_result(
        self,
        error: Exception,
        stage: ProcessingStage,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Crée un résultat d'erreur."""
        return ProcessingResult(
            content="",
            metadata=metadata or {},
            stage=stage,
            error=str(error),
            processing_time=0.0
        )