# core/processing/pipeline.py
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass
from datetime import datetime
import asyncio

from .base_processor import (
    BaseProcessor,
    ProcessingResult,
    ProcessingStage,
    ProcessingError
)
from .chunking import ChunkingStrategy, Chunk
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger(__name__)

@dataclass
class PipelineResult:
    """Résultat complet du pipeline de traitement."""
    original_content: ProcessingResult
    chunks: List[Chunk]
    metadata: Dict[str, Any]
    errors: List[str]
    processing_time: float
    timestamp: datetime = datetime.utcnow()

class ProcessingPipeline:
    """Pipeline de traitement des documents."""

    def __init__(
        self,
        processor: BaseProcessor,
        chunking_strategy: ChunkingStrategy,
        embedding_model = None,  # Type sera défini lors de l'intégration complète
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise le pipeline.
        
        Args:
            processor: Processeur de documents
            chunking_strategy: Stratégie de chunking
            embedding_model: Modèle pour créer les embeddings
            config: Configuration du pipeline
        """
        self.processor = processor
        self.chunking_strategy = chunking_strategy
        self.embedding_model = embedding_model
        self.config = config or {}
        self._validate_config()

    def _validate_config(self) -> None:
        """Valide la configuration du pipeline."""
        required = {
            "max_chunk_size": int,
            "min_chunk_size": int,
            "parallel_processing": bool,
            "max_retries": int
        }
        
        for key, expected_type in required.items():
            if key not in self.config:
                self.config[key] = self._get_default_config(key)
            elif not isinstance(self.config[key], expected_type):
                raise ValueError(f"Configuration invalide pour {key}: type attendu {expected_type}")

    def _get_default_config(self, key: str) -> Any:
        """Retourne la configuration par défaut pour une clé."""
        defaults = {
            "max_chunk_size": 1000,
            "min_chunk_size": 100,
            "parallel_processing": True,
            "max_retries": 3
        }
        return defaults.get(key)

    async def process(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Traite un document à travers le pipeline complet.
        
        Args:
            content: Contenu à traiter
            metadata: Métadonnées optionnelles
            
        Returns:
            Résultat complet du pipeline
        """
        start_time = datetime.utcnow()
        errors = []

        try:
            # 1. Traitement initial du document
            with metrics.timer("document_processing"):
                processing_result = await self.processor.process(content, metadata)
                if processing_result.error:
                    errors.append(f"Erreur de traitement: {processing_result.error}")
                    return self._create_error_result(processing_result, start_time)

            # 2. Chunking du contenu
            with metrics.timer("document_chunking"):
                chunks = self.chunking_strategy.create_chunks(
                    processing_result.content,
                    processing_result.metadata
                )

            # 3. Création des embeddings si un modèle est disponible
            if self.embedding_model and chunks:
                with metrics.timer("embedding_creation"):
                    await self._create_embeddings(chunks)

            # 4. Enrichissement des métadonnées
            final_metadata = self._enrich_metadata(
                processing_result.metadata,
                {
                    "pipeline_info": {
                        "processor": self.processor.__class__.__name__,
                        "chunking_strategy": self.chunking_strategy.__class__.__name__,
                        "has_embeddings": bool(self.embedding_model),
                        "chunk_count": len(chunks),
                        "config": self.config
                    }
                }
            )

            # Création du résultat final
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Mise à jour des métriques
            metrics.increment_counter("documents_processed")
            metrics.increment_counter("chunks_created", len(chunks))
            metrics.record_value("processing_time", processing_time)

            return PipelineResult(
                original_content=processing_result,
                chunks=chunks,
                metadata=final_metadata,
                errors=errors,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Erreur dans le pipeline: {e}")
            metrics.increment_counter("pipeline_errors")
            errors.append(str(e))
            return self._create_error_result(None, start_time, str(e))

    async def _create_embeddings(self, chunks: List[Chunk]) -> None:
        """Crée les embeddings pour une liste de chunks."""
        if not chunks:
            return

        try:
            # Préparation des textes
            texts = [chunk.content for chunk in chunks]
            
            # Création des embeddings en parallèle si configuré
            if self.config["parallel_processing"]:
                embeddings = await asyncio.gather(*[
                    self.embedding_model.create_embedding(text)
                    for text in texts
                ])
            else:
                embeddings = []
                for text in texts:
                    embedding = await self.embedding_model.create_embedding(text)
                    embeddings.append(embedding)

            # Attribution des embeddings aux chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

        except Exception as e:
            logger.error(f"Erreur création embeddings: {e}")
            metrics.increment_counter("embedding_errors")

    def _create_error_result(
        self,
        processing_result: Optional[ProcessingResult],
        start_time: datetime,
        error: Optional[str] = None
    ) -> PipelineResult:
        """Crée un résultat d'erreur."""
        return PipelineResult(
            original_content=processing_result or ProcessingResult(
                content="",
                metadata={},
                error=error,
                stage=ProcessingStage.EXTRACTION,
                processing_time=0.0
            ),
            chunks=[],
            metadata={"error": error} if error else {},
            errors=[error] if error else [],
            processing_time=(datetime.utcnow() - start_time).total_seconds()
        )

    def _enrich_metadata(
        self,
        original: Dict[str, Any],
        new: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enrichit les métadonnées avec de nouvelles informations."""
        metadata = original.copy()
        for key, value in new.items():
            if key in metadata and isinstance(metadata[key], dict):
                metadata[key].update(value)
            else:
                metadata[key] = value
                
        metadata["last_modified"] = datetime.utcnow().isoformat()
        return metadata

    async def batch_process(
        self,
        contents: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[PipelineResult]:
        """Traite un lot de documents."""
        if not contents:
            return []

        metadata_list = metadata or [None] * len(contents)
        
        if self.config["parallel_processing"]:
            # Traitement parallèle
            tasks = [
                self.process(content, meta)
                for content, meta in zip(contents, metadata_list)
            ]
            return await asyncio.gather(*tasks)
        else:
            # Traitement séquentiel
            results = []
            for content, meta in zip(contents, metadata_list):
                result = await self.process(content, meta)
                results.append(result)
            return results