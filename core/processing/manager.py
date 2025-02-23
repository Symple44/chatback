# core/processing/manager.py
from typing import Dict, Any, Optional, List, Type, Union
from datetime import datetime
import asyncio

from .pipeline import ProcessingPipeline, PipelineResult
from .base_processor import BaseProcessor
from .chunking import ChunkingStrategy, TokenChunker
from .processors.pdf_processor import PDFProcessor
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger(__name__)

class ProcessingManager:
    """Gestionnaire des pipelines de traitement de documents."""

    def __init__(self, components):
        """
        Initialise le gestionnaire de traitement.
        
        Args:
            components: Composants partagés de l'application
        """
        self.components = components
        self.pipelines: Dict[str, ProcessingPipeline] = {}
        self._initialized = False
        self._processing_queue = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._max_workers = 3
        self._queue_processor_task = None

    async def initialize(self) -> None:
        """Initialise le gestionnaire et les pipelines par défaut."""
        if self._initialized:
            return

        try:
            # Création des pipelines par défaut
            await self._create_default_pipelines()
            
            # Démarrage des workers de traitement
            self._start_queue_processor()
            
            self._initialized = True
            logger.info("ProcessingManager initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation ProcessingManager: {e}")
            raise

    async def _create_default_pipelines(self) -> None:
        """Crée les pipelines de traitement par défaut."""
        try:
            # Pipeline PDF par défaut
            pdf_config = {
                "extract_images": True,
                "dpi": 300,
                "max_file_size": 100 * 1024 * 1024  # 100MB
            }
            pdf_processor = PDFProcessor(pdf_config)

            # Configuration du chunking par défaut
            chunking_config = {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "min_chunk_size": 100
            }
            chunking_strategy = TokenChunker(chunking_config)

            # Création du pipeline PDF
            self.pipelines["pdf"] = ProcessingPipeline(
                processor=pdf_processor,
                chunking_strategy=chunking_strategy,
                embedding_model=self.components.model,
                config={
                    "max_chunk_size": 1000,
                    "min_chunk_size": 100,
                    "parallel_processing": True,
                    "max_retries": 3
                }
            )

            logger.info("Pipelines par défaut créés")

        except Exception as e:
            logger.error(f"Erreur création pipelines par défaut: {e}")
            raise

    def _start_queue_processor(self) -> None:
        """Démarre le processeur de file d'attente."""
        if self._queue_processor_task is None:
            self._queue_processor_task = asyncio.create_task(self._process_queue())
            logger.info("Processeur de file d'attente démarré")

    async def _process_queue(self) -> None:
        """Traite les éléments de la file d'attente."""
        while True:
            try:
                # Attente d'un élément dans la queue
                queue_item = await self._processing_queue.get()
                
                if queue_item is None:  # Signal d'arrêt
                    break

                content, metadata, pipeline_name, future = queue_item
                
                try:
                    # Traitement du document
                    result = await self.process_document(content, metadata, pipeline_name)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self._processing_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur dans le processeur de queue: {e}")
                await asyncio.sleep(1)  # Éviter une boucle d'erreurs trop rapide

    async def process_document(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        pipeline_name: str = "pdf"
    ) -> PipelineResult:
        """
        Traite un document avec le pipeline spécifié.
        
        Args:
            content: Contenu à traiter
            metadata: Métadonnées optionnelles
            pipeline_name: Nom du pipeline à utiliser
            
        Returns:
            Résultat du traitement
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Vérification du pipeline
            if pipeline_name not in self.pipelines:
                raise ValueError(f"Pipeline non trouvé: {pipeline_name}")

            pipeline = self.pipelines[pipeline_name]
            
            # Enrichissement des métadonnées
            processing_metadata = self._create_processing_metadata(metadata)

            # Traitement du document
            result = await pipeline.process(content, processing_metadata)
            
            # Mise à jour des métriques
            metrics.increment_counter(f"documents_processed_{pipeline_name}")
            if result.errors:
                metrics.increment_counter(f"processing_errors_{pipeline_name}")

            return result

        except Exception as e:
            logger.error(f"Erreur traitement document: {e}")
            metrics.increment_counter("processing_errors")
            raise

    async def process_document_async(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        pipeline_name: str = "pdf"
    ) -> asyncio.Future[PipelineResult]:
        """
        Enqueue un document pour traitement asynchrone.
        
        Args:
            content: Contenu à traiter
            metadata: Métadonnées optionnelles
            pipeline_name: Nom du pipeline à utiliser
            
        Returns:
            Future contenant le résultat du traitement
        """
        future = asyncio.Future()
        await self._processing_queue.put((content, metadata, pipeline_name, future))
        return future

    def _create_processing_metadata(
        self,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Crée les métadonnées de traitement."""
        processing_metadata = metadata.copy() if metadata else {}
        processing_metadata.update({
            "processing_info": {
                "start_time": datetime.utcnow().isoformat(),
                "version": self.components.config.VERSION,
                "processor_id": id(self)
            }
        })
        return processing_metadata

    async def add_pipeline(
        self,
        name: str,
        processor: BaseProcessor,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Ajoute un nouveau pipeline de traitement.
        
        Args:
            name: Nom du pipeline
            processor: Processeur de documents
            chunking_strategy: Stratégie de chunking optionnelle
            config: Configuration du pipeline
        """
        if name in self.pipelines:
            raise ValueError(f"Pipeline {name} existe déjà")

        if chunking_strategy is None:
            chunking_strategy = TokenChunker({
                "chunk_size": 500,
                "chunk_overlap": 50,
                "min_chunk_size": 100
            })

        pipeline = ProcessingPipeline(
            processor=processor,
            chunking_strategy=chunking_strategy,
            embedding_model=self.components.model,
            config=config
        )

        self.pipelines[name] = pipeline
        logger.info(f"Pipeline {name} ajouté avec succès")

    async def cleanup(self) -> None:
        """Nettoie les ressources du gestionnaire."""
        try:
            # Arrêt du processeur de queue
            if self._queue_processor_task:
                await self._processing_queue.put(None)  # Signal d'arrêt
                await self._queue_processor_task
                self._queue_processor_task = None

            # Nettoyage des pipelines
            for name, pipeline in self.pipelines.items():
                try:
                    await pipeline.processor.cleanup()
                except Exception as e:
                    logger.error(f"Erreur nettoyage pipeline {name}: {e}")

            self.pipelines.clear()
            self._initialized = False
            
            logger.info("ProcessingManager nettoyé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage ProcessingManager: {e}")
            raise