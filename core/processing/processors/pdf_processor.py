# core/processing/processors/pdf_processor.py
import time
from typing import Dict, Any, Optional, List, Union, BinaryIO
import fitz  # PyMuPDF
from PIL import Image
import io
import tempfile
import os

from ..base_processor import (
    BaseProcessor,
    ProcessingResult,
    ProcessingStage,
    ProcessingError
)
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger(__name__)

class PDFProcessor(BaseProcessor):
    """Processeur pour les fichiers PDF."""

    def _validate_config(self) -> None:
        """Valide la configuration du processeur PDF."""
        required_configs = ["extract_images", "dpi", "max_file_size"]
        missing = [f for f in required_configs if f not in self.config]
        if missing:
            raise ProcessingError(
                f"Configuration manquante: {', '.join(missing)}",
                ProcessingStage.EXTRACTION
            )

    async def process(
        self,
        content: Union[str, bytes, BinaryIO],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Traite un fichier PDF.
        
        Args:
            content: Contenu du PDF (chemin, bytes ou file-like object)
            metadata: Métadonnées optionnelles
            
        Returns:
            Résultat du traitement
        """
        start_time = time.time()
        try:
            # Vérification du type de contenu
            if isinstance(content, str):
                # C'est un chemin de fichier
                pdf_document = fitz.open(content)
            elif isinstance(content, (bytes, BinaryIO)):
                # C'est un contenu binaire ou un fichier
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    if isinstance(content, bytes):
                        temp_file.write(content)
                    else:
                        temp_file.write(content.read())
                    temp_file.flush()
                    pdf_document = fitz.open(temp_file.name)
                    # Nettoyage du fichier temporaire après ouverture
                    os.unlink(temp_file.name)
            else:
                raise ProcessingError(
                    "Type de contenu non supporté",
                    ProcessingStage.EXTRACTION
                )

            # Extraction du texte et des métadonnées
            extracted_text = []
            images_data = []
            pdf_metadata = pdf_document.metadata

            # Traitement page par page
            for page_num, page in enumerate(pdf_document):
                # Extraction du texte
                text = page.get_text()
                if text.strip():
                    extracted_text.append(f"[Page {page_num + 1}]\n{text}")

                # Extraction des images si configuré
                if self.config.get("extract_images", False):
                    images = self._extract_images_from_page(page)
                    if images:
                        images_data.extend([
                            {
                                "page": page_num + 1,
                                "image": img_data,
                                "index": idx
                            }
                            for idx, img_data in enumerate(images)
                        ])

            # Construction des métadonnées enrichies
            enriched_metadata = self._merge_metadata(
                metadata,
                {
                    "pdf_info": {
                        "title": pdf_metadata.get("title", ""),
                        "author": pdf_metadata.get("author", ""),
                        "subject": pdf_metadata.get("subject", ""),
                        "creator": pdf_metadata.get("creator", ""),
                        "page_count": len(pdf_document),
                        "has_images": bool(images_data)
                    },
                    "processing": {
                        "extractor": "PyMuPDF",
                        "version": fitz.__version__,
                        "dpi": self.config["dpi"]
                    }
                }
            )

            # Nettoyage
            pdf_document.close()

            # Mise à jour des métriques
            metrics.increment_counter("pdf_processed")
            if images_data:
                metrics.increment_counter("pdf_images_extracted", len(images_data))

            return ProcessingResult(
                content="\n\n".join(extracted_text),
                metadata=enriched_metadata,
                stage=ProcessingStage.EXTRACTION,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Erreur traitement PDF: {e}")
            metrics.increment_counter("pdf_processing_errors")
            return self._create_error_result(
                error=e,
                stage=ProcessingStage.EXTRACTION,
                metadata=metadata
            )

    def _extract_images_from_page(self, page: fitz.Page) -> List[bytes]:
        """
        Extrait les images d'une page PDF.
        
        Args:
            page: Page PDF à traiter
            
        Returns:
            Liste des images extraites en format bytes
        """
        images = []
        try:
            # Extraction des images
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                try:
                    base_image = page.parent.extract_image(img_info[0])
                    if base_image:
                        image_bytes = base_image["image"]
                        # Conversion et redimensionnement si nécessaire
                        with Image.open(io.BytesIO(image_bytes)) as img:
                            # Redimensionnement selon la config DPI
                            dpi = self.config.get("dpi", 300)
                            if dpi != 300:  # DPI par défaut
                                w, h = img.size
                                new_w = int(w * dpi / 300)
                                new_h = int(h * dpi / 300)
                                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                            
                            # Conversion en bytes
                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format=img.format or 'PNG')
                            images.append(img_byte_arr.getvalue())
                            
                except Exception as img_error:
                    logger.warning(f"Erreur extraction image {img_index}: {img_error}")
                    continue

        except Exception as e:
            logger.error(f"Erreur extraction images de la page: {e}")

        return images

    async def batch_process(
        self,
        contents: List[Union[str, bytes, BinaryIO]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[ProcessingResult]:
        """
        Traite un lot de PDFs.
        
        Args:
            contents: Liste des PDFs à traiter
            metadata: Liste des métadonnées optionnelles
            
        Returns:
            Liste des résultats de traitement
        """
        results = []
        metadata_list = metadata or [None] * len(contents)

        for content, meta in zip(contents, metadata_list):
            result = await self.process(content, meta)
            results.append(result)

        return results