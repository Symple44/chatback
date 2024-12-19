# core/document_processing/extractor.py
from typing import List, Dict, Optional, Any, Tuple, Generator
from dataclasses import dataclass
import fitz  # PyMuPDF
from pathlib import Path
import os
import io
import asyncio
from PIL import Image
import numpy as np
import base64
from collections import defaultdict

from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .image_processing import PDFImageProcessor

logger = get_logger("document_extractor")

@dataclass
class DocumentFragment:
    """Fragment de document avec contexte et images."""
    text: str
    page_num: int
    images: List[Dict[str, Any]]
    confidence: float
    source: str
    context_before: str = ""
    context_after: str = ""
    metadata: Dict[str, Any] = None

class DocumentExtractor:
    def __init__(self):
        """Initialise l'extracteur de documents."""
        self.context_window = 200  # Taille de la fenêtre de contexte
        self.image_max_size = (800, 800)  # Taille maximale des images
        self.supported_formats = {'.pdf', '.docx', '.txt'}
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.image_processor = PDFImageProcessor()

    async def extract_document_content(
        self,
        file_path: str,
        extract_images: bool = True
    ) -> Dict[str, Any]:
        """
        Extrait le contenu complet d'un document.
        
        Args:
            file_path: Chemin du fichier
            extract_images: Si True, extrait aussi les images
            
        Returns:
            Contenu extrait avec métadonnées
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Format de fichier non supporté: {file_ext}")

            with metrics.timer("document_extraction"):
                if file_ext == '.pdf':
                    return await self._extract_pdf_content(file_path, extract_images)
                elif file_ext == '.docx':
                    return await self._extract_docx_content(file_path, extract_images)
                else:
                    return await self._extract_text_content(file_path)

        except Exception as e:
            logger.error(f"Erreur extraction document {file_path}: {e}")
            metrics.increment_counter("document_extraction_errors")
            raise

    async def _extract_pdf_content(
        self,
        file_path: str,
        extract_images: bool
    ) -> Dict[str, Any]:
        """Extrait le contenu d'un PDF."""
        try:
            doc = fitz.open(file_path)
            content = {
                "pages": [],
                "metadata": doc.metadata,
                "total_pages": len(doc),
                "images": []
            }

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_content = {
                    "number": page_num + 1,
                    "text": page.get_text(),
                    "images": []
                }

                if extract_images:
                    # Extraction des images de la page
                    for img_index, img in enumerate(page.get_images()):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            
                            if base_image:
                                image_data = await self._process_image(
                                    base_image["image"],
                                    f"page_{page_num + 1}_img_{img_index + 1}"
                                )
                                if image_data:
                                    page_content["images"].append(image_data)
                                    content["images"].append({
                                        **image_data,
                                        "page": page_num + 1
                                    })
                        except Exception as e:
                            logger.error(f"Erreur extraction image: {e}")

                content["pages"].append(page_content)

            return content

        except Exception as e:
            logger.error(f"Erreur extraction PDF {file_path}: {e}")
            raise

    async def _process_image(
        self,
        image_data: bytes,
        image_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Traite et optimise une image.
        
        Args:
            image_data: Données de l'image
            image_name: Nom de l'image
            
        Returns:
            Informations sur l'image traitée
        """
        try:
            # Conversion en image PIL
            with Image.open(io.BytesIO(image_data)) as img:
                # Redimensionnement si nécessaire
                if img.size[0] > self.image_max_size[0] or img.size[1] > self.image_max_size[1]:
                    img.thumbnail(self.image_max_size, Image.LANCZOS)

                # Conversion en RGB si nécessaire
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')

                # Sauvegarde optimisée
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=85, optimize=True)
                image_data = output.getvalue()

                return {
                    "name": image_name,
                    "data": base64.b64encode(image_data).decode('utf-8'),
                    "size": len(image_data),
                    "dimensions": img.size,
                    "format": "JPEG"
                }

        except Exception as e:
            logger.error(f"Erreur traitement image {image_name}: {e}")
            return None

    async def extract_relevant_fragments(
        self,
        doc_path: str,
        keywords: List[str],
        max_fragments: int = 3,
        context_size: int = 200
    ) -> List[DocumentFragment]:
        """
        Extrait les fragments pertinents d'un document.
        
        Args:
            doc_path: Chemin du document
            keywords: Mots-clés à rechercher
            max_fragments: Nombre maximum de fragments
            context_size: Taille du contexte en caractères
            
        Returns:
            Liste des fragments pertinents
        """
        try:
            fragments = []
            doc = fitz.open(doc_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                # Recherche des positions des mots-clés
                for keyword in keywords:
                    positions = self._find_keyword_positions(text.lower(), keyword.lower())

                    for pos in positions:
                        # Extraction du contexte
                        start = max(0, pos - context_size)
                        end = min(len(text), pos + len(keyword) + context_size)

                        fragment_text = text[start:end]
                        images = await self._extract_page_images(page, start, end)

                        fragment = DocumentFragment(
                            text=fragment_text,
                            page_num=page_num + 1,
                            images=images,
                            confidence=self._calculate_relevance(fragment_text, keywords),
                            source=doc_path,
                            context_before=text[max(0, start-100):start],
                            context_after=text[end:min(len(text), end+100)],
                            metadata={
                                "extracted_at": datetime.utcnow().isoformat(),
                                "keywords": keywords
                            }
                        )
                        
                        fragments.append(fragment)

            # Tri et limitation des fragments
            fragments.sort(key=lambda x: x.confidence, reverse=True)
            return fragments[:max_fragments]

        except Exception as e:
            logger.error(f"Erreur extraction fragments {doc_path}: {e}")
            return []

    def _find_keyword_positions(self, text: str, keyword: str) -> List[int]:
        """Trouve toutes les positions d'un mot-clé."""
        positions = []
        start = 0
        while True:
            pos = text.find(keyword, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions

    async def _extract_page_images(
        self,
        page: fitz.Page,
        start_pos: int,
        end_pos: int
    ) -> List[Dict[str, Any]]:
        """Extrait les images près d'une portion de texte."""
        try:
            images = []
            for img in page.get_images():
                xref = img[0]
                bbox = page.get_image_bbox(img)

                # Vérifie si l'image est proche du texte
                if self._is_image_near_text(bbox, start_pos, end_pos, page):
                    base_image = page.parent.extract_image(xref)
                    if base_image:
                        image_data = await self._process_image(
                            base_image["image"],
                            f"img_{xref}"
                        )
                        if image_data:
                            image_data["position"] = {
                                "x": bbox.x0,
                                "y": bbox.y0,
                                "width": bbox.width,
                                "height": bbox.height
                            }
                            images.append(image_data)

            return images

        except Exception as e:
            logger.error(f"Erreur extraction images page: {e}")
            return []

    def _is_image_near_text(
        self,
        img_bbox: fitz.Rect,
        start_pos: int,
        end_pos: int,
        page: fitz.Page,
        max_distance: int = 100
    ) -> bool:
        """Vérifie si une image est proche d'une portion de texte."""
        try:
            # Récupère les coordonnées du texte
            text_areas = page.get_text("dict")["blocks"]
            text_rects = []
            
            for block in text_areas:
                if isinstance(block, dict) and "bbox" in block:
                    text_rects.append(fitz.Rect(block["bbox"]))

            # Vérifie la distance avec chaque zone de texte
            for text_rect in text_rects:
                distance = min(
                    abs(img_bbox.y0 - text_rect.y1),
                    abs(text_rect.y0 - img_bbox.y1)
                )
                if distance <= max_distance:
                    return True

            return False

        except Exception as e:
            logger.error(f"Erreur vérification proximité image: {e}")
            return False

    def _calculate_relevance(self, text: str, keywords: List[str]) -> float:
        """Calcule la pertinence d'un fragment."""
        try:
            text = text.lower()
            total_score = 0
            word_count = len(text.split())

            for keyword in keywords:
                keyword = keyword.lower()
                # Compte les occurrences
                count = text.count(keyword)
                # Normalise le score par la longueur du texte
                score = count * (100 / max(word_count, 1))
                total_score += score

            # Normalise le score final
            return min(total_score / len(keywords), 1.0)

        except Exception as e:
            logger.error(f"Erreur calcul pertinence: {e}")
            return 0.0

    async def cleanup(self):
        """Nettoie les ressources temporaires."""
        try:
            # Suppression des fichiers temporaires
            for file in self.temp_dir.glob("*"):
                file.unlink()
            logger.info("Nettoyage des ressources terminé")
        except Exception as e:
            logger.error(f"Erreur nettoyage ressources: {e}")
