# core/document_processing/extractor.py
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import fitz
from core.utils.logger import get_logger
from .image_processing import PDFImageProcessor

logger = get_logger(__name__)

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

class DocumentExtractor:
    def __init__(self):
        """Initialise l'extracteur avec son processeur d'images."""
        self.context_window = 200
        self.image_processor = PDFImageProcessor()
        self.image_max_size = (800, 800)

    async def extract_relevant_fragments(
        self,
        doc_path: str,
        keywords: List[str],
        max_fragments: int = 3
    ) -> List[DocumentFragment]:
        try:
            fragments = []
            doc = fitz.open(doc_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                for keyword in keywords:
                    positions = self._find_keyword_positions(text.lower(), keyword.lower())
                    
                    for pos in positions:
                        # Extraction du contexte textuel
                        start = max(0, pos - self.context_window)
                        end = min(len(text), pos + len(keyword) + self.context_window)
                        
                        fragment_text = text[start:end]
                        
                        # Extraction des images avec tous les paramètres requis
                        fragment_images = await self._process_fragment_images(
                            page=page,
                            text=fragment_text,
                            start=start,
                            end=end
                        )
                        
                        # Création du fragment
                        fragment = DocumentFragment(
                            text=fragment_text,
                            page_num=page_num + 1,
                            images=fragment_images,
                            confidence=self._calculate_relevance(fragment_text, keywords),
                            source=doc_path,
                            context_before=text[max(0, start-100):start],
                            context_after=text[end:min(len(text), end+100)]
                        )
                        
                        fragments.append(fragment)

            # Tri et limitation des fragments
            fragments.sort(key=lambda x: x.confidence, reverse=True)
            return fragments[:max_fragments]
            
        except Exception as e:
            logger.error(f"Erreur extraction fragments: {e}")
            return []

    async def _process_fragment_images(
        self,
        page: fitz.Page,
        text: str,
        start: int,
        end: int
    ) -> List[Dict[str, Any]]:
        """
        Traite les images associées à un fragment de texte.
        """
        try:
            images = []
            
            # Récupérer les images de la page d'une manière différente
            for image in page.get_images(full=True):
                try:
                    xref = image[0]  # le numéro de référence de l'image
                    
                    # Extraire l'image
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Convertir en PNG si l'image a des couleurs
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    # Obtenir les données de l'image
                    img_data = pix.tobytes()
                    
                    # Encoder en base64 pour le transfert
                    import base64
                    encoded_image = base64.b64encode(img_data).decode()
                    
                    # Obtenir la position de l'image
                    bbox = page.get_image_bbox(image)
                    
                    images.append({
                        "data": encoded_image,
                        "type": "png",
                        "position": {
                            "x": bbox.x0,
                            "y": bbox.y0,
                            "width": bbox.width,
                            "height": bbox.height
                        }
                    })
                    
                    # Libérer la mémoire
                    pix = None
                    
                except Exception as e:
                    logger.error(f"Erreur traitement image spécifique: {e}")
                    continue
                    
            return images
            
        except Exception as e:
            logger.error(f"Erreur traitement images fragment: {e}")
            return []

    def _find_keyword_positions(self, text: str, keyword: str) -> List[int]:
        """Trouve les positions d'un mot-clé."""
        positions = []
        start = 0
        while True:
            pos = text.find(keyword, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions

    def _is_image_near_text(
        self,
        img_rect: fitz.Rect,
        text: str,
        page: fitz.Page,
        max_distance: int = 100
    ) -> bool:
        """Vérifie la proximité image-texte."""
        text_areas = page.search_for(text)
        
        for text_rect in text_areas:
            distance = min(
                abs(img_rect.y0 - text_rect.y1),
                abs(text_rect.y0 - img_rect.y1)
            )
            if distance <= max_distance:
                return True
        return False

    def _calculate_relevance(self, text: str, keywords: List[str]) -> float:
        """Calcule la pertinence du fragment."""
        text = text.lower()
        score = 0
        
        for keyword in keywords:
            keyword = keyword.lower()
            count = text.count(keyword)
            score += count * (1 / len(text)) * 100
            
        return min(score, 1.0)