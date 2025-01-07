# core/document_processing/image_processing.py
from typing import Dict, Optional, Tuple, Union, List
from pdf2image import convert_from_path
from PIL import Image
import fitz
import io
import os
import base64
from datetime import datetime

from core.utils.logger import get_logger
from core.config.config import settings

logger = get_logger("image_processing")

class PDFImageProcessor:
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialise le processeur d'images PDF.
        Args:
            temp_dir: Répertoire temporaire pour le traitement des images
        """
        self.temp_dir = temp_dir or "temp"
        os.makedirs(self.temp_dir, exist_ok=True)

    async def extract_images_from_pdf(
        self,
        pdf_path: str,
        page_numbers: Optional[List[int]] = None
    ) -> List[Dict[str, any]]:
        """
        Extrait toutes les images d'un PDF avec leurs métadonnées.
        """
        images = []
        try:
            doc = fitz.open(pdf_path)
            pages_to_process = page_numbers or range(len(doc))
            
            for page_num in pages_to_process:
                if page_num >= len(doc):
                    continue
                    
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    
                    if base_image:
                        # Optimiser l'image
                        processed_image = await self.process_and_optimize_image(
                            base_image["image"],
                            quality=85
                        )
                        
                        if processed_image:
                            images.append({
                                "data": processed_image["data"],
                                "mime_type": processed_image["mime_type"],
                                "page_number": page_num + 1,
                                "image_index": img_index,
                                "metadata": self._extract_image_metadata(base_image)
                            })
            
            return images
            
        except Exception as e:
            logger.error(f"Erreur extraction images PDF: {e}")
            return []
        finally:
            if 'doc' in locals():
                doc.close()

    async def process_and_optimize_image(
        self,
        image_data: bytes,
        max_size: Tuple[int, int] = (800, 800),
        quality: int = 85
    ) -> Optional[Dict[str, any]]:
        """
        Traite et optimise une image.
        
        Args:
            image_data: Données de l'image
            max_size: Taille maximale (largeur, hauteur)
            quality: Qualité de compression JPEG
            
        Returns:
            Informations sur l'image traitée
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Redimensionnement si nécessaire
            if (image.width > max_size[0] or image.height > max_size[1]):
                image.thumbnail(max_size, Image.LANCZOS)
            
            # Conversion en RGB si nécessaire
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            
            # Compression
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=quality, optimize=True)
            optimized_data = output.getvalue()
            
            return {
                "data": base64.b64encode(optimized_data).decode(),
                "mime_type": "image/jpeg",
                "width": image.width,
                "height": image.height
            }
            
        except Exception as e:
            logger.error(f"Erreur optimisation image: {e}")
            return None

    def _extract_image_metadata(self, image_info: Dict) -> Dict[str, any]:
        """
        Extrait les métadonnées d'une image.
        """
        try:
            metadata = {
                "format": image_info.get("ext", "").upper(),
                "colorspace": image_info.get("colorspace", ""),
                "width": image_info.get("width"),
                "height": image_info.get("height")
            }
            
            # Ajout d'informations supplémentaires si disponibles
            if "xres" in image_info:
                metadata["dpi"] = image_info["xres"]
            if "size" in image_info:
                metadata["size_bytes"] = len(image_info["image"])
                
            return metadata
            
        except Exception as e:
            logger.error(f"Erreur extraction métadonnées: {e}")
            return {}

    def cleanup(self):
        """Nettoie les fichiers temporaires."""
        try:
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        except Exception as e:
            logger.error(f"Erreur nettoyage fichiers temporaires: {e}")
