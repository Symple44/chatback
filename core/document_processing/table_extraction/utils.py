# core/document_processing/table_extraction/utils.py
from typing import List, Dict, Any, Optional, Union, Tuple, BinaryIO, AsyncGenerator
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import io
import time
import hashlib
import numpy as np
import cv2
import fitz
import pdf2image
from PIL import Image
from contextlib import asynccontextmanager
import tempfile

from core.utils.logger import get_logger

logger = get_logger("table_extraction_utils")

@asynccontextmanager
async def temp_file_manager(content: bytes, suffix: str = '.pdf'):
    """
    Crée et gère un fichier temporaire.
    
    Args:
        content: Contenu du fichier
        suffix: Extension du fichier
        
    Yields:
        Chemin du fichier temporaire
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        temp_file.write(content)
        temp_file.close()
        yield temp_file.name
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

async def parse_page_range(file_path: str, pages: Union[str, List[int]]) -> List[int]:
    """
    Convertit une spécification de pages en liste d'indices.
    
    Args:
        file_path: Chemin du fichier PDF
        pages: Spécification des pages ("all", "1,3,5-7", ou liste)
        
    Returns:
        Liste d'indices de pages (commençant à 0)
    """
    try:
        if pages == "all":
            # Convertir tout le document
            with fitz.open(file_path) as pdf:
                return list(range(len(pdf)))
                
        elif isinstance(pages, str):
            # Format "1,3,5-7"
            page_indices = []
            for part in pages.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    page_indices.extend(range(start-1, end))
                else:
                    page_indices.append(int(part)-1)
            return page_indices
            
        else:
            # Liste directe
            return [p-1 for p in pages]  # Convertir de 1-indexed à 0-indexed
    
    except Exception as e:
        logger.error(f"Erreur parsing plage de pages: {e}")
        return []

async def convert_pdf_to_images(file_path: str, page_indices: List[int], dpi: int = 300) -> List[np.ndarray]:
    """
    Convertit des pages PDF en images.
    
    Args:
        file_path: Chemin du fichier PDF
        page_indices: Indices des pages à convertir (0-indexed)
        dpi: Résolution en DPI
        
    Returns:
        Liste d'images au format numpy array
    """
    try:
        # Conversion en indices 1-indexed pour pdf2image
        first_page = min(page_indices) + 1 if page_indices else 1
        last_page = max(page_indices) + 1 if page_indices else None
        
        # Utilisation d'un thread pour ne pas bloquer
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=4)
        
        # Conversion du PDF en images
        pil_images = await loop.run_in_executor(
            executor,
            lambda: pdf2image.convert_from_path(
                file_path,
                dpi=dpi,
                first_page=first_page,
                last_page=last_page,
                use_pdftocairo=True,  # Plus rapide et meilleure qualité
                grayscale=False  # Garder la couleur pour meilleure détection
            )
        )
        
        # Conversion des images PIL en numpy arrays
        images = [np.array(img) for img in pil_images]
        
        # Nettoyage de l'executor
        executor.shutdown(wait=False)
        
        return images
        
    except Exception as e:
        logger.error(f"Erreur conversion PDF en images: {e}")
        return []

async def estimate_pdf_complexity(file_path: str, sample_pages: int = 5) -> float:
    """
    Évalue la complexité d'un PDF pour déterminer la meilleure stratégie d'extraction.
    
    Args:
        file_path: Chemin du fichier PDF
        sample_pages: Nombre de pages à échantillonner
        
    Returns:
        Score de complexité (0-1)
    """
    try:
        # Facteurs de complexité
        factors = {
            "page_count": 0.0,          # Nombre de pages
            "font_variety": 0.0,        # Variété de polices
            "layout_complexity": 0.0,   # Complexité de mise en page
            "image_ratio": 0.0,         # Ratio d'images
            "text_density": 0.0,        # Densité de texte
            "table_indicators": 0.0     # Indicateurs de tableaux (lignes, séparateurs)
        }
        
        with fitz.open(file_path) as doc:
            # Score basé sur le nombre de pages
            page_count = len(doc)
            if page_count > 100:
                factors["page_count"] = 1.0
            elif page_count > 50:
                factors["page_count"] = 0.8
            elif page_count > 20:
                factors["page_count"] = 0.6
            elif page_count > 5:
                factors["page_count"] = 0.4
            else:
                factors["page_count"] = 0.2
                
            # Échantillonnage de pages
            sample_size = min(sample_pages, page_count)
            step = max(1, page_count // sample_size)
            sampled_pages = list(range(0, page_count, step))[:sample_size]
            
            # Analyse des pages échantillonnées
            all_fonts = set()
            total_elements = 0
            total_images = 0
            total_text = 0
            total_lines = 0
            total_rect = 0
            
            for page_idx in sampled_pages:
                page = doc[page_idx]
                
                # Variété de polices
                fonts = page.get_fonts()
                all_fonts.update([font[3] for font in fonts])
                
                # Nombre d'éléments
                elements = len(page.get_text("dict")["blocks"])
                total_elements += elements
                
                # Nombre d'images
                images = page.get_images()
                total_images += len(images)
                
                # Quantité de texte
                text = page.get_text()
                total_text += len(text)
                
                # Nombre de lignes (potentiels tableaux)
                paths = page.get_drawings()
                total_lines += len([p for p in paths if p["type"] == "l"])  # Lignes
                total_rect += len([p for p in paths if p["type"] == "re"])  # Rectangles
            
            # Calcul des facteurs
            
            # Variété de polices
            if len(all_fonts) > 10:
                factors["font_variety"] = 1.0
            elif len(all_fonts) > 5:
                factors["font_variety"] = 0.7
            elif len(all_fonts) > 2:
                factors["font_variety"] = 0.4
            else:
                factors["font_variety"] = 0.2
            
            # Complexité de mise en page
            avg_elements = total_elements / sample_size if sample_size > 0 else 0
            if avg_elements > 30:
                factors["layout_complexity"] = 1.0
            elif avg_elements > 20:
                factors["layout_complexity"] = 0.8
            elif avg_elements > 10:
                factors["layout_complexity"] = 0.5
            else:
                factors["layout_complexity"] = 0.3
            
            # Ratio d'images
            if total_images > 0 and total_text > 0:
                image_text_ratio = total_images / (total_text / 1000)  # Images par 1000 caractères
                if image_text_ratio > 0.5:
                    factors["image_ratio"] = 1.0
                elif image_text_ratio > 0.2:
                    factors["image_ratio"] = 0.7
                elif image_text_ratio > 0.1:
                    factors["image_ratio"] = 0.4
                else:
                    factors["image_ratio"] = 0.2
            
            # Densité de texte
            avg_text = total_text / sample_size if sample_size > 0 else 0
            if avg_text > 5000:
                factors["text_density"] = 1.0
            elif avg_text > 2500:
                factors["text_density"] = 0.8
            elif avg_text > 1000:
                factors["text_density"] = 0.5
            else:
                factors["text_density"] = 0.3
            
            # Indicateurs de tableaux (lignes, rectangles)
            avg_lines = total_lines / sample_size if sample_size > 0 else 0
            avg_rect = total_rect / sample_size if sample_size > 0 else 0
            
            if avg_lines > 20 or avg_rect > 10:
                factors["table_indicators"] = 1.0
            elif avg_lines > 10 or avg_rect > 5:
                factors["table_indicators"] = 0.7
            elif avg_lines > 5 or avg_rect > 2:
                factors["table_indicators"] = 0.4
            else:
                factors["table_indicators"] = 0.2
            
            # Score final pondéré
            weights = {
                "page_count": 0.05,
                "font_variety": 0.15,
                "layout_complexity": 0.25,
                "image_ratio": 0.15,
                "text_density": 0.15,
                "table_indicators": 0.25
            }
            
            final_score = sum(factors[k] * weights[k] for k in factors)
            
            return final_score
            
    except Exception as e:
        logger.error(f"Erreur estimation complexité PDF: {e}")
        return 0.5  # Valeur moyenne par défaut

async def get_extraction_cache_key(
    file_path: str,
    pages: Union[str, List[int]],
    strategy: str,
    output_format: str,
    ocr_config: Optional[Dict] = None
) -> str:
    """
    Génère une clé de cache pour l'extraction.
    
    Args:
        file_path: Chemin du fichier PDF
        pages: Pages à analyser
        strategy: Stratégie d'extraction
        output_format: Format de sortie
        ocr_config: Configuration OCR
        
    Returns:
        Clé de cache unique
    """
    try:
        # Hash du contenu du fichier
        file_hash = ""
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Chaîne de paramètres
        if isinstance(pages, list):
            pages_str = ",".join(map(str, pages))
        else:
            pages_str = str(pages)
            
        ocr_str = ""
        if ocr_config:
            ocr_items = []
            for k in sorted(ocr_config.keys()):
                ocr_items.append(f"{k}:{ocr_config[k]}")
            ocr_str = ";".join(ocr_items)
        
        # Création de la clé
        key_parts = [
            file_hash,
            pages_str,
            strategy,
            output_format,
            ocr_str
        ]
        
        key_string = ":".join(key_parts)
        return f"pdf:tables:{hashlib.md5(key_string.encode()).hexdigest()}"
        
    except Exception as e:
        logger.error(f"Erreur génération clé cache: {e}")
        # Clé de secours basée sur l'heure
        return f"pdf:tables:fallback:{int(time.time())}"