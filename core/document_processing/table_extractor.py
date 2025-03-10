# core/document_processing/table_extractor.py
from typing import List, Dict, Any, Optional, Union, Tuple, BinaryIO, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import camelot
import tabula
import pdfplumber
from io import BytesIO
import json
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2
import pytesseract
from PIL import Image, ImageOps, ImageEnhance
import pdf2image
import os
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
import contextlib
import functools
import hashlib

from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings

logger = get_logger("table_extractor")

@contextlib.asynccontextmanager
async def temp_file_manager(content: bytes, suffix: str = '.pdf'):
    """Gestionnaire de fichier temporaire asynchrone."""
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        temp_file.write(content)
        temp_file.close()
        yield temp_file.name
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@dataclass
class OCRConfig:
    """Configuration pour l'OCR."""
    lang: str = "fra+eng"
    enhance_image: bool = True
    deskew: bool = True
    preprocess_type: str = "thresh"
    psm: int = 6
    force_grid: bool = False

@dataclass
class ExtractedTable:
    """Représentation d'un tableau extrait."""
    data: Any  # DataFrame ou liste
    table_id: int
    page: int
    rows: int
    columns: int
    confidence: float = 0.0
    extraction_method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class TableExtractor:
    """Classe pour extraire des tableaux de fichiers PDF."""
    
    def __init__(self, cache_enabled: bool = True, table_detector: Optional[Any] = None):
        """
        Initialisation de l'extracteur de tableaux.
        
        Args:
            cache_enabled: Active ou désactive le cache
            table_detector: Instance du détecteur de tableaux IA (optionnel)
        """
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        self.table_detector = table_detector  # Stockage de la référence au détecteur
        
        # Configuration OCR
        self.tesseract_cmd = os.environ.get("TESSERACT_CMD", "tesseract")
        self.tesseract_lang = os.environ.get("TESSERACT_LANG", "fra+eng")
        self.dpi = int(os.environ.get("OCR_DPI", "300"))
        self.temp_dir = Path("temp/ocr")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.scanned_threshold = 0.1  # Seuil pour détecter les PDF scannés
        
        # Cache pour les résultats d'extraction (clé: hash du fichier + params, valeur: résultats)
        self.cache_enabled = cache_enabled
        self.cache = {}
        
        # Vérifier que tesseract fonctionne
        try:
            if os.path.exists(self.tesseract_cmd) and os.access(self.tesseract_cmd, os.X_OK):
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            
            # Vérifier que tesseract fonctionne
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR initialisé, version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logger.error(f"Erreur initialisation Tesseract OCR: {e}")
            logger.warning("La fonctionnalité OCR pour PDF scannés pourrait ne pas fonctionner correctement")
    
    async def extract_tables(
        self, 
        file_path: Union[str, Path, BytesIO], 
        pages: Union[str, List[int]] = "all",
        extraction_method: str = "auto",
        output_format: str = "pandas",
        ocr_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extrait les tableaux d'un fichier PDF.
        
        Cette méthode principale coordonne le processus d'extraction selon la méthode choisie.
        
        Args:
            file_path: Chemin du fichier PDF ou objet BytesIO
            pages: Pages à analyser ("all" ou liste de numéros de page)
            extraction_method: Méthode d'extraction ("auto", "tabula", "camelot", "pdfplumber", "ocr", "ai", "hybrid")
            output_format: Format de sortie ("pandas", "csv", "json", "html", "excel")
            ocr_config: Configuration spécifique pour l'OCR
            
        Returns:
            Liste de tables extraites dans le format demandé
        """
        try:
            with metrics.timer("table_extraction"):
                # Configuration OCR par défaut
                default_ocr_config = {
                    "lang": self.tesseract_lang,
                    "enhance_image": True,
                    "deskew": True,
                    "preprocess_type": "thresh",
                    "psm": 6,
                    "force_grid": False
                }
                
                # Fusionner avec la configuration fournie
                if ocr_config:
                    ocr_config = {**default_ocr_config, **ocr_config}
                else:
                    ocr_config = default_ocr_config
                
                # Gestion du fichier source
                file_source, is_temp_file = await self._prepare_file_source(file_path)
                
                try:
                    # Détecter si le PDF est scanné si méthode est auto
                    is_scanned = False
                    if extraction_method == "auto":
                        is_scanned, confidence = await self._is_scanned_pdf(file_source, pages)
                        logger.info(f"PDF scanné: {is_scanned} (confiance: {confidence:.2f})")
                        
                        extraction_method = "ocr" if is_scanned else await self._detect_best_method(file_source, pages)
                    
                    # Extraction des tableaux selon la méthode choisie
                    loop = asyncio.get_event_loop()
                    
                    if extraction_method == "ocr":
                        tables = await self._extract_with_ocr(file_source, pages, ocr_config)
                    elif extraction_method == "ai" and hasattr(self, 'table_detector'):
                        # Utiliser la détection par IA si disponible
                        tables = await self._extract_with_ai(file_source, pages, ocr_config)
                    elif extraction_method == "hybrid" and hasattr(self, 'table_detector'):
                        # Méthode hybride combinant l'IA et les méthodes traditionnelles
                        tables = await self._extract_with_hybrid(file_source, pages, ocr_config)
                    elif extraction_method == "tabula":
                        tables = await loop.run_in_executor(
                            self.executor, 
                            lambda: self._extract_with_tabula(file_source, pages)
                        )
                    elif extraction_method == "camelot":
                        tables = await loop.run_in_executor(
                            self.executor, 
                            lambda: self._extract_with_camelot(file_source, pages)
                        )
                    elif extraction_method == "pdfplumber":
                        tables = await loop.run_in_executor(
                            self.executor, 
                            lambda: self._extract_with_pdfplumber(file_source, pages)
                        )
                    else:
                        # Si méthode inconnue ou si l'IA est demandée mais non disponible
                        if extraction_method in ["ai", "hybrid"]:
                            logger.warning(f"Méthode {extraction_method} non disponible, utilisation de tabula")
                            tables = await loop.run_in_executor(
                                self.executor, 
                                lambda: self._extract_with_tabula(file_source, pages)
                            )
                        else:
                            raise ValueError(f"Méthode d'extraction inconnue: {extraction_method}")
                    
                    # Conversion vers le format de sortie désiré
                    result = await self._convert_output_format(tables, output_format)
                    return result
                        
                finally:
                    # Nettoyage du fichier temporaire si nécessaire
                    if is_temp_file and os.path.exists(file_source):
                        try:
                            os.unlink(file_source)
                        except Exception as e:
                            logger.error(f"Erreur suppression fichier temporaire: {e}")
                
        except Exception as e:
            logger.error(f"Erreur extraction tableaux: {e}", exc_info=True)
            metrics.increment_counter("table_extraction_errors")
            return []
    
    async def _prepare_file_source(self, file_path: Union[str, Path, BytesIO]) -> Tuple[str, bool]:
        """
        Prépare la source du fichier pour l'extraction.
        
        Args:
            file_path: Chemin du fichier ou BytesIO
            
        Returns:
            Tuple (chemin_fichier, est_temporaire)
        """
        if isinstance(file_path, BytesIO):
            temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            temp_file.write(file_path.getvalue())
            temp_file.close()
            return temp_file.name, True
        elif isinstance(file_path, Path):
            return str(file_path), False
        else:
            return file_path, False
    
    async def _is_scanned_pdf(self, file_path: str, pages: Union[str, List[int]]) -> Tuple[bool, float]:
        """
        Détecte si un PDF est scanné en analysant la densité de texte sélectionnable.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Pages à analyser
            
        Returns:
            Tuple (est_scanné, niveau_de_confiance)
        """
        try:
            # Initialiser les compteurs
            total_chars = 0
            total_pixels = 0
            text_blocks = 0
            
            # Convertir pages en liste
            page_list = await self._parse_page_range(file_path, pages)
            
            # Limiter le nombre de pages à analyser
            page_list = page_list[:min(5, len(page_list))]
            
            with pdfplumber.open(file_path) as pdf:
                for page_idx in page_list:
                    if page_idx >= len(pdf.pages):
                        continue
                    
                    page = pdf.pages[page_idx]
                    
                    # Compter les caractères de texte
                    text = page.extract_text() or ""
                    total_chars += len(text)
                    
                    # Compter les blocs de texte
                    words = page.extract_words()
                    text_blocks += len(words)
                    
                    # Calculer la taille de la page en pixels
                    width, height = float(page.width), float(page.height)
                    # Convertir en pixels
                    width_px = width * 72  # 72 DPI est standard pour PDF
                    height_px = height * 72
                    total_pixels += width_px * height_px
            
            # Calcul de la densité de texte
            char_density = total_chars / max(1, total_pixels) * 10000  # Caractères par 10000 pixels
            blocks_density = text_blocks / max(1, total_pixels) * 10000  # Blocs par 10000 pixels
            
            # Heuristique: une densité très faible indique un PDF scanné
            is_scanned = char_density < self.scanned_threshold or blocks_density < 0.2
            
            # Calcul du niveau de confiance
            if char_density < 0.01 or blocks_density < 0.05:  # Quasiment pas de texte
                confidence = 0.95
            elif char_density < self.scanned_threshold:
                confidence = 0.8
            elif char_density < self.scanned_threshold * 2:
                confidence = 0.5
            else:
                confidence = 0.1
            
            # Log pour le débogage
            logger.debug(f"Densité de caractères: {char_density:.4f}, " 
                        f"Densité de blocs: {blocks_density:.4f}, "
                        f"Confiance PDF scanné: {confidence:.2f}")
            
            return is_scanned, confidence
            
        except Exception as e:
            logger.error(f"Erreur détection PDF scanné: {e}")
            # En cas d'erreur, supposons que c'est un PDF normal
            return False, 0.0
    
    async def _parse_page_range(self, file_path: str, pages: Union[str, List[int]]) -> List[int]:
        """
        Convertit une spécification de pages en liste d'indices.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Spécification des pages ("all", "1,3,5-7", ou liste)
            
        Returns:
            Liste d'indices de pages (commençant à 0)
        """
        if pages == "all":
            # Convertir tout le document
            with pdfplumber.open(file_path) as pdf:
                return list(range(len(pdf.pages)))
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
    
    async def _extract_with_ocr(
        self, 
        file_path: str, 
        pages: Union[str, List[int]], 
        ocr_config: Dict[str, Any]
    ) -> List[pd.DataFrame]:
        """
        Extrait les tableaux à partir d'un PDF scanné en utilisant l'OCR.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Pages à analyser
            ocr_config: Configuration OCR
            
        Returns:
            Liste de DataFrames pandas contenant les tableaux
        """
        try:
            logger.info(f"Extraction avec OCR: {file_path}")
            
            # Convertir pages en liste
            page_list = await self._parse_page_range(file_path, pages)
            
            # Convertir le PDF en images
            images = await self._convert_pdf_to_images(file_path, page_list)
            
            # Créer un dictionnaire pour associer l'index de l'image à la page
            image_to_page = {}
            if isinstance(pages, str) and pages == "all":
                for i in range(len(images)):
                    image_to_page[i] = i + 1
            else:
                for i, page_idx in enumerate(page_list):
                    if i < len(images):
                        image_to_page[i] = page_idx + 1  # +1 car page_idx est 0-based
            
            # Liste pour stocker les DataFrames
            tables_list = []
            
            # Configurer l'OCR
            ocr_params = OCRConfig(
                lang=ocr_config.get("lang", self.tesseract_lang),
                enhance_image=ocr_config.get("enhance_image", True),
                deskew=ocr_config.get("deskew", True),
                preprocess_type=ocr_config.get("preprocess_type", "thresh"),
                psm=ocr_config.get("psm", 6),
                force_grid=ocr_config.get("force_grid", False)
            )
            
            # Traiter chaque image en parallèle
            tasks = []
            for img_idx, img in enumerate(images):
                page_num = image_to_page.get(img_idx, img_idx + 1)
                tasks.append(self._process_image_for_tables(img, page_num, ocr_params))
            
            # Exécuter les tâches de traitement d'image
            results = await asyncio.gather(*tasks)
            
            # Rassembler les résultats
            for page_tables in results:
                if page_tables:
                    tables_list.extend(page_tables)
            
            logger.info(f"OCR terminé: {len(tables_list)} tableaux extraits")
            return tables_list
            
        except Exception as e:
            logger.error(f"Erreur extraction OCR: {e}", exc_info=True)
            return []
    
    async def _convert_pdf_to_images(self, file_path: str, page_list: List[int]) -> List[np.ndarray]:
        """
        Convertit les pages d'un PDF en images.
        
        Args:
            file_path: Chemin du fichier PDF
            page_list: Liste des indices de pages (0-indexed)
            
        Returns:
            Liste d'images au format numpy array
        """
        loop = asyncio.get_event_loop()
        
        # Ajuster pour pdf2image qui utilise des indices 1-indexed
        first_page = min(page_list) + 1 if page_list else 1
        last_page = max(page_list) + 1 if page_list else None
        
        # Convertir le PDF en images
        pil_images = await loop.run_in_executor(
            self.executor,
            lambda: pdf2image.convert_from_path(
                file_path, 
                dpi=self.dpi, 
                first_page=first_page,
                last_page=last_page
            )
        )
        
        # Convertir les images PIL en numpy arrays
        images = [np.array(img) for img in pil_images]
        
        return images
    
    async def _process_image_for_tables(self, img: np.ndarray, page_num: int, ocr_params: OCRConfig) -> List[pd.DataFrame]:
        """
        Traite une image pour détecter et extraire les tableaux.
        
        Args:
            img: Image au format numpy array
            page_num: Numéro de la page
            ocr_params: Paramètres OCR
            
        Returns:
            Liste de DataFrames pandas contenant les tableaux
        """
        try:
            # Prétraitement de l'image
            processed_img = await self._preprocess_image(
                img, 
                ocr_params.enhance_image,
                ocr_params.deskew,
                ocr_params.preprocess_type
            )
            
            # Détection des tableaux
            table_regions = await self._detect_table_regions(processed_img)
            
            logger.debug(f"Page {page_num}: {len(table_regions)} tableaux détectés")
            
            # Pour chaque région de tableau détectée
            tables = []
            for i, region in enumerate(table_regions):
                try:
                    # Extraire la région d'intérêt
                    x, y, w, h = region
                    table_img = processed_img[y:y+h, x:x+w]
                    
                    # Apporter des améliorations supplémentaires à l'image du tableau
                    table_img = await self._enhance_table_image(table_img)
                    
                    # Extraire le texte du tableau avec OCR
                    df = await self._ocr_table_to_dataframe(
                        table_img,
                        lang=ocr_params.lang,
                        psm=ocr_params.psm,
                        force_grid=ocr_params.force_grid
                    )
                    
                    # Ajouter à la liste si le DataFrame n'est pas vide
                    if df is not None and not df.empty and len(df.columns) > 1:
                        # Ajouter des colonnes d'information
                        df = df.copy()  # Créer une copie pour éviter les avertissements
                        df.insert(0, "_page", page_num)
                        df.insert(1, "_table", i + 1)
                        tables.append(df)
                        
                except Exception as e:
                    logger.error(f"Erreur extraction tableau {i+1} sur page {page_num}: {e}")
            
            return tables
            
        except Exception as e:
            logger.error(f"Erreur traitement image page {page_num}: {e}")
            return []
    
    async def _preprocess_image(
        self, 
        image: np.ndarray, 
        enhance: bool = True, 
        deskew: bool = True, 
        preprocess_type: str = "thresh"
    ) -> np.ndarray:
        """
        Prétraite une image pour améliorer la détection et l'OCR des tableaux.
        
        Args:
            image: Image à prétraiter (numpy array)
            enhance: Si True, améliore le contraste de l'image
            deskew: Si True, redresse l'image
            preprocess_type: Type de prétraitement ('thresh', 'adaptive', 'blur')
            
        Returns:
            Image prétraitée
        """
        try:
            # Convertir en PIL Image pour certains traitements
            pil_image = Image.fromarray(image)
            
            # Redressement si demandé
            if deskew:
                loop = asyncio.get_event_loop()
                pil_image = await loop.run_in_executor(
                    self.executor,
                    self._deskew_image,
                    pil_image
                )
            
            # Amélioration du contraste si demandée
            if enhance:
                # Augmenter le contraste
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.5)
                
                # Augmenter la netteté
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.5)
            
            # Reconvertir en numpy array
            image = np.array(pil_image)
            
            # Convertir en niveaux de gris si l'image est en couleur
            if len(image.shape) > 2 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Appliquer le prétraitement selon le type
            if preprocess_type == "thresh":
                # Seuillage binaire global
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return binary
            elif preprocess_type == "adaptive":
                # Seuillage adaptatif
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
                return binary
            elif preprocess_type == "blur":
                # Flou gaussien pour réduire le bruit
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return binary
            else:
                # Par défaut, retourner l'image en niveaux de gris
                return gray
                
        except Exception as e:
            logger.error(f"Erreur prétraitement image: {e}")
            # En cas d'erreur, retourner l'image originale
            return image
    
    def _deskew_image(self, pil_image: Image.Image) -> Image.Image:
        """
        Redresse une image inclinée.
        
        Args:
            pil_image: Image PIL à redresser
            
        Returns:
            Image redressée
        """
        try:
            # Convertir en OpenCV
            image = np.array(pil_image)
            
            # Convertir en niveaux de gris
            if len(image.shape) > 2 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Détecter les bords
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Détecter les lignes
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            if lines is None:
                return pil_image
            
            # Calculer l'angle moyen
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:  # Éviter division par zéro
                    continue
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                # Ne considérer que les angles proches de l'horizontale ou verticale
                if abs(angle) < 45 or abs(angle - 90) < 45 or abs(angle + 90) < 45:
                    angles.append(angle)
            
            if not angles:
                return pil_image
            
            # Angle moyen
            median_angle = np.median(angles)
            
            # Calculer l'angle de correction
            if abs(median_angle) < 45:  # Proche de l'horizontale
                skew_angle = median_angle
            elif abs(median_angle - 90) < 45:  # Proche de la verticale (vers le haut)
                skew_angle = median_angle - 90
            elif abs(median_angle + 90) < 45:  # Proche de la verticale (vers le bas)
                skew_angle = median_angle + 90
            else:
                skew_angle = 0
            
            # Ne corriger que si l'angle est significatif
            if abs(skew_angle) > 0.5:
                # Rotation de l'image avec PIL
                return pil_image.rotate(skew_angle, resample=Image.BICUBIC, expand=True)
            
            return pil_image
            
        except Exception as e:
            logger.error(f"Erreur redressement image: {e}")
            return pil_image
    
    async def _detect_table_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Détecte les régions contenant des tableaux dans une image.
        
        Args:
            image: Image prétraitée (numpy array)
            
        Returns:
            Liste de tuples (x, y, largeur, hauteur) délimitant les tableaux
        """
        try:
            # Si la détection par IA est disponible et configurée, l'utiliser
            if hasattr(self, 'table_detector') and self.table_detector:
                detections = await self.table_detector.detect_tables(image)
                if detections:
                    return [(d["x"], d["y"], d["width"], d["height"]) for d in detections]
            
            # Sinon, utiliser une méthode basée sur la détection de lignes
            # Convertir en niveaux de gris si nécessaire
            if len(image.shape) > 2 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Appliquer un filtre pour détecter les lignes horizontales et verticales
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            # Détecter les lignes horizontales
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # Détecter les lignes verticales
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combiner les lignes
            table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
            
            # Dilater pour connecter les lignes proches
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            table_mask = cv2.dilate(table_mask, kernel, iterations=4)
            
            # Trouver les contours des zones de tableau
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrer les petits contours
            min_area = 0.01 * image.shape[0] * image.shape[1]  # Au moins 1% de l'image
            table_regions = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > min_area:
                    # Vérifier les proportions (pas trop allongé)
                    aspect_ratio = w / h
                    if 0.2 < aspect_ratio < 5:  # Filtrer les formes trop allongées
                        table_regions.append((x, y, w, h))
            
            # Si aucune région détectée avec cette méthode, utiliser toute l'image
            if not table_regions:
                height, width = image.shape[:2]
                table_regions = [(0, 0, width, height)]
            
            return table_regions
            
        except Exception as e:
            logger.error(f"Erreur détection régions de tableau: {e}")
            # En cas d'erreur, considérer toute l'image comme un tableau
            height, width = image.shape[:2]
            return [(0, 0, width, height)]
    
    async def _enhance_table_image(self, image: np.ndarray) -> np.ndarray:
        """
        Améliore une image de tableau pour optimiser l'OCR.
        
        Args:
            image: Image du tableau (numpy array)
            
        Returns:
            Image améliorée
        """
        try:
            # Convertir en PIL Image pour certains traitements
            pil_image = Image.fromarray(image)
            
            # Augmenter le contraste
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(2.0)
            
            # Augmenter la netteté
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(2.0)
            
            # Reconvertir en numpy array
            enhanced_image = np.array(pil_image)
            
            # Convertir en niveaux de gris si l'image est en couleur
            if len(enhanced_image.shape) > 2 and enhanced_image.shape[2] == 3:
                gray = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = enhanced_image
            
            # Supprimer le bruit
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # Améliorer le contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Binarisation avec Otsu
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return binary
            
        except Exception as e:
            logger.error(f"Erreur amélioration image de tableau: {e}")
            # En cas d'erreur, retourner l'image originale
            return image
    
    async def _ocr_table_to_dataframe(
        self, 
        table_image: np.ndarray, 
        lang: str = "fra+eng", 
        psm: int = 6,
        force_grid: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Convertit une image de tableau en DataFrame avec OCR.
        
        Args:
            table_image: Image du tableau (numpy array)
            lang: Langues pour Tesseract OCR
            psm: Mode de segmentation de page pour Tesseract
            force_grid: Si True, force une approche grille
            
        Returns:
            DataFrame pandas ou None en cas d'échec
        """
        try:
            # Si force_grid est activé, utiliser l'approche grille
            if force_grid:
                return await self._ocr_table_grid_approach(table_image, lang, psm)
            
            # Sinon, utiliser l'OCR pour récupérer tout le texte
            loop = asyncio.get_event_loop()
            
            # Configurer les options OCR
            custom_config = f'--oem 3 --psm {psm} -l {lang}'
            
            # Exécuter l'OCR avec Tesseract
            text = await loop.run_in_executor(
                self.executor,
                lambda: pytesseract.image_to_string(table_image, config=custom_config)
            )
            
            if not text.strip():
                logger.warning("Aucun texte détecté par OCR")
                return None
            
            # Analyser le texte pour en faire un tableau
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Détecter le séparateur (espace, tabulation, etc.)
            separator = self._detect_separator(lines)
            
            # Parser les lignes en fonction du séparateur
            rows = []
            for line in lines:
                if separator == 'whitespace':
                    # Diviser par espaces multiples
                    row = [col.strip() for col in line.split() if col.strip()]
                elif separator in line:
                    # Diviser par le séparateur détecté
                    row = [col.strip() for col in line.split(separator) if col.strip()]
                else:
                    # Si le séparateur n'est pas présent, essayer de deviner les colonnes
                    row = self._guess_columns(line)
                
                if row:  # N'ajouter que les lignes non vides
                    rows.append(row)
            
            if not rows:
                logger.warning("Aucune donnée valide détectée après analyse du texte")
                return None
            
            # Normaliser le nombre de colonnes (utiliser le mode)
            col_counts = [len(row) for row in rows]
            if not col_counts:
                return None
                
            # Utiliser le nombre de colonnes le plus fréquent
            from collections import Counter
            most_common_count = Counter(col_counts).most_common(1)[0][0]
            
            # Filtrer les lignes ayant le bon nombre de colonnes
            valid_rows = [row for row in rows if len(row) == most_common_count]
            
            if not valid_rows:
                # Tentative de récupération: ajuster les lignes pour qu'elles aient toutes le même nombre de colonnes
                normalized_rows = []
                for row in rows:
                    if len(row) < most_common_count:
                        # Ajouter des cellules vides
                        row = row + [''] * (most_common_count - len(row))
                    elif len(row) > most_common_count:
                        # Tronquer
                        row = row[:most_common_count]
                    normalized_rows.append(row)
                valid_rows = normalized_rows
            
            # Créer le DataFrame
            if valid_rows:
                # Première ligne comme en-tête si elle semble appropriée
                first_row = valid_rows[0]
                other_rows = valid_rows[1:]
                
                # Vérifier si la première ligne est un en-tête valide
                is_header = self._is_valid_header(first_row, other_rows)
                
                if is_header and len(other_rows) > 0:
                    df = pd.DataFrame(other_rows, columns=first_row)
                else:
                    # Utiliser des noms de colonnes génériques
                    columns = [f'Column{i+1}' for i in range(most_common_count)]
                    df = pd.DataFrame(valid_rows, columns=columns)
                
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur OCR tableau vers DataFrame: {e}")
            return None
    
    def _detect_separator(self, lines: List[str]) -> str:
        """
        Détecte le séparateur le plus probable dans les lignes de texte.
        
        Args:
            lines: Liste de lignes de texte
            
        Returns:
            Séparateur détecté ('|', ',', ';', 'tab', 'whitespace')
        """
        separators = {'|': 0, ',': 0, ';': 0, '\t': 0}
        
        for line in lines:
            for sep, count in separators.items():
                separators[sep] += line.count(sep)
        
        # Trouver le séparateur le plus fréquent
        max_sep = max(separators.items(), key=lambda x: x[1])
        
        # Si un séparateur est clairement identifié
        if max_sep[1] > 0:
            return max_sep[0]
        
        # Sinon, supposer que c'est un séparateur par espaces
        return 'whitespace'
    
    def _guess_columns(self, line: str) -> List[str]:
        """
        Devine les colonnes dans une ligne sans séparateur clair.
        
        Args:
            line: Ligne de texte
            
        Returns:
            Liste de valeurs de colonnes
        """
        # Recherche de motifs d'espacement réguliers
        parts = []
        current_part = ""
        space_count = 0
        
        for char in line:
            if char == ' ':
                space_count += 1
                if space_count >= 3:  # Considérer 3 espaces ou plus comme un séparateur
                    if current_part:
                        parts.append(current_part)
                        current_part = ""
                    space_count = 0
                else:
                    current_part += char
            else:
                space_count = 0
                current_part += char
        
        if current_part:
            parts.append(current_part)
        
        # Si aucune colonne n'a été identifiée, diviser par espaces simples
        if len(parts) <= 1:
            parts = line.split()
        
        return [p.strip() for p in parts]
    
    def _is_valid_header(self, header_row: List[str], data_rows: List[List[str]]) -> bool:
        """
        Vérifie si une ligne peut être considérée comme un en-tête valide.
        
        Args:
            header_row: Ligne d'en-tête potentielle
            data_rows: Autres lignes de données
            
        Returns:
            True si la ligne est un en-tête valide, False sinon
        """
        if not data_rows:
            return False
            
        # Vérifier si l'en-tête contient des chaînes numériques
        header_numeric = sum(1 for cell in header_row if cell.replace('.', '').isdigit())
        
        # Échantillon des lignes de données pour vérifier si elles sont numériques
        sample_size = min(5, len(data_rows))
        sample_rows = data_rows[:sample_size]
        
        numeric_counts = []
        for row in sample_rows:
            numeric_count = sum(1 for cell in row if cell.replace('.', '').isdigit())
            numeric_counts.append(numeric_count)
        
        # Calculer le pourcentage moyen de cellules numériques dans les données
        avg_numeric_percent = sum(numeric_counts) / (sample_size * len(header_row)) if sample_size > 0 else 0
        
        # Si l'en-tête contient significativement moins de numériques que les données, c'est probablement un en-tête
        header_numeric_percent = header_numeric / len(header_row)
        
        return header_numeric_percent < avg_numeric_percent * 0.5
    
    async def _ocr_table_grid_approach(
        self, 
        table_image: np.ndarray, 
        lang: str = "fra+eng", 
        psm: int = 6
    ) -> Optional[pd.DataFrame]:
        """
        Utilise une approche grille pour extraire un tableau avec OCR.
        
        Args:
            table_image: Image du tableau (numpy array)
            lang: Langues pour Tesseract OCR
            psm: Mode de segmentation de page pour Tesseract
            
        Returns:
            DataFrame pandas ou None en cas d'échec
        """
        try:
            # Convertir en niveaux de gris si nécessaire
            if len(table_image.shape) > 2 and table_image.shape[2] == 3:
                gray = cv2.cvtColor(table_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = table_image
            
            # Détection des lignes
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Détecter les lignes horizontales et verticales
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            # Détecter les lignes horizontales
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # Détecter les lignes verticales
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combiner les lignes
            table_grid = cv2.bitwise_or(horizontal_lines, vertical_lines)
            
            # Dilater pour améliorer la détection
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            table_grid = cv2.dilate(table_grid, kernel, iterations=2)
            
            # Inverser l'image pour que les lignes soient noires
            table_grid = 255 - table_grid
            
            # Trouver les contours des cellules
            contours, _ = cv2.findContours(255 - table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Trier les contours par position (y puis x)
            bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
            
            # Filtrer les contours trop petits
            min_area = 0.0001 * table_image.shape[0] * table_image.shape[1]
            bounding_boxes = [(x, y, w, h) for x, y, w, h in bounding_boxes if w*h > min_area]
            
            if not bounding_boxes:
                logger.warning("Aucune cellule détectée dans la grille")
                return None
            
            # Estimer le nombre de lignes et colonnes
            y_positions = sorted(set(y for _, y, _, _ in bounding_boxes))
            x_positions = sorted(set(x for x, _, _, _ in bounding_boxes))
            
            # Tolérance pour considérer des positions proches comme identiques
            tolerance_y = table_image.shape[0] // 30
            tolerance_x = table_image.shape[1] // 30
            
            # Regrouper les positions proches
            y_groups = self._cluster_positions(y_positions, tolerance_y)
            x_groups = self._cluster_positions(x_positions, tolerance_x)
            
            num_rows = len(y_groups)
            num_cols = len(x_groups)
            
            if num_rows <= 1 or num_cols <= 1:
                logger.warning(f"Détection de grille insuffisante: {num_rows} lignes, {num_cols} colonnes")
                return None
            
            # Initialiser un tableau vide
            table_data = [['' for _ in range(num_cols)] for _ in range(num_rows)]
            
            # Configurer les options OCR
            custom_config = f'--oem 3 --psm {psm} -l {lang}'
            
            # Pour chaque cellule détectée
            loop = asyncio.get_event_loop()
            for x, y, w, h in bounding_boxes:
                # Trouver la ligne et la colonne correspondantes
                row_idx = next((i for i, group in enumerate(y_groups) if min(group) <= y <= max(group)), None)
                col_idx = next((i for i, group in enumerate(x_groups) if min(group) <= x <= max(group)), None)
                
                if row_idx is not None and col_idx is not None:
                    # Extraire la cellule
                    cell_img = table_image[y:y+h, x:x+w]
                    
                    # Appliquer l'OCR sur cette cellule
                    cell_text = await loop.run_in_executor(
                        self.executor,
                        lambda: pytesseract.image_to_string(cell_img, config=custom_config).strip()
                    )
                    
                    # Stocker le texte dans le tableau
                    table_data[row_idx][col_idx] = cell_text
            
            # Convertir en DataFrame
            if table_data:
                # Utiliser la première ligne comme en-tête
                header = table_data[0]
                data = table_data[1:]
                
                # Vérifier si l'en-tête est valide
                if self._is_valid_header(header, data):
                    df = pd.DataFrame(data, columns=header)
                else:
                    # Utiliser des en-têtes génériques
                    columns = [f'Column{i+1}' for i in range(num_cols)]
                    df = pd.DataFrame(table_data, columns=columns)
                
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur approche grille OCR: {e}")
            return None
    
    def _cluster_positions(self, positions: List[int], tolerance: int) -> List[List[int]]:
        """
        Regroupe des positions proches en clusters.
        
        Args:
            positions: Liste des positions (triées)
            tolerance: Tolérance pour considérer des positions comme appartenant au même groupe
            
        Returns:
            Liste de groupes de positions
        """
        if not positions:
            return []
            
        groups = []
        current_group = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current_group[-1] <= tolerance:
                current_group.append(pos)
            else:
                groups.append(current_group)
                current_group = [pos]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    async def _detect_best_method(self, file_path: str, pages: Union[str, List[int]]) -> str:
        """
        Détermine la meilleure méthode d'extraction.
        """
        try:
            # Heuristique simple: essayer d'extraire avec chaque méthode et voir laquelle donne les meilleurs résultats
            loop = asyncio.get_event_loop()
            
            # Test avec Camelot (limité à quelques pages pour plus d'efficacité)
            test_pages = "1-3" if isinstance(pages, str) and pages == "all" else pages
            
            camelot_result = await loop.run_in_executor(
                self.executor,
                lambda: self._test_camelot(file_path, test_pages)
            )
            
            # Test avec Tabula (généralement meilleur pour tableaux simples sans bordures)
            tabula_result = await loop.run_in_executor(
                self.executor,
                lambda: self._test_tabula(file_path, test_pages)
            )
            
            # Test rapide avec pdfplumber
            pdfplumber_score = await loop.run_in_executor(
                self.executor,
                lambda: self._test_pdfplumber(file_path, test_pages)
            )
            
            # Choix basé sur le nombre et la qualité des tableaux détectés
            scores = {
                "camelot": camelot_result.get("score", 0),
                "tabula": tabula_result.get("score", 0),
                "pdfplumber": pdfplumber_score.get("score", 0)
            }
            
            logger.debug(f"Scores des méthodes: {scores}")
            
            # Sélectionner la méthode avec le score le plus élevé
            best_method, best_score = max(scores.items(), key=lambda x: x[1])
            
            # Si le meilleur score est trop faible, envisager des alternatives
            if best_score < 0.5:
                if hasattr(self, 'table_detector'):
                    logger.info(f"Scores faibles pour toutes les méthodes traditionnelles, utilisation d'IA")
                    return "ai"
                else:
                    logger.info(f"Scores faibles pour toutes les méthodes traditionnelles, utilisation hybride")
                    return "hybrid"
            
            logger.info(f"Meilleure méthode détectée: {best_method} (score: {best_score:.2f})")
            return best_method
            
        except Exception as e:
            logger.warning(f"Erreur détection méthode, utilisation de tabula par défaut: {e}")
            return "tabula"
    
    def _test_camelot(self, file_path: str, pages: Union[str, List[int]]) -> Dict[str, Any]:
        """Test amélioré avec Camelot."""
        try:
            if isinstance(pages, list):
                # Convertir la liste en chaîne pour camelot
                page_str = ",".join(map(str, [p+1 for p in pages]))  # +1 car camelot utilise 1-indexed
            else:
                page_str = pages
            
            # Limiter à quelques pages pour le test
            if page_str == "all":
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    max_pages = min(len(pdf.pages), 5)
                page_str = f"1-{max_pages}"
            
            # Essayer d'abord avec lattice (pour les tableaux avec bordures)
            try:
                tables_lattice = camelot.read_pdf(
                    file_path, 
                    pages=page_str,
                    flavor="lattice",
                    suppress_stdout=True
                )
                
                # Calculer le score pour les tables lattice
                if len(tables_lattice) > 0:
                    avg_accuracy = sum(table.parsing_report['accuracy'] for table in tables_lattice) / len(tables_lattice)
                    lattice_score = avg_accuracy * len(tables_lattice)
                else:
                    lattice_score = 0
            except Exception:
                lattice_score = 0
                tables_lattice = []
            
            # Essayer avec stream (pour les tableaux sans bordures)
            try:
                tables_stream = camelot.read_pdf(
                    file_path, 
                    pages=page_str,
                    flavor="stream",
                    suppress_stdout=True
                )
                
                # Calculer le score pour les tables stream
                if len(tables_stream) > 0:
                    avg_accuracy = sum(table.parsing_report['accuracy'] for table in tables_stream) / len(tables_stream)
                    stream_score = avg_accuracy * len(tables_stream)
                else:
                    stream_score = 0
            except Exception:
                stream_score = 0
                tables_stream = []
            
            # Score global et nombre de tables
            total_tables = len(tables_lattice) + len(tables_stream)
            score = max(lattice_score, stream_score)
            
            return {
                "score": score,
                "tables": total_tables,
                "lattice_score": lattice_score,
                "stream_score": stream_score
            }
            
        except Exception as e:
            logger.debug(f"Test camelot échoué: {e}")
            return {"score": 0, "tables": 0}
    
    def _test_tabula(self, file_path: str, pages: Union[str, List[int]]) -> Dict[str, Any]:
        """Test amélioré avec Tabula."""
        try:
            # Convertir pages pour tabula
            if isinstance(pages, list):
                # Convertir la liste en chaîne pour tabula
                page_list = [p+1 for p in pages]  # +1 car tabula utilise 1-indexed
            elif pages == "all":
                # Limiter à quelques pages pour le test
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    max_pages = min(len(pdf.pages), 5)
                page_list = list(range(1, max_pages + 1))
            else:
                # Convertir la chaîne en liste
                parts = []
                for part in pages.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        parts.extend(range(start, end + 1))
                    else:
                        parts.append(int(part))
                page_list = parts
            
            # Extraire les tableaux avec tabula
            tables = tabula.read_pdf(
                file_path,
                pages=page_list,
                multiple_tables=True,
                guess=True
            )
            
            # Calcul du score basé sur le nombre de tables et leur complexité
            if len(tables) == 0:
                return {"score": 0, "tables": 0}
            
            # Évaluation heuristique de la qualité
            complexity_score = 0
            for table in tables:
                if table.empty:
                    continue
                    
                # Évaluer la complexité du tableau
                complexity = min(1.0, (len(table) * len(table.columns)) / 100)
                
                # Évaluer la qualité des données
                non_null_ratio = 1.0 - (table.isna().sum().sum() / (len(table) * len(table.columns)))
                
                # Score pour ce tableau
                table_score = complexity * non_null_ratio
                complexity_score += table_score
            
            # Score final
            score = complexity_score * len(tables)
            
            return {
                "score": score,
                "tables": len(tables),
                "complexity_score": complexity_score
            }
            
        except Exception as e:
            logger.debug(f"Test tabula échoué: {e}")
            return {"score": 0, "tables": 0}
    
    def _test_pdfplumber(self, file_path: str, pages: Union[str, List[int]]) -> Dict[str, Any]:
        """Test amélioré avec pdfplumber."""
        try:
            # Convertir pages pour pdfplumber
            if isinstance(pages, list):
                page_indices = pages  # pdfplumber utilise 0-indexed
            elif pages == "all":
                # Limiter à quelques pages pour le test
                with pdfplumber.open(file_path) as pdf:
                    page_indices = list(range(min(5, len(pdf.pages))))
            else:
                # Convertir la chaîne en liste
                parts = []
                for part in pages.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        parts.extend(range(start-1, end))  # -1 car pdfplumber utilise 0-indexed
                    else:
                        parts.append(int(part)-1)  # -1 car pdfplumber utilise 0-indexed
                page_indices = parts
            
            # Extrait les tableaux avec pdfplumber
            tables_count = 0
            complexity_score = 0
            
            with pdfplumber.open(file_path) as pdf:
                for page_idx in page_indices:
                    if page_idx >= len(pdf.pages):
                        continue
                        
                    page = pdf.pages[page_idx]
                    tables = page.extract_tables()
                    
                    for table in tables:
                        if not table or len(table) == 0 or len(table[0]) == 0:
                            continue
                            
                        tables_count += 1
                        
                        # Évaluer la complexité du tableau
                        rows = len(table)
                        cols = len(table[0])
                        complexity = min(1.0, (rows * cols) / 100)
                        
                        # Évaluer la qualité des données
                        non_empty_count = sum(1 for row in table for cell in row if cell and cell.strip())
                        total_cells = rows * cols
                        non_empty_ratio = non_empty_count / total_cells if total_cells > 0 else 0
                        
                        # Score pour ce tableau
                        table_score = complexity * non_empty_ratio
                        complexity_score += table_score
            
            # Score final
            score = complexity_score * tables_count
            
            return {
                "score": score,
                "tables": tables_count,
                "complexity_score": complexity_score
            }
            
        except Exception as e:
            logger.debug(f"Test pdfplumber échoué: {e}")
            return {"score": 0, "tables": 0}
    
    def _extract_with_tabula(self, file_path: str, pages: Union[str, List[int]]) -> List[pd.DataFrame]:
        """
        Extrait les tableaux avec Tabula.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Pages à analyser
            
        Returns:
            Liste de DataFrames pandas
        """
        try:
            # Convertir les pages au format attendu par tabula
            if isinstance(pages, list):
                page_list = [p+1 for p in pages]  # +1 car tabula utilise 1-indexed
            elif pages == "all":
                page_list = "all"
            else:
                # Convertir la chaîne en liste
                parts = []
                for part in pages.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        parts.extend(range(start, end + 1))
                    else:
                        parts.append(int(part))
                page_list = parts
            
            # Extraire les tableaux avec tabula
            tables = tabula.read_pdf(
                file_path,
                pages=page_list,
                multiple_tables=True,
                guess=True
            )
            
            # Nettoyer les tableaux extraits
            cleaned_tables = []
            for table in tables:
                if table is not None and not table.empty:
                    # Supprimer les colonnes vides
                    if table.shape[1] > 1:  # Vérifier qu'il y a plus d'une colonne
                        table = table.dropna(axis=1, how='all')
                    
                    # Supprimer les lignes vides
                    table = table.dropna(how='all')
                    
                    if not table.empty:
                        cleaned_tables.append(table)
            
            return cleaned_tables
            
        except Exception as e:
            logger.error(f"Erreur extraction avec tabula: {e}")
            return []
    
    def _extract_with_camelot(self, file_path: str, pages: Union[str, List[int]]) -> List[pd.DataFrame]:
        """
        Extrait les tableaux avec Camelot.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Pages à analyser
            
        Returns:
            Liste de DataFrames pandas
        """
        try:
            # Convertir pages au format attendu par camelot
            if isinstance(pages, list):
                page_str = ",".join([str(p+1) for p in pages])  # +1 car camelot utilise 1-indexed
            else:
                page_str = pages
            
            # Essayer d'abord avec lattice (pour les tableaux avec bordures)
            tables_lattice = camelot.read_pdf(
                file_path, 
                pages=page_str,
                flavor="lattice",
                suppress_stdout=True
            )
            
            # Ensuite avec stream (pour les tableaux sans bordures)
            tables_stream = camelot.read_pdf(
                file_path, 
                pages=page_str,
                flavor="stream",
                suppress_stdout=True
            )
            
            # Combiner les résultats
            tables = []
            
            # Ajouter les tables lattice
            for table in tables_lattice:
                df = table.df
                if not df.empty:
                    tables.append(df)
            
            # Ajouter les tables stream
            for table in tables_stream:
                df = table.df
                if not df.empty:
                    # Vérifier si cette table n'est pas un doublon d'une table lattice
                    is_duplicate = False
                    for existing_table in tables:
                        if df.shape == existing_table.shape:
                            # Si même taille, vérifier le contenu
                            similarity = self._calculate_dataframe_similarity(df, existing_table)
                            if similarity > 0.7:  # Seuil arbitraire de similarité
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        tables.append(df)
            
            return tables
            
        except Exception as e:
            logger.error(f"Erreur extraction avec camelot: {e}")
            return []
    
    def _calculate_dataframe_similarity(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """
        Calcule un score de similarité entre deux DataFrames.
        
        Args:
            df1: Premier DataFrame
            df2: Second DataFrame
            
        Returns:
            Score de similarité (0-1)
        """
        # Si les dimensions sont différentes, les DataFrames sont différents
        if df1.shape != df2.shape:
            return 0.0
        
        # Convertir en chaînes pour comparaison
        df1_str = df1.astype(str)
        df2_str = df2.astype(str)
        
        # Calculer le nombre de cellules identiques
        identical_cells = (df1_str == df2_str).sum().sum()
        total_cells = df1.size
        
        # Similarité = proportion de cellules identiques
        return identical_cells / total_cells if total_cells > 0 else 0.0
    
    def _extract_with_pdfplumber(self, file_path: str, pages: Union[str, List[int]]) -> List[pd.DataFrame]:
        """
        Extrait les tableaux avec pdfplumber.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Pages à analyser
            
        Returns:
            Liste de DataFrames pandas
        """
        try:
            # Convertir pages au format attendu par pdfplumber
            if isinstance(pages, list):
                page_indices = pages  # pdfplumber utilise 0-indexed
            elif pages == "all":
                # Toutes les pages
                with pdfplumber.open(file_path) as pdf:
                    page_indices = list(range(len(pdf.pages)))
            else:
                # Convertir la chaîne en liste
                parts = []
                for part in pages.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        parts.extend(range(start-1, end))  # -1 car pdfplumber utilise 0-indexed
                    else:
                        parts.append(int(part)-1)  # -1 car pdfplumber utilise 0-indexed
                page_indices = parts
            
            # Extraire les tableaux
            tables = []
            
            with pdfplumber.open(file_path) as pdf:
                for page_idx in page_indices:
                    if page_idx >= len(pdf.pages):
                        continue
                        
                    page = pdf.pages[page_idx]
                    extracted_tables = page.extract_tables()
                    
                    for table in extracted_tables:
                        if not table or len(table) == 0 or len(table[0]) == 0:
                            continue
                            
                        # Convertir en DataFrame pandas
                        headers = table[0]
                        data = table[1:]
                        
                        # Vérifier si les en-têtes sont valides
                        if all(header for header in headers):
                            df = pd.DataFrame(data, columns=headers)
                        else:
                            # Utiliser des noms de colonnes génériques
                            columns = [f'Column{i+1}' for i in range(len(table[0]))]
                            df = pd.DataFrame(table, columns=columns)
                        
                        # Nettoyer le DataFrame
                        df = df.dropna(how='all').dropna(axis=1, how='all')
                        
                        if not df.empty:
                            tables.append(df)
            
            return tables
            
        except Exception as e:
            logger.error(f"Erreur extraction avec pdfplumber: {e}")
            return []
    
    async def _convert_output_format(self, tables: List[pd.DataFrame], output_format: str) -> List[Dict[str, Any]]:
        """
        Convertit les tables extraites dans le format de sortie demandé.
        
        Args:
            tables: Liste de DataFrames pandas
            output_format: Format de sortie ("pandas", "csv", "json", "html", "excel")
            
        Returns:
            Liste de dictionnaires contenant les tables formatées
        """
        result = []
        
        try:
            for i, df in enumerate(tables):
                if df is None or df.empty:
                    continue
                
                table_dict = {
                    "table_id": i + 1,
                    "page": df.get("_page", i + 1) if "_page" in df.columns else i + 1,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "extraction_method": "auto",
                    "confidence": 0.8  # Valeur arbitraire
                }
                
                # Retirer les colonnes de métadonnées si elles existent
                if "_page" in df.columns:
                    df = df.drop("_page", axis=1)
                if "_table" in df.columns:
                    df = df.drop("_table", axis=1)
                
                # Formater selon le format demandé
                if output_format == "pandas":
                    table_dict["data"] = df
                elif output_format == "csv":
                    table_dict["data"] = df.to_csv(index=False)
                elif output_format == "json":
                    # Conversion en liste de dictionnaires
                    table_dict["data"] = json.loads(df.to_json(orient="records"))
                elif output_format == "html":
                    table_dict["data"] = df.to_html(index=False)
                elif output_format == "excel":
                    # Pour excel, on garde le DataFrame, la conversion sera faite lors de la sauvegarde
                    table_dict["data"] = df
                else:
                    # Format par défaut: pandas
                    table_dict["data"] = df
                
                result.append(table_dict)
            
        except Exception as e:
            logger.error(f"Erreur conversion format {output_format}: {e}")
            # Tenter une conversion de secours
            return await self._fallback_format_conversion(tables, output_format)
        
        return result
    
    async def _fallback_format_conversion(self, tables: List[pd.DataFrame], output_format: str) -> List[Dict[str, Any]]:
        """
        Conversion de secours en cas d'échec de la conversion principale.
        
        Args:
            tables: Liste de DataFrames pandas
            output_format: Format de sortie désiré
            
        Returns:
            Liste de tables au format demandé
        """
        # Conversion de secours simple
        try:
            result = []
            for i, table in enumerate(tables):
                if not isinstance(table, pd.DataFrame):
                    continue
                    
                table_data = {
                    "table_id": i+1,
                    "page": getattr(table, "_page", i+1) if hasattr(table, "_page") else i+1,
                    "rows": len(table),
                    "columns": len(table.columns),
                    "extraction_method": "auto",
                    "confidence": 0.7
                }
                
                if output_format == "pandas":
                    table_data["data"] = table
                elif output_format == "csv":
                    try:
                        table_data["data"] = table.to_csv(index=False)
                    except:
                        table_data["data"] = str(table)
                elif output_format == "json":
                    try:
                        table_data["data"] = json.loads(table.to_json(orient="records"))
                    except:
                        table_data["data"] = table.values.tolist()
                elif output_format == "html":
                    try:
                        table_data["data"] = table.to_html(index=False)
                    except:
                        table_data["data"] = f"<table><tr><td>{str(table)}</td></tr></table>"
                else:
                    table_data["data"] = table.values.tolist()
                
                result.append(table_data)
                
            return result
        except Exception as e:
            logger.error(f"Conversion de secours échouée: {e}")
            # Retourner une liste vide plutôt que de déclencher une autre exception
            return []

    async def get_tables_as_images(
        self,
        file_obj: Union[str, BytesIO],
        pages: Union[str, List[int]] = "all"
    ) -> List[Dict[str, Any]]:
        """
        Obtient les tableaux sous forme d'images.
        
        Args:
            file_obj: Chemin du fichier PDF ou BytesIO
            pages: Pages à analyser
            
        Returns:
            Liste de dictionnaires contenant les images des tableaux
        """
        try:
            # Préparer le fichier source
            file_path, is_temp_file = await self._prepare_file_source(file_obj)
            
            try:
                # Convertir pages en liste
                page_list = await self._parse_page_range(file_path, pages)
                
                # Convertir les pages en images
                images = await self._convert_pdf_to_images(file_path, page_list)
                
                # Détecter les tableaux dans chaque image
                result = []
                
                for i, image in enumerate(images):
                    page_num = page_list[i] + 1 if i < len(page_list) else i + 1
                    
                    # Détecter les régions de tableau
                    table_regions = await self._detect_table_regions(image)
                    
                    for j, region in enumerate(table_regions):
                        x, y, w, h = region
                        
                        # Extraire l'image du tableau
                        table_img = image[y:y+h, x:x+w]
                        
                        # Convertir en base64
                        is_success, buffer = cv2.imencode(".png", table_img)
                        if is_success:
                            img_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Ajouter l'image au résultat
                            result.append({
                                "table_id": j + 1,
                                "page": page_num,
                                "data": img_base64,
                                "mime_type": "image/png",
                                "width": w,
                                "height": h,
                                "position": {"x": x, "y": y, "width": w, "height": h}
                            })
                
                return result
                
            finally:
                # Nettoyer le fichier temporaire si nécessaire
                if is_temp_file and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                    except Exception as e:
                        logger.error(f"Erreur suppression fichier temporaire: {e}")
        
        except Exception as e:
            logger.error(f"Erreur extraction images de tableaux: {e}")
            return []

    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            # Fermer l'executor pour libérer les threads
            self.executor.shutdown(wait=False)
            
            # Nettoyer le répertoire temporaire
            try:
                if os.path.exists(self.temp_dir):
                    for file in os.listdir(self.temp_dir):
                        file_path = os.path.join(self.temp_dir, file)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
            except Exception as e:
                logger.error(f"Erreur nettoyage répertoire temporaire: {e}")
            
            # Si le détecteur IA est utilisé, le nettoyer
            if hasattr(self, 'table_detector'):
                await self.table_detector.cleanup()
            
            # Vider le cache
            self.cache.clear()
            
            logger.info("Ressources du TableExtractor nettoyées")
        except Exception as e:
            logger.error(f"Erreur nettoyage TableExtractor: {e}")