# core/document_processing/table_extractor.py
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, BinaryIO
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
from PIL import Image
import pdf2image

from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings

logger = get_logger("table_extractor")

class TableExtractor:
    """Classe pour extraire des tableaux de fichiers PDF, y compris les PDF scannés."""
    
    def __init__(self):
        """Initialisation de l'extracteur de tableaux."""
        self.executor = ThreadPoolExecutor(max_workers=4)
        # Configuration OCR
        self.tesseract_cmd = os.environ.get("TESSERACT_CMD", "tesseract")
        self.tesseract_lang = os.environ.get("TESSERACT_LANG", "fra+eng")
        self.dpi = int(os.environ.get("OCR_DPI", "300"))
        self.temp_dir = "temp/ocr"
        self.scanned_threshold = 0.1  # Seuil pour détecter les PDF scannés
        
        # S'assurer que le répertoire temp existe
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Configuration de pytesseract
        try:
            if os.path.exists(self.tesseract_cmd) and os.access(self.tesseract_cmd, os.X_OK):
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            
            # Vérifier que tesseract fonctionne
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR initialisé, version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logger.error(f"Erreur initialisation Tesseract OCR: {e}")
            logger.warning("La fonctionnalité OCR pour PDF scannés pourrait ne pas fonctionner correctement")
    
    async def _hybrid_table_detection(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Méthode hybride qui combine la détection par IA et les méthodes traditionnelles.
        
        Cette approche est plus robuste car elle utilise des méthodes complémentaires
        et fusionne les résultats.
        
        Args:
            img: Image au format numpy array
            
        Returns:
            Liste de coordonnées (x, y, w, h) des régions de tableau
        """
        try:
            # Détection par IA
            ai_regions = []
            try:
                if not hasattr(self, 'table_detector') or self.table_detector is None:
                    from core.document_processing.table_detection import TableDetectionModel
                    self.table_detector = TableDetectionModel()
                    await self.table_detector.initialize()
                    
                ai_detections = await self.table_detector.detect_tables(img)
                ai_regions = [(detection["x"], detection["y"], detection["width"], detection["height"]) 
                            for detection in ai_detections]
                logger.info(f"Détection par IA: {len(ai_regions)} tableaux détectés")
            except Exception as e:
                logger.warning(f"Échec de la détection par IA: {e}")
            
            # Détection traditionnelle
            traditional_regions = []
            try:
                traditional_regions = await self._detect_table_regions(img)
                logger.info(f"Détection traditionnelle: {len(traditional_regions)} tableaux détectés")
            except Exception as e:
                logger.warning(f"Échec de la détection traditionnelle: {e}")
            
            # Combiner les résultats en évitant les duplications
            combined_regions = []
            
            # Ajouter d'abord les régions détectées par IA
            combined_regions.extend(ai_regions)
            
            # Ajouter les régions traditionnelles qui ne chevauchent pas trop les régions IA
            for trad_region in traditional_regions:
                tx, ty, tw, th = trad_region
                is_duplicate = False
                
                for ai_region in ai_regions:
                    ax, ay, aw, ah = ai_region
                    
                    # Calculer le chevauchement
                    x_overlap = max(0, min(tx + tw, ax + aw) - max(tx, ax))
                    y_overlap = max(0, min(ty + th, ay + ah) - max(ty, ay))
                    overlap_area = x_overlap * y_overlap
                    
                    # Si le chevauchement est significatif, considérer comme doublon
                    trad_area = tw * th
                    ai_area = aw * ah
                    smaller_area = min(trad_area, ai_area)
                    
                    if overlap_area > 0.5 * smaller_area:
                        is_duplicate = True
                        break
                
                # Si ce n'est pas un doublon, l'ajouter
                if not is_duplicate:
                    combined_regions.append(trad_region)
            
            logger.info(f"Détection hybride: {len(combined_regions)} tableaux uniques détectés")
            
            # Si aucun tableau n'est détecté, essayer une méthode alternative
            if not combined_regions:
                return await self._detect_table_regions_alternate(img)
                
            return combined_regions
            
        except Exception as e:
            logger.error(f"Erreur détection hybride des tableaux: {e}")
            # En cas d'échec total, revenir à la méthode traditionnelle de secours
            return await self._detect_table_regions_alternate(img)
    
    async def _detect_table_regions_with_model(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Détecte les régions de tableau en utilisant le modèle de deep learning.
        
        Args:
            img: Image au format numpy array
            
        Returns:
            Liste de coordonnées (x, y, w, h) des régions de tableau
        """
        try:
            # Si le détecteur n'est pas encore initialisé
            if not hasattr(self, 'table_detector') or self.table_detector is None:
                from core.document_processing.table_detection import TableDetectionModel
                self.table_detector = TableDetectionModel()
                await self.table_detector.initialize()
                logger.info("Détecteur de tableaux par IA initialisé")
            
            # Détection des tableaux avec le modèle
            detections = await self.table_detector.detect_tables(img)
            
            # Conversion au format de coordonnées attendu
            table_regions = []
            for detection in detections:
                x, y = detection["x"], detection["y"]
                w, h = detection["width"], detection["height"]
                
                # Ajouter une marge pour s'assurer de capturer tout le tableau
                margin = 10
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(img.shape[1] - x, w + 2 * margin)
                h = min(img.shape[0] - y, h + 2 * margin)
                
                table_regions.append((x, y, w, h))
                
            logger.info(f"Détection de tableaux par IA: {len(table_regions)} tableaux détectés")
            return table_regions
            
        except Exception as e:
            logger.error(f"Erreur détection des tableaux par IA: {e}")
            # En cas d'échec, on revient à la méthode traditionnelle
            return await self._detect_table_regions(img)
    
    async def extract_tables(
        self, 
        file_path: Union[str, Path, BytesIO], 
        pages: Union[str, List[int]] = "all",
        extraction_method: str = "auto",
        output_format: str = "pandas",
        ocr_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extrait les tableaux d'un fichier PDF, y compris les PDF scannés.
        
        Args:
            file_path: Chemin du fichier PDF ou objet BytesIO
            pages: Pages à analyser ("all" ou liste de numéros de page)
            extraction_method: Méthode d'extraction ("auto", "tabula", "camelot", "pdfplumber", "ocr")
            output_format: Format de sortie ("pandas", "csv", "json", "html")
            ocr_config: Configuration spécifique pour l'OCR {
                "lang": "fra+eng",
                "enhance_image": True,
                "deskew": True,
                "preprocess_type": "thresh",
                "psm": 6  # Page segmentation mode
            }
            
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
                    "psm": 6
                }
                
                # Fusionner avec la configuration fournie
                if ocr_config:
                    ocr_config = {**default_ocr_config, **ocr_config}
                else:
                    ocr_config = default_ocr_config
                
                # Si file_path est un BytesIO, le sauvegarder temporairement
                temp_file = None
                if isinstance(file_path, BytesIO):
                    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                    temp_file.write(file_path.getvalue())
                    temp_file.close()
                    file_path = temp_file.name
                
                # Conversion du chemin en str si c'est un Path
                if isinstance(file_path, Path):
                    file_path = str(file_path)
                
                # Détecter si le PDF est scanné si méthode est auto
                is_scanned = False
                if extraction_method == "auto":
                    is_scanned, confidence = await self._is_scanned_pdf(file_path, pages)
                    logger.info(f"PDF scanné: {is_scanned} (confiance: {confidence:.2f})")
                    
                    extraction_method = "ocr" if is_scanned else await self._detect_best_method(file_path, pages)
                
                # Extraction des tableaux selon la méthode choisie
                loop = asyncio.get_event_loop()
                # Si OCR est activé et aucun tableau n'est trouvé, tenter une approche drastique
                if extraction_method == "ocr" and not tables_list:
                    logger.info("Aucun tableau détecté avec la méthode standard, tentative avec force_grid=True")
                    
                    for img_idx, img in enumerate(images):
                        page_num = image_to_page.get(img_idx, img_idx + 1)
                        
                        # Convertir l'image PIL en format numpy pour OpenCV
                        img_np = np.array(img)
                        
                        # Méthode drastique: considérer toute la page comme un tableau potentiel
                        processed_img = await self._preprocess_image(
                            img_np, 
                            ocr_config.get("enhance_image", True),
                            ocr_config.get("deskew", True),
                            ocr_config.get("preprocess_type", "thresh")
                        )
                        
                        # Extraire les dimensions pour créer une région qui couvre la page entière avec des marges
                        height, width = processed_img.shape[:2]
                        margin_x = width // 10
                        margin_y = height // 10
                        table_region = (margin_x, margin_y, width - 2*margin_x, height - 2*margin_y)
                        
                        # Extraire la région et appliquer OCR avec force_grid=True
                        table_img = processed_img[table_region[1]:table_region[1]+table_region[3], 
                                                table_region[0]:table_region[0]+table_region[2]]
                        
                        # Tenter l'OCR avec force_grid=True pour extraire un tableau même sans lignes visibles
                        df = await self._ocr_table_to_dataframe(
                            table_img,
                            lang=ocr_config.get("lang", self.tesseract_lang),
                            psm=ocr_config.get("psm", 6),
                            force_grid=True
                        )
                        
                        if df is not None and not df.empty and len(df.columns) > 1:
                            df.insert(0, "_page", page_num)
                            df.insert(1, "_table", 1)
                            tables_list.append(df)
                            logger.info(f"Tableau extrait avec méthode de secours pour la page {page_num}")
                            break  # Un seul tableau est suffisant dans ce cas
                elif extraction_method == "tabula":
                    tables = await loop.run_in_executor(
                        self.executor, 
                        lambda: self._extract_with_tabula(file_path, pages)
                    )
                elif extraction_method == "camelot":
                    tables = await loop.run_in_executor(
                        self.executor, 
                        lambda: self._extract_with_camelot(file_path, pages)
                    )
                elif extraction_method == "pdfplumber":
                    tables = await loop.run_in_executor(
                        self.executor, 
                        lambda: self._extract_with_pdfplumber(file_path, pages)
                    )
                else:
                    raise ValueError(f"Méthode d'extraction inconnue: {extraction_method}")
                
                # Nettoyage du fichier temporaire
                if temp_file and os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                
                # Conversion vers le format de sortie désiré
                return await self._convert_output_format(tables, output_format)
                
        except Exception as e:
            logger.error(f"Erreur extraction tableaux: {e}")
            metrics.increment_counter("table_extraction_errors")
            return []
    
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
            
            # Convertir pages en liste
            if pages == "all":
                with pdfplumber.open(file_path) as pdf:
                    page_list = list(range(len(pdf.pages)))
            else:
                if isinstance(pages, str):
                    # Convertir pages au format "1,3,5-7"
                    page_nums = []
                    for part in pages.split(','):
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            page_nums.extend(range(start-1, end))
                        else:
                            page_nums.append(int(part)-1)
                    page_list = page_nums
                else:
                    page_list = [p-1 for p in pages]
            
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
                    
                    # Calculer la taille de la page en pixels
                    width, height = float(page.width), float(page.height)
                    # Convertir en pixels
                    width_px = width * 72  # 72 DPI est standard pour PDF
                    height_px = height * 72
                    total_pixels += width_px * height_px
            
            # Calcul de la densité de texte
            char_density = total_chars / max(1, total_pixels) * 10000  # Caractères par 10000 pixels
            
            # Heuristique: une densité très faible indique un PDF scanné
            # La valeur seuil peut être ajustée en fonction des résultats
            is_scanned = char_density < self.scanned_threshold
            
            # Calcul du niveau de confiance
            if char_density < 0.01:  # Quasiment pas de texte
                confidence = 0.95
            elif char_density < self.scanned_threshold:
                confidence = 0.8
            elif char_density < self.scanned_threshold * 2:
                confidence = 0.5
            else:
                confidence = 0.1
            
            return is_scanned, confidence
            
        except Exception as e:
            logger.error(f"Erreur détection PDF scanné: {e}")
            # En cas d'erreur, supposons que c'est un PDF normal
            return False, 0.0
    
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
            if pages == "all":
                # Convertir tout le document (limité à 20 pages pour éviter les problèmes de mémoire)
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    page_list = list(range(1, min(len(pdf.pages) + 1, 21)))
            elif isinstance(pages, str):
                # Format "1,3,5-7"
                page_list = []
                for part in pages.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        page_list.extend(range(start, end + 1))
                    else:
                        page_list.append(int(part))
            else:
                page_list = pages
            
            # Convertir le PDF en images
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(
                self.executor,
                lambda: pdf2image.convert_from_path(
                    file_path, 
                    dpi=self.dpi, 
                    first_page=min(page_list) if page_list else 1,
                    last_page=max(page_list) if page_list else None
                )
            )
            
            # Créer un dictionnaire pour associer l'index de l'image à la page
            image_to_page = {}
            if pages == "all":
                for i, _ in enumerate(images):
                    image_to_page[i] = i + 1
            else:
                for i, page_num in enumerate(page_list):
                    if i < len(images):
                        image_to_page[i] = page_num
            
            # Liste pour stocker les DataFrames
            tables_list = []  # Initialisation correcte de tables_list
            
            # Traiter chaque image pour détecter et extraire les tableaux
            for img_idx, img in enumerate(images):
                page_num = image_to_page.get(img_idx, img_idx + 1)
                
                # Convertir l'image PIL en format numpy pour OpenCV
                img_np = np.array(img)
                
                # Prétraitement de l'image
                processed_img = await self._preprocess_image(
                    img_np, 
                    ocr_config.get("enhance_image", True),
                    ocr_config.get("deskew", True),
                    ocr_config.get("preprocess_type", "thresh")
                )
                
                # Détection des tableaux
                table_regions = await self._detect_table_regions(processed_img)
                
                logger.info(f"Page {page_num}: {len(table_regions)} tableaux détectés")
                
                # Pour chaque région de tableau détectée
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
                            lang=ocr_config.get("lang", self.tesseract_lang),
                            psm=ocr_config.get("psm", 6),
                            force_grid=ocr_config.get("force_grid", False)
                        )
                        
                        # Ajouter à la liste si le DataFrame n'est pas vide
                        if df is not None and not df.empty and len(df.columns) > 1:
                            # Ajouter des colonnes d'information
                            df = df.copy()  # Créer une copie pour éviter les avertissements SettingWithCopyWarning
                            df.insert(0, "_page", page_num)
                            df.insert(1, "_table", i + 1)
                            tables_list.append(df)
                            
                    except Exception as e:
                        logger.error(f"Erreur extraction tableau {i+1} sur page {page_num}: {e}")
            
            logger.info(f"OCR terminé: {len(tables_list)} tableaux extraits")
            return tables_list
            
        except Exception as e:
            logger.error(f"Erreur extraction OCR: {e}")
            return []  # Retourner une liste vide en cas d'erreur
    
    async def _preprocess_image(
        self, 
        img: np.ndarray, 
        enhance: bool = True, 
        deskew: bool = True,
        preprocess_type: str = "thresh"
    ) -> np.ndarray:
        """
        Prétraite une image pour améliorer la détection des tableaux.
        
        Args:
            img: Image au format numpy
            enhance: Si True, améliore le contraste
            deskew: Si True, corrige l'inclinaison
            preprocess_type: Type de prétraitement ('thresh', 'blur', 'adaptive')
            
        Returns:
            Image prétraitée
        """
        # Convertir en niveaux de gris si l'image est en couleur
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Appliquer un filtre de netteté pour améliorer les contours
        if enhance:
            kernel = np.array([[-1, -1, -1], 
                               [-1,  9, -1], 
                               [-1, -1, -1]])
            gray = cv2.filter2D(gray, -1, kernel)
        
        # Correction de l'inclinaison si nécessaire
        if deskew:
            gray = await self._deskew_image(gray)
        
        # Appliquer le prétraitement spécifié
        if preprocess_type == "thresh":
            # Seuillage adaptatif pour améliorer les lignes
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        elif preprocess_type == "blur":
            # Flou gaussien pour réduire le bruit
            processed = cv2.GaussianBlur(gray, (5, 5), 0)
            _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif preprocess_type == "adaptive":
            # Seuillage adaptatif avec bloc plus grand
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 15, 2
            )
        else:
            # Seuillage Otsu par défaut
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return processed
    
    async def _deskew_image(self, img: np.ndarray) -> np.ndarray:
        """
        Corrige l'inclinaison d'une image.
        
        Args:
            img: Image en niveaux de gris
            
        Returns:
            Image corrigée
        """
        # Appliquer un seuillage pour obtenir une image binaire
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Trouver les contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer les petits contours
        contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        if not contours:
            return img
        
        # Déterminer l'angle de rotation
        angles = []
        for contour in contours:
            # Calculer la boîte englobante
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Déterminer l'angle
            angle = rect[2]
            
            # Normaliser l'angle entre -45 et 45 degrés
            if angle < -45:
                angle = 90 + angle
            
            angles.append(angle)
        
        # Utiliser l'angle médian pour éviter les valeurs aberrantes
        angle = np.median(angles)
        
        # Si l'angle est significatif (> 0.5 degrés), corriger l'inclinaison
        if abs(angle) > 0.5:
            # Centre de l'image
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            
            # Matrice de rotation
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Appliquer la rotation
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
        
        return img
    
    async def _detect_table_regions(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Détecte les régions de l'image qui contiennent des tableaux.
        
        Args:
            img: Image prétraitée
            
        Returns:
            Liste de coordonnées (x, y, w, h) des régions de tableau
        """
        # Appliquer une morphologie pour faire ressortir les lignes horizontales et verticales
        kernel_h = np.ones((1, 40), np.uint8)
        kernel_v = np.ones((40, 1), np.uint8)
        
        # Extraire les lignes horizontales et verticales
        horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_h)
        vertical = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_v)
        
        # Combiner les lignes horizontales et verticales
        tables = cv2.add(horizontal, vertical)
        
        # Dilatation pour connecter les éléments
        kernel = np.ones((5, 5), np.uint8)
        tables = cv2.dilate(tables, kernel, iterations=2)
        
        # Trouver les contours des régions de tableau
        contours, _ = cv2.findContours(tables, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer les contours trop petits
        min_area = 0.003 * img.shape[0] * img.shape[1]  # 0.3% de l'image
        table_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Si aucun tableau n'est détecté avec la méthode des lignes, essayer une autre approche
        if not table_contours:
            # Essayer une approche basée sur les blocs de texte
            return await self._detect_table_regions_alternate(img)
        
        # Convertir les contours en rectangles (x, y, w, h)
        table_regions = []
        for contour in table_contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Ajouter une marge autour du tableau
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)
            
            table_regions.append((x, y, w, h))
        
        return table_regions
    
    async def _detect_table_regions_alternate(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Méthode alternative améliorée de détection des tableaux basée sur le contenu structuré.
        Particulièrement utile pour les tableaux avec des lignes fines ou des bordures peu visibles.
        
        Args:
            img: Image prétraitée
            
        Returns:
            Liste de coordonnées (x, y, w, h) des régions de tableau
        """
        try:
            # Détection basée sur la reconnaissance des lignes horizontales et verticales
            # avec plusieurs seuils pour augmenter la sensibilité
            
            height, width = img.shape[:2]
            
            # 1. Détection traditionnelle des lignes avec morphologie
            kernels_h = [np.ones((1, length), np.uint8) for length in [30, 40, 50]]
            kernels_v = [np.ones((length, 1), np.uint8) for length in [30, 40, 50]]
            
            # Essayer différents kernels pour augmenter la sensibilité
            horizontal_regions = []
            vertical_regions = []
            
            for kernel_h in kernels_h:
                horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_h)
                horizontal_regions.append(horizontal)
                
            for kernel_v in kernels_v:
                vertical = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_v)
                vertical_regions.append(vertical)
            
            # Combiner les différentes détections
            horizontal_combined = np.zeros_like(img)
            for h_region in horizontal_regions:
                horizontal_combined = np.maximum(horizontal_combined, h_region)
                
            vertical_combined = np.zeros_like(img)
            for v_region in vertical_regions:
                vertical_combined = np.maximum(vertical_combined, v_region)
            
            # Combiner les lignes horizontales et verticales
            table_regions = cv2.add(horizontal_combined, vertical_combined)
            
            # 2. Si la détection traditionnelle ne fonctionne pas, essayer une approche basée sur le texte
            contours, _ = cv2.findContours(table_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_table_area = 0.03 * width * height  # Au moins 3% de l'image
            
            valid_table_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_table_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    valid_table_regions.append((x, y, w, h))
            
            # 3. Si aucune région n'est trouvée, essayer l'approche basée sur les blocs de texte
            if not valid_table_regions:
                # Dilater l'image pour connecter les caractères proches
                kernel = np.ones((5, 20), np.uint8)  # Kernel plus grand pour mieux capturer les lignes de texte
                dilated = cv2.dilate(img, kernel, iterations=2)
                
                # Trouver les contours des blocs de texte
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filtrer les contours trop petits
                text_blocks = []
                min_block_area = 0.0005 * width * height  # 0.05% de l'image
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > min_block_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        text_blocks.append((x, y, w, h))
                
                # Trier les blocs par position verticale (y)
                text_blocks.sort(key=lambda b: b[1])
                
                # Grouper les blocs en lignes (possibles lignes d'un tableau)
                rows = []
                current_row = [text_blocks[0]] if text_blocks else []
                
                for block in text_blocks[1:]:
                    _, prev_y, _, prev_h = current_row[-1] if current_row else (0, 0, 0, 0)
                    _, curr_y, _, _ = block
                    
                    # Si ce bloc est proche verticalement du précédent, on le considère sur la même ligne
                    if abs(curr_y - prev_y) < 20:  # 20 pixels de tolérance
                        current_row.append(block)
                    else:
                        # Nouvelle ligne
                        if current_row:
                            rows.append(current_row)
                        current_row = [block]
                
                # Ajouter la dernière ligne si elle existe
                if current_row:
                    rows.append(current_row)
                
                # Considérer comme tableau si on a au moins 3 lignes avec au moins 2 blocs chacune
                structured_rows = [row for row in rows if len(row) >= 2]
                
                if len(structured_rows) >= 3:
                    # Déterminer les limites du tableau
                    min_x = min([min([block[0] for block in row]) for row in structured_rows])
                    min_y = min([min([block[1] for block in row]) for row in structured_rows])
                    max_x = max([max([block[0] + block[2] for block in row]) for row in structured_rows])
                    max_y = max([max([block[1] + block[3] for block in row]) for row in structured_rows])
                    
                    # Ajouter une marge
                    margin = 20
                    min_x = max(0, min_x - margin)
                    min_y = max(0, min_y - margin)
                    max_x = min(width, max_x + margin)
                    max_y = min(height, max_y + margin)
                    
                    valid_table_regions.append((min_x, min_y, max_x - min_x, max_y - min_y))
            
            # 4. Si toujours aucune région détectée, utiliser l'heuristique des zones horizontales alignées
            if not valid_table_regions:
                # Essayer de détecter des lignes horizontales de façon plus agressive
                more_aggressive_kernel = np.ones((1, 100), np.uint8)
                horizontal_aggressive = cv2.morphologyEx(img, cv2.MORPH_OPEN, more_aggressive_kernel)
                
                # Trouver les contours des lignes horizontales
                contours, _ = cv2.findContours(horizontal_aggressive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Si au moins 3 lignes horizontales sont trouvées, c'est probablement un tableau
                if len(contours) >= 3:
                    # Trier les contours par position y
                    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
                    
                    # Obtenir les rectangles englobants
                    bounding_rects = [cv2.boundingRect(c) for c in contours]
                    
                    # Vérifier l'alignement horizontal (même largeur approximative)
                    widths = [rect[2] for rect in bounding_rects]
                    avg_width = sum(widths) / len(widths)
                    aligned_rects = [rect for rect in bounding_rects if abs(rect[2] - avg_width) / avg_width < 0.3]
                    
                    if len(aligned_rects) >= 3:
                        # Déterminer les limites du tableau
                        min_x = min(rect[0] for rect in aligned_rects)
                        min_y = min(rect[1] for rect in aligned_rects)
                        max_x = max(rect[0] + rect[2] for rect in aligned_rects)
                        max_y = max(rect[1] + rect[3] for rect in aligned_rects)
                        
                        # Ajouter une marge
                        margin = 20
                        min_x = max(0, min_x - margin)
                        min_y = max(0, min_y - margin)
                        max_x = min(width, max_x + margin)
                        max_y = min(height, max_y + margin)
                        
                        valid_table_regions.append((min_x, min_y, max_x - min_x, max_y - min_y))
            
            # 5. Dernier recours: Si toujours rien n'est détecté et si le document a une structure tabulaire
            # (détectée par la présence de texte aligné), considérer une grande partie de la page
            if not valid_table_regions:
                # Appliquer OCR pour trouver les lignes de texte
                from pytesseract import Output
                import pytesseract
                
                # Convertir en image PIL
                pil_img = Image.fromarray(cv2.bitwise_not(img))  # Inverser pour OCR
                
                # Obtenir les informations de ligne et de mot
                ocr_data = pytesseract.image_to_data(pil_img, output_type=Output.DICT, config='--psm 6')
                
                # Créer des boîtes autour des lignes de texte
                n_boxes = len(ocr_data['text'])
                line_boxes = {}
                
                for i in range(n_boxes):
                    if int(ocr_data['conf'][i]) > 0:  # Ignorer les résultats de faible confiance
                        x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                        line_num = ocr_data['line_num'][i]
                        
                        if line_num not in line_boxes:
                            line_boxes[line_num] = [x, y, w, h]
                        else:
                            # Étendre la boîte existante
                            curr_x, curr_y, curr_w, curr_h = line_boxes[line_num]
                            new_x = min(curr_x, x)
                            new_y = min(curr_y, y)
                            new_w = max(curr_x + curr_w, x + w) - new_x
                            new_h = max(curr_y + curr_h, y + h) - new_y
                            line_boxes[line_num] = [new_x, new_y, new_w, new_h]
                
                # Vérifier si les lignes sont alignées (possible structure de tableau)
                if len(line_boxes) >= 3:
                    line_values = list(line_boxes.values())
                    left_positions = [box[0] for box in line_values]
                    widths = [box[2] for box in line_values]
                    
                    # Calculer la variance des positions de gauche et des largeurs
                    left_variance = np.var(left_positions) / (np.mean(left_positions) + 1e-5)
                    width_variance = np.var(widths) / (np.mean(widths) + 1e-5)
                    
                    # Si les lignes sont bien alignées, c'est potentiellement un tableau
                    if left_variance < 0.1 and width_variance < 0.3:
                        min_x = min(box[0] for box in line_values)
                        min_y = min(box[1] for box in line_values)
                        max_x = max(box[0] + box[2] for box in line_values)
                        max_y = max(box[1] + box[3] for box in line_values)
                        
                        # Ajouter une marge
                        margin_h = 30
                        margin_v = 50  # Marge verticale plus grande pour inclure les en-têtes
                        min_x = max(0, min_x - margin_h)
                        min_y = max(0, min_y - margin_v)
                        max_x = min(width, max_x + margin_h)
                        max_y = min(height, max_y + margin_v)
                        
                        valid_table_regions.append((min_x, min_y, max_x - min_x, max_y - min_y))
            
            # 6. Si après toutes ces tentatives on ne trouve rien, tenter une approche drastique
            # basée sur l'analyse globale de la structure du document
            if not valid_table_regions:
                # Recherche de pattern répétitif de type tableau: lignes horizontales régulièrement espacées
                # et texte organisé en colonnes
                
                # Projection horizontale pour détecter les rangées
                h_projection = np.sum(img, axis=1)
                h_projection = h_projection / np.max(h_projection)  # Normaliser
                
                # Trouver les pics (possibles rangées de tableau)
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(1 - h_projection, height=0.5, distance=20)
                
                # Si on a plusieurs pics régulièrement espacés, c'est peut-être un tableau
                if len(peaks) >= 4:  # Au moins 4 lignes horizontales
                    spacing = np.diff(peaks)
                    avg_spacing = np.mean(spacing)
                    regular_spacing = np.all(np.abs(spacing - avg_spacing) < 0.3 * avg_spacing)
                    
                    if regular_spacing:
                        # Projection verticale pour détecter les colonnes
                        v_projection = np.sum(img, axis=0)
                        v_projection = v_projection / np.max(v_projection)  # Normaliser
                        
                        v_peaks, _ = find_peaks(1 - v_projection, height=0.5, distance=20)
                        
                        # Si on a plusieurs pics verticaux, c'est presque certainement un tableau
                        if len(v_peaks) >= 3:  # Au moins 3 colonnes/lignes verticales
                            min_y = max(0, peaks[0] - avg_spacing/2)
                            max_y = min(height, peaks[-1] + avg_spacing/2)
                            min_x = max(0, v_peaks[0] - 20)
                            max_x = min(width, v_peaks[-1] + 20)
                            
                            valid_table_regions.append((int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)))
            
            # 7. Dernier recours absolu: si nous savons que c'est un document professionnel
            # avec une forte probabilité de contenir un tableau au centre, utiliser une heuristique hardcodée
            if not valid_table_regions:
                # Vérifier s'il y a une zone dense de pixels noirs au centre du document
                # qui pourrait être un tableau
                center_region = img[height//3:2*height//3, width//4:3*width//4]
                center_density = np.sum(center_region) / (center_region.size)
                
                # Si la densité est suffisamment élevée (beaucoup de pixels noirs), considérer comme un tableau
                if center_density > 10:  # Valeur arbitraire à ajuster
                    # Utiliser la région centrale avec une marge
                    table_x = width//4 - 50
                    table_y = height//3 - 50
                    table_w = width//2 + 100
                    table_h = height//3 + 100
                    
                    valid_table_regions.append((
                        max(0, table_x),
                        max(0, table_y),
                        min(width - table_x, table_w),
                        min(height - table_y, table_h)
                    ))
            
            # Enfin, si vraiment rien n'est détecté, considérer toute la page comme un tableau
            if not valid_table_regions:
                # Exclure les marges (10% de chaque côté)
                margin_x = width // 10
                margin_y = height // 10
                valid_table_regions.append((
                    margin_x,
                    margin_y,
                    width - 2 * margin_x,
                    height - 2 * margin_y
                ))
                
            return valid_table_regions
            
        except Exception as e:
            logger.error(f"Erreur détection alternative des tableaux: {e}", exc_info=True)
            # En cas d'échec, retourner un rectangle qui couvre la majorité de la page
            return [(width//10, height//10, width*8//10, height*8//10)]
    
    async def _enhance_table_image(self, img: np.ndarray) -> np.ndarray:
        """
        Améliore l'image d'un tableau pour l'OCR.
        
        Args:
            img: Image du tableau
            
        Returns:
            Image améliorée
        """
        # Réduire le bruit avec un filtre bilatéral
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Augmenter le contraste
        _, enhanced = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return enhanced
    
    async def _ocr_table_to_dataframe(
        self, 
        img: np.ndarray, 
        lang: str = "fra+eng",
        psm: int = 6,
        force_grid: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Applique l'OCR à l'image d'un tableau et convertit le résultat en DataFrame.
        Version améliorée pour mieux gérer les tableaux avec bordures fines ou manuscrits.
        
        Args:
            img: Image du tableau
            lang: Langues pour Tesseract (fra+eng par défaut)
            psm: Mode de segmentation de page Tesseract
            force_grid: Force la création d'une grille même si le tableau n'est pas bien délimité
            
        Returns:
            DataFrame contenant les données du tableau
        """
        try:
            from pytesseract import Output
            import pytesseract
            from PIL import Image
            import pandas as pd
            import numpy as np
            
            # Sauvegarder l'image temporairement
            temp_img_path = os.path.join(self.temp_dir, f"table_{uuid.uuid4()}.png")
            cv2.imwrite(temp_img_path, img)
            
            # Configuration OCR - ajouter l'option de tableau pour améliorer la détection
            custom_config = f'--oem 3 --psm {psm} -l {lang}'
            
            # Inverser l'image pour l'OCR (texte noir sur fond blanc)
            pil_img = Image.open(temp_img_path)
            
            # 1. D'abord, essayer de détecter directement la structure du tableau
            try:
                # Obtenir les données avec des informations de position et de structure
                loop = asyncio.get_event_loop()
                ocr_data = await loop.run_in_executor(
                    self.executor,
                    lambda: pytesseract.image_to_data(
                        pil_img,
                        config=custom_config,
                        output_type=Output.DICT
                    )
                )
                
                # Si aucun texte n'est reconnu, retourner None
                if not ocr_data['text'] or all(txt == '' for txt in ocr_data['text']):
                    # Essayer avec une autre méthode
                    pass
                else:
                    # Analyser les résultats OCR pour détecter la structure du tableau
                    
                    # Regrouper par ligne (par block_num ou line_num)
                    grouped_by_line = {}
                    for i in range(len(ocr_data['text'])):
                        if int(ocr_data['conf'][i]) <= 0:  # Ignorer les résultats de faible confiance
                            continue
                            
                        line_num = ocr_data['line_num'][i]
                        if line_num not in grouped_by_line:
                            grouped_by_line[line_num] = []
                            
                        grouped_by_line[line_num].append({
                            'text': ocr_data['text'][i],
                            'left': ocr_data['left'][i],
                            'width': ocr_data['width'][i],
                            'center_x': ocr_data['left'][i] + ocr_data['width'][i] // 2,
                            'conf': ocr_data['conf'][i]
                        })
                    
                    # Trier les lignes par position verticale (top)
                    line_nums = sorted(grouped_by_line.keys(), key=lambda k: ocr_data['top'][ocr_data['line_num'].index(k)])
                    
                    # Ignorer les lignes sans texte significatif
                    valid_lines = [ln for ln in line_nums if any(word['text'].strip() for word in grouped_by_line[ln])]
                    
                    if not valid_lines or len(valid_lines) < 2:
                        # Pas assez de lignes valides, essayer une autre approche
                        pass
                    else:
                        # Détecter les colonnes en analysant les positions horizontales
                        # Pour chaque ligne, trier les mots par position
                        for line_num in valid_lines:
                            grouped_by_line[line_num] = sorted(grouped_by_line[line_num], key=lambda w: w['left'])
                        
                        # Trouver le nombre maximal de mots dans une ligne pour estimer le nombre de colonnes
                        max_words = max(len(grouped_by_line[ln]) for ln in valid_lines)
                        
                        if max_words < 2:
                            # Pas assez de colonnes, essayer une autre approche
                            pass
                        else:
                            # Déduire les positions des colonnes à partir des mots
                            # Pour chaque ligne, créer une liste de centres de mot
                            all_centers = []
                            for ln in valid_lines:
                                centers = [word['center_x'] for word in grouped_by_line[ln]]
                                all_centers.extend(centers)
                            
                            # Effectuer un clustering pour trouver les centres des colonnes
                            from sklearn.cluster import KMeans
                            
                            # Estimer le nombre de colonnes
                            n_columns = min(max_words, 10)  # Limiter à 10 colonnes maximum
                            
                            if len(all_centers) < n_columns:
                                # Pas assez de données pour le clustering
                                n_columns = len(all_centers)
                            
                            if n_columns < 2:
                                # Pas assez de colonnes, essayer une autre approche
                                pass
                            else:
                                centers_array = np.array(all_centers).reshape(-1, 1)
                                kmeans = KMeans(n_clusters=n_columns, random_state=0).fit(centers_array)
                                column_centers = sorted(kmeans.cluster_centers_.flatten())
                                
                                # Créer un DataFrame à partir des données
                                table_data = []
                                
                                for ln in valid_lines:
                                    row_data = [''] * n_columns
                                    for word in grouped_by_line[ln]:
                                        # Déterminer à quelle colonne appartient ce mot
                                        distances = [abs(word['center_x'] - center) for center in column_centers]
                                        col_idx = distances.index(min(distances))
                                        
                                        # Ajouter le texte à la colonne appropriée
                                        if row_data[col_idx]:
                                            row_data[col_idx] += ' ' + word['text'].strip()
                                        else:
                                            row_data[col_idx] = word['text'].strip()
                                    
                                    table_data.append(row_data)
                                
                                # Créer le DataFrame
                                df = pd.DataFrame(table_data)
                                
                                # Utiliser la première ligne comme en-tête si approprié
                                if len(df) > 1 and self._is_header_row(df.iloc[0]):
                                    df.columns = df.iloc[0]
                                    df = df.iloc[1:].reset_index(drop=True)
                                
                                # Nettoyer le DataFrame
                                df = df.apply(lambda x: x.str.strip() if x.dtype == object else x)
                                
                                # Vérifier si le DataFrame est valide (plus d'une colonne et au moins une ligne)
                                if df.shape[1] >= 2 and df.shape[0] >= 1:
                                    return df
            except Exception as e:
                logger.debug(f"Première méthode OCR échouée: {e}")
                # Continuer avec les méthodes alternatives
            
            # 2. Si la première méthode échoue, essayer avec une détection plus aggressive de la structure
            try:
                # Appliquer une segmentation structurée
                custom_config_structured = f'--oem 3 --psm 6 -l {lang}'
                
                # Utiliser TSV pour capturer la structure
                loop = asyncio.get_event_loop()
                tsv_output = await loop.run_in_executor(
                    self.executor,
                    lambda: pytesseract.image_to_string(
                        pil_img,
                        config=custom_config_structured + ' tsv',
                        output_type=pytesseract.Output.DATAFRAME
                    )
                )
                
                # Si des données sont détectées
                if not tsv_output.empty:
                    # Filtrer et nettoyer
                    text_data = tsv_output[tsv_output['conf'] > 0]
                    
                    if text_data.empty:
                        # Aucun texte valide détecté
                        pass
                    else:
                        # Regrouper par position verticale pour former des lignes
                        text_data['line_group'] = text_data['top'] // 15  # Regrouper par blocs de 15 pixels
                        
                        # Créer une liste pour stocker les lignes du tableau
                        table_rows = []
                        
                        # Pour chaque groupe de ligne
                        for line_num, group in text_data.groupby('line_group'):
                            # Trier les éléments par position horizontale
                            sorted_group = group.sort_values('left')
                            
                            # Extraire le texte de chaque élément
                            row_texts = sorted_group['text'].tolist()
                            
                            # Ignorer les lignes vides
                            if not any(text.strip() for text in row_texts):
                                continue
                                
                            # Ajouter cette ligne à notre tableau
                            table_rows.append(row_texts)
                        
                        # Si des lignes ont été extraites
                        if table_rows:
                            # Déterminer le nombre maximum de colonnes
                            max_cols = max(len(row) for row in table_rows)
                            
                            # Normaliser les lignes
                            normalized_rows = []
                            for row in table_rows:
                                # Étendre la ligne avec des valeurs vides si nécessaire
                                normalized_row = row + [''] * (max_cols - len(row))
                                normalized_rows.append(normalized_row)
                            
                            # Créer un DataFrame
                            df = pd.DataFrame(normalized_rows)
                            
                            # Utiliser la première ligne comme en-tête si approprié
                            if len(df) > 1 and self._is_header_row(df.iloc[0]):
                                df.columns = df.iloc[0]
                                df = df.iloc[1:].reset_index(drop=True)
                            
                            # Nettoyer le DataFrame
                            df = df.apply(lambda x: x.str.strip() if x.dtype == object else x)
                            
                            # Vérifier que le DataFrame a au moins 2 colonnes et une ligne
                            if df.shape[1] >= 2 and df.shape[0] >= 1:
                                return df
            except Exception as e:
                logger.debug(f"Deuxième méthode OCR échouée: {e}")
                # Continuer avec la méthode suivante
            
            # 3. Si les méthodes précédentes échouent, essayer une approche basée sur une grille régulière
            if force_grid:
                try:
                    # Obtenir le texte brut
                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(
                        self.executor,
                        lambda: pytesseract.image_to_string(
                            pil_img,
                            config=custom_config
                        )
                    )
                    
                    # Diviser en lignes
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    
                    if not lines or len(lines) < 2:
                        # Pas assez de texte détecté
                        return None
                    
                    # Essayer de détecter un séparateur de colonne commun (espace ou tabulation)
                    # en analysant les occurrences de caractères d'espacement
                    potential_separators = ['\t', ' | ', ' + ', '  ', ' ']
                    separator = None
                    
                    for sep in potential_separators:
                        if all(sep in line for line in lines[:min(5, len(lines))]):
                            separator = sep
                            break
                    
                    if separator:
                        # Diviser chaque ligne selon le séparateur
                        table_data = [line.split(separator) for line in lines]
                        
                        # Déterminer le nombre maximum de colonnes
                        max_cols = max(len(row) for row in table_data)
                        
                        # Normaliser pour que toutes les lignes aient le même nombre de colonnes
                        for i in range(len(table_data)):
                            if len(table_data[i]) < max_cols:
                                table_data[i].extend([''] * (max_cols - len(table_data[i])))
                        
                        # Créer le DataFrame
                        df = pd.DataFrame(table_data)
                        
                        # Utiliser la première ligne comme en-tête si approprié
                        if len(df) > 1 and self._is_header_row(df.iloc[0]):
                            df.columns = df.iloc[0]
                            df = df.iloc[1:].reset_index(drop=True)
                        
                        # Nettoyer les données
                        df = df.apply(lambda x: x.str.strip() if x.dtype == object else x)
                        
                        return df
                    else:
                        # Si aucun séparateur commun n'est trouvé, essayer de diviser en colonnes fixes
                        # basées sur la distribution du texte
                        
                        # Trouver la ligne la plus longue
                        max_line_length = max(len(line) for line in lines)
                        
                        # Diviser en colonnes régulières (estimation naïve)
                        n_columns = max(2, min(10, max_line_length // 10))  # Entre 2 et 10 colonnes
                        column_width = max_line_length // n_columns
                        
                        table_data = []
                        for line in lines:
                            row = []
                            for i in range(n_columns):
                                start = i * column_width
                                end = (i + 1) * column_width if i < n_columns - 1 else len(line)
                                cell = line[start:end].strip() if start < len(line) else ""
                                row.append(cell)
                            table_data.append(row)
                        
                        # Créer le DataFrame
                        df = pd.DataFrame(table_data)
                        
                        # Utiliser la première ligne comme en-tête si approprié
                        if len(df) > 1 and self._is_header_row(df.iloc[0]):
                            df.columns = df.iloc[0]
                            df = df.iloc[1:].reset_index(drop=True)
                        
                        # Nettoyer les données
                        df = df.apply(lambda x: x.str.strip() if x.dtype == object else x)
                        
                        return df
                except Exception as e:
                    logger.error(f"Troisième méthode OCR échouée: {e}")
                    # Continuer avec la prochaine tentative ou retourner None
            
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_img_path):
                os.unlink(temp_img_path)
                
            # Si toutes les méthodes ont échoué, retourner None
            return None
                
        except Exception as e:
            logger.error(f"Erreur OCR tableau: {e}")
            return None
    
    # Les méthodes existantes restent inchangées
    async def _detect_best_method(self, file_path: str, pages: Union[str, List[int]]) -> str:
        """
        Détecte la meilleure méthode d'extraction pour le PDF.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Pages à analyser
            
        Returns:
            Nom de la méthode à utiliser
        """
        try:
            # Heuristique simple: essayer d'extraire avec chaque méthode et voir laquelle donne les meilleurs résultats
            loop = asyncio.get_event_loop()
            
            # Test avec Camelot (généralement meilleur pour tableaux complexes avec bordures)
            camelot_tables = await loop.run_in_executor(
                self.executor,
                lambda: self._test_camelot(file_path, pages)
            )
            
            # Test avec Tabula (généralement meilleur pour tableaux simples sans bordures)
            tabula_tables = await loop.run_in_executor(
                self.executor,
                lambda: self._test_tabula(file_path, pages)
            )
            
            # Choix basé sur le nombre et la qualité des tableaux détectés
            if camelot_tables["score"] > tabula_tables["score"]:
                logger.info(f"Méthode choisie: camelot (score: {camelot_tables['score']:.2f})")
                return "camelot"
            else:
                logger.info(f"Méthode choisie: tabula (score: {tabula_tables['score']:.2f})")
                return "tabula"
            
        except Exception as e:
            logger.warning(f"Erreur détection méthode, utilisation de tabula par défaut: {e}")
            return "tabula"
    
    def _test_camelot(self, file_path: str, pages: Union[str, List[int]]) -> Dict[str, Any]:
        """Test rapide avec Camelot."""
        try:
            if pages == "all":
                # Limiter à quelques pages pour le test
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    max_pages = min(len(pdf.pages), 5)
                pages = f"1-{max_pages}"
            
            tables = camelot.read_pdf(
                file_path, 
                pages=pages,
                flavor="lattice",  # essayer d'abord avec détection de bordures
                suppress_stdout=True
            )
            
            # Calcul du score basé sur le nombre de tables et leur qualité
            if len(tables) == 0:
                return {"score": 0, "tables": 0}
            
            avg_accuracy = sum(table.parsing_report['accuracy'] for table in tables) / len(tables)
            return {"score": avg_accuracy * len(tables), "tables": len(tables)}
            
        except Exception as e:
            logger.debug(f"Test camelot échoué: {e}")
            return {"score": 0, "tables": 0}
    
    def _test_tabula(self, file_path: str, pages: Union[str, List[int]]) -> Dict[str, Any]:
        """Test rapide avec Tabula."""
        try:
            if pages == "all":
                # Limiter à quelques pages pour le test
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    max_pages = min(len(pdf.pages), 5)
                pages = [i+1 for i in range(max_pages)]
            
            tables = tabula.read_pdf(
                file_path,
                pages=pages,
                multiple_tables=True,
                guess=True
            )
            
            # Calcul du score basé sur le nombre de tables et leur complexité
            if len(tables) == 0:
                return {"score": 0, "tables": 0}
            
            # Évaluation heuristique de la qualité basée sur les dimensions et les valeurs non-NaN
            avg_size = sum(len(table) * len(table.columns) for table in tables) / len(tables)
            avg_non_nan = sum(table.count().sum() / (len(table) * len(table.columns)) for table in tables) / len(tables)
            
            return {"score": avg_size * avg_non_nan * len(tables), "tables": len(tables)}
            
        except Exception as e:
            logger.debug(f"Test tabula échoué: {e}")
            return {"score": 0, "tables": 0}
    
    def _extract_with_tabula(self, file_path: str, pages: Union[str, List[int]]) -> List[pd.DataFrame]:
        """
        Extrait les tableaux avec tabula-py.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Pages à analyser
            
        Returns:
            Liste de DataFrames pandas contenant les tableaux
        """
        try:
            logger.info(f"Extraction avec tabula: {file_path}")
            tables = tabula.read_pdf(
                file_path,
                pages=pages,
                multiple_tables=True,
                guess=True,
                pandas_options={'header': None}  # Ne pas supposer que la première ligne est l'en-tête
            )
            
            # Traitement des tableaux
            processed_tables = []
            for i, table in enumerate(tables):
                if len(table) > 0 and len(table.columns) > 0:
                    # Nettoyage: suppression des colonnes et lignes vides
                    table = table.dropna(how='all', axis=0).dropna(how='all', axis=1)
                    
                    # Si le tableau a au moins une ligne et une colonne
                    if not table.empty and len(table.columns) > 0:
                        # Détection des en-têtes si la première ligne semble être des en-têtes
                        if self._is_header_row(table.iloc[0]):
                            table.columns = table.iloc[0]
                            table = table.iloc[1:].reset_index(drop=True)
                        
                        processed_tables.append(table)
            
            return processed_tables
            
        except Exception as e:
            logger.error(f"Erreur extraction tabula: {e}")
            return []
    
    def _extract_with_camelot(self, file_path: str, pages: Union[str, List[int]]) -> List[pd.DataFrame]:
        """
        Extrait les tableaux avec camelot.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Pages à analyser
            
        Returns:
            Liste de DataFrames pandas contenant les tableaux
        """
        try:
            logger.info(f"Extraction avec camelot: {file_path}")
            
            # Formater les pages pour camelot
            if isinstance(pages, list):
                pages = ','.join(map(str, pages))
            
            # Essayer d'abord avec la méthode lattice (meilleure pour tableaux avec bordures)
            tables_lattice = camelot.read_pdf(
                file_path,
                pages=pages,
                flavor='lattice',
                suppress_stdout=True
            )
            
            # Si aucun tableau ou tableaux de mauvaise qualité, essayer la méthode stream
            if len(tables_lattice) == 0 or all(table.parsing_report['accuracy'] < 80 for table in tables_lattice):
                tables_stream = camelot.read_pdf(
                    file_path,
                    pages=pages,
                    flavor='stream',
                    suppress_stdout=True
                )
                
                # Utiliser les résultats de la meilleure méthode
                if len(tables_stream) > len(tables_lattice) or (
                    len(tables_stream) > 0 and 
                    sum(t.parsing_report['accuracy'] for t in tables_stream) > 
                    sum(t.parsing_report['accuracy'] for t in tables_lattice)
                ):
                    tables = tables_stream
                else:
                    tables = tables_lattice
            else:
                tables = tables_lattice
            
            # Conversion en dataframes pandas
            processed_tables = []
            for table in tables:
                if table.parsing_report['accuracy'] > 50:  # Seuil de qualité minimal
                    df = table.df
                    # Suppression des lignes et colonnes vides
                    df = df.replace('', np.nan).dropna(how='all', axis=0).dropna(how='all', axis=1)
                    
                    # Si le tableau n'est pas vide
                    if not df.empty and len(df.columns) > 0:
                        processed_tables.append(df)
            
            return processed_tables
            
        except Exception as e:
            logger.error(f"Erreur extraction camelot: {e}")
            return []
    
    def _extract_with_pdfplumber(self, file_path: str, pages: Union[str, List[int]]) -> List[pd.DataFrame]:
        """
        Extrait les tableaux avec pdfplumber.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Pages à analyser
            
        Returns:
            Liste de DataFrames pandas contenant les tableaux
        """
        try:
            logger.info(f"Extraction avec pdfplumber: {file_path}")
            
            with pdfplumber.open(file_path) as pdf:
                # Détermination des pages à analyser
                if pages == "all":
                    page_list = range(len(pdf.pages))
                else:
                    page_list = [p-1 for p in pages] if isinstance(pages, list) else []
                
                processed_tables = []
                
                for i in page_list:
                    if i < 0 or i >= len(pdf.pages):
                        continue
                        
                    page = pdf.pages[i]
                    tables = page.extract_tables()
                    
                    for table in tables:
                        if table and len(table) > 0 and len(table[0]) > 0:
                            # Conversion en DataFrame
                            df = pd.DataFrame(table)
                            
                            # Nettoyage
                            df = df.replace('', np.nan).dropna(how='all', axis=0).dropna(how='all', axis=1)
                            
                            # Si la première ligne semble être des en-têtes
                            if len(df) > 1 and self._is_header_row(df.iloc[0]):
                                df.columns = df.iloc[0]
                                df = df.iloc[1:].reset_index(drop=True)
                            
                            # Si le tableau n'est pas vide
                            if not df.empty and len(df.columns) > 0:
                                processed_tables.append(df)
                
                return processed_tables
                
        except Exception as e:
            logger.error(f"Erreur extraction pdfplumber: {e}")
            return []
    
    def _is_header_row(self, row) -> bool:
        """
        Détermine si une ligne est probablement une ligne d'en-tête.
        
        Args:
            row: Ligne à vérifier
            
        Returns:
            True si la ligne semble être un en-tête
        """
        try:
            # Vérifier les valeurs non nulles
            values = [str(v).strip() for v in row if not pd.isna(v) and str(v).strip()]
            
            if len(values) < 2:
                return False
            
            # Heuristiques pour détecter un en-tête:
            # 1. Les valeurs sont relativement courtes
            avg_len = sum(len(v) for v in values) / len(values)
            if avg_len > 30:  # En-têtes généralement courts
                return False
                
            # 2. Les valeurs commencent généralement par une majuscule
            capitals = sum(1 for v in values if v and v[0].isupper())
            if capitals / len(values) < 0.5:  # Moins de la moitié commence par une majuscule
                return False
                
            # 3. Peu de valeurs numériques
            numerics = sum(1 for v in values if v.replace('.', '', 1).replace(',', '', 1).isdigit())
            if numerics / len(values) > 0.5:  # Plus de la moitié sont des nombres
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _convert_output_format(
        self, 
        tables: List[pd.DataFrame], 
        output_format: str
    ) -> List[Dict[str, Any]]:
        """
        Convertit les tableaux au format de sortie désiré.
        
        Args:
            tables: Liste de DataFrames pandas
            output_format: Format de sortie désiré
            
        Returns:
            Liste de tables au format demandé
        """
        result = []
        
        for i, table in enumerate(tables):
            table_data = {"table_id": i+1, "rows": len(table), "columns": len(table.columns)}
            
            if output_format == "pandas":
                table_data["data"] = table
            elif output_format == "csv":
                table_data["data"] = table.to_csv(index=False)
            elif output_format == "json":
                # Gérer les types non sérialisables
                table_data["data"] = json.loads(
                    table.replace({np.nan: None}).to_json(orient="records")
                )
            elif output_format == "html":
                table_data["data"] = table.to_html(index=False, na_rep="")
            elif output_format == "excel":
                buffer = BytesIO()
                table.to_excel(buffer, index=False)
                buffer.seek(0)
                table_data["data"] = base64.b64encode(buffer.read()).decode("utf-8")
                table_data["format"] = "base64"
                buffer.close()
            else:
                table_data["data"] = table.to_dict(orient="records")
            
            result.append(table_data)
        
        return result
    
    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            # Nettoyer le répertoire temporaire
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
            
            # Fermer l'executor
            self.executor.shutdown(wait=False)
            
            logger.info("Ressources du TableExtractor nettoyées")
        except Exception as e:
            logger.error(f"Erreur nettoyage TableExtractor: {e}")