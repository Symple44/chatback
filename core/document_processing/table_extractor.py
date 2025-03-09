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
                if extraction_method == "ocr":
                    # Pour les PDF scannés, on utilise l'OCR
                    tables = await self._extract_with_ocr(file_path, pages, ocr_config)
                    metrics.increment_counter("ocr_table_extractions")
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
            tables_list = []
            
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
                            psm=ocr_config.get("psm", 6)
                        )
                        
                        # Ajouter à la liste si le DataFrame n'est pas vide
                        if df is not None and not df.empty and len(df.columns) > 1:
                            df.insert(0, "_page", page_num)
                            df.insert(1, "_table", i + 1)
                            tables_list.append(df)
                            
                    except Exception as e:
                        logger.error(f"Erreur extraction tableau {i+1} sur page {page_num}: {e}")
            
            logger.info(f"OCR terminé: {len(tables_list)} tableaux extraits")
            return tables_list
            
        except Exception as e:
            logger.error(f"Erreur extraction OCR: {e}")
            return []
    
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
        Méthode alternative de détection des tableaux basée sur les blocs de texte.
        
        Args:
            img: Image prétraitée
            
        Returns:
            Liste de coordonnées (x, y, w, h) des régions de tableau
        """
        # Appliquer un filtre de dilatation pour connecter les caractères en lignes
        kernel = np.ones((5, 20), np.uint8)
        dilated = cv2.dilate(img, kernel, iterations=1)
        
        # Trouver les contours des blocs de texte
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer les contours trop petits
        min_area = 0.001 * img.shape[0] * img.shape[1]  # 0.1% de l'image
        text_blocks = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]
        
        # S'il y a moins de 2 blocs, considérer toute l'image comme un tableau
        if len(text_blocks) < 2:
            return [(0, 0, img.shape[1], img.shape[0])]
        
        # Regrouper les blocs qui sont alignés horizontalement (potentielles lignes d'un tableau)
        text_blocks.sort(key=lambda b: b[1])  # Trier par coordonnée y
        
        # Détecter les lignes de texte en recherchant les blocs alignés horizontalement
        y_tolerance = 20  # Tolérance en pixels pour considérer les blocs sur la même ligne
        lines = []
        current_line = [text_blocks[0]]
        
        for block in text_blocks[1:]:
            y = block[1]
            prev_y = current_line[-1][1]
            
            if abs(y - prev_y) <= y_tolerance:
                # Même ligne
                current_line.append(block)
            else:
                # Nouvelle ligne
                if len(current_line) >= 2:  # Au moins 2 blocs dans la ligne
                    lines.append(current_line)
                current_line = [block]
        
        # Ajouter la dernière ligne si elle contient au moins 2 blocs
        if len(current_line) >= 2:
            lines.append(current_line)
        
        # S'il y a au moins 2 lignes, on a probablement un tableau
        if len(lines) >= 2:
            # Déterminer les limites du tableau
            min_x = min([min([b[0] for b in line]) for line in lines])
            min_y = min([min([b[1] for b in line]) for line in lines])
            max_x = max([max([b[0] + b[2] for b in line]) for line in lines])
            max_y = max([max([b[1] + b[3] for b in line]) for line in lines])
            
            # Ajouter une marge
            margin = 20
            min_x = max(0, min_x - margin)
            min_y = max(0, min_y - margin)
            max_x = min(img.shape[1], max_x + margin)
            max_y = min(img.shape[0], max_y + margin)
            
            return [(min_x, min_y, max_x - min_x, max_y - min_y)]
        
        # Si aucun tableau n'est détecté, retourner vide
        return []
    
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
        psm: int = 6
    ) -> Optional[pd.DataFrame]:
        """
        Applique l'OCR à l'image d'un tableau et convertit le résultat en DataFrame.
        
        Args:
            img: Image du tableau
            lang: Langues pour Tesseract (fra+eng par défaut)
            psm: Mode de segmentation de page Tesseract
            
        Returns:
            DataFrame contenant les données du tableau
        """
        try:
            # Sauvegarder l'image temporairement
            temp_img_path = os.path.join(self.temp_dir, f"table_{uuid.uuid4()}.png")
            cv2.imwrite(temp_img_path, img)
            
            # Configuration OCR
            custom_config = f'--oem 3 --psm {psm} -l {lang}'
            
            # Utiliser le mode TSV de Tesseract pour obtenir un tableau
            loop = asyncio.get_event_loop()
            tsv_output = await loop.run_in_executor(
                self.executor,
                lambda: pytesseract.image_to_data(
                    Image.open(temp_img_path),
                    config=custom_config,
                    output_type=pytesseract.Output.DATAFRAME
                )
            )
            
            # Nettoyer
            if os.path.exists(temp_img_path):
                os.unlink(temp_img_path)
            
            # Filtrer les résultats pour n'inclure que le texte reconnu
            tsv_output = tsv_output[tsv_output['conf'] > 0]
            
            # Si aucun texte n'est reconnu, retourner None
            if tsv_output.empty:
                return None
            
            # Traiter les résultats pour reconstruire le tableau
            # Regrouper par ligne (déterminer les lignes du tableau)
            tsv_output['line_num'] = tsv_output['top'] // 10  # Regrouper les éléments à peu près à la même hauteur
            
            # Reconstruire les lignes et colonnes
            table_data = []
            for line_num, line_group in tsv_output.groupby('line_num'):
                # Trier les éléments de gauche à droite
                line_group = line_group.sort_values('left')
                
                # Créer une liste pour cette ligne
                row_data = []
                for _, word_data in line_group.iterrows():
                    if str(word_data['text']).strip():  # Ignorer les valeurs vides
                        row_data.append(str(word_data['text']))
                
                if row_data:  # Si la ligne contient du texte
                    table_data.append(row_data)
            
            # Si aucune donnée n'est extraite, retourner None
            if not table_data:
                return None
            
            # Déterminer le nombre maximum de colonnes
            max_cols = max(len(row) for row in table_data)
            
            # Normaliser les lignes pour qu'elles aient toutes le même nombre de colonnes
            normalized_data = []
            for row in table_data:
                # Étendre la ligne avec des valeurs vides si nécessaire
                normalized_row = row + [''] * (max_cols - len(row))
                normalized_data.append(normalized_row)
            
            # Créer un DataFrame
            df = pd.DataFrame(normalized_data)
            
            # Utiliser la première ligne comme en-tête si elle semble être un en-tête
            if len(df) > 1 and self._is_header_row(df.iloc[0]):
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
            
            # Nettoyer les données
            df = df.apply(lambda x: x.str.strip() if x.dtype == object else x)
            
            return df
            
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