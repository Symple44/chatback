# core/document_processing/table_extractor.py
from typing import List, Dict, Any, Optional, Union, Tuple, BinaryIO
from dataclasses import dataclass
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
import os
import tempfile
import uuid
from pathlib import Path
from datetime import datetime

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
    
    async def extract_tables_auto(
        self,
        file_source: Union[str, Path, BytesIO],
        output_format: str = "pandas",
        max_pages: Optional[int] = None,
        save_to: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extrait automatiquement les tableaux d'un PDF avec détection intelligente.
        
        Cette méthode simplifiée détecte automatiquement le type de PDF (texte ou image) 
        et utilise la meilleure stratégie d'extraction sans configuration manuelle.
        
        Args:
            file_source: Chemin du fichier PDF ou objet BytesIO
            output_format: Format de sortie ('pandas', 'csv', 'json', 'html', 'excel')
            max_pages: Nombre maximum de pages à traiter (None pour toutes)
            save_to: Chemin où sauvegarder les résultats (None pour ne pas sauvegarder)
            
        Returns:
            Liste des tableaux extraits avec leurs métadonnées
        """
        try:
            with metrics.timer("table_extraction_auto"):
                # Préparation du fichier source
                temp_file = None
                file_path = None
                
                try:
                    # Gestion des différents types d'entrée
                    if isinstance(file_source, BytesIO):
                        temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                        temp_file.write(file_source.getvalue())
                        temp_file.close()
                        file_path = temp_file.name
                    elif isinstance(file_source, (str, Path)):
                        file_path = str(file_source)
                    else:
                        raise ValueError("Source de fichier non prise en charge")
                    
                    # Déterminer les pages à analyser
                    pages = "all"
                    if max_pages:
                        # Compter le nombre total de pages
                        with pdfplumber.open(file_path) as pdf:
                            total_pages = len(pdf.pages)
                        # Limiter aux max_pages premières pages
                        pages = f"1-{min(max_pages, total_pages)}"
                    
                    # 1. Détecter automatiquement si le PDF est scanné
                    is_scanned, confidence = await self._is_scanned_pdf(file_path, pages)
                    logger.info(f"Détection automatique: PDF scanné = {is_scanned} (confiance: {confidence:.2f})")
                    
                    # 2. Choisir la stratégie d'extraction optimale
                    extraction_method = None
                    
                    if is_scanned:
                        # Pour les PDFs scannés, utiliser directement l'OCR
                        extraction_method = "ocr"
                        logger.info("Méthode choisie: OCR (document scanné)")
                    else:
                        # Pour les PDFs textuels, tester différentes méthodes et sélectionner la meilleure
                        loop = asyncio.get_event_loop()
                        
                        # Test avec Camelot
                        camelot_result = await loop.run_in_executor(
                            self.executor, 
                            lambda: self._test_camelot(file_path, pages)
                        )
                        
                        # Test avec Tabula
                        tabula_result = await loop.run_in_executor(
                            self.executor,
                            lambda: self._test_tabula(file_path, pages)
                        )
                        
                        # Test avec Pdfplumber pour les cas particuliers
                        pdfplumber_score = 0
                        try:
                            with pdfplumber.open(file_path) as pdf:
                                test_pages = pdf.pages[:min(3, len(pdf.pages))]
                                tables_count = sum(1 for p in test_pages if p.extract_tables())
                                pdfplumber_score = tables_count * 5  # Score arbitraire
                        except Exception as e:
                            logger.debug(f"Test pdfplumber échoué: {e}")
                        
                        # Décider de la meilleure méthode
                        scores = {
                            "camelot": camelot_result.get("score", 0),
                            "tabula": tabula_result.get("score", 0),
                            "pdfplumber": pdfplumber_score
                        }
                        
                        best_method = max(scores.items(), key=lambda x: x[1])[0]
                        best_score = scores[best_method]
                        
                        logger.info(f"Scores des méthodes: {scores}")
                        
                        # Si aucune méthode n'est satisfaisante, essayer l'AI ou OCR
                        if best_score < 1.0:
                            # Vérifier si détection IA disponible
                            if hasattr(self, 'table_detector') and self.table_detector:
                                extraction_method = "ai"
                                logger.info("Méthode choisie: AI (aucune méthode traditionnelle efficace)")
                            else:
                                extraction_method = "ocr"
                                logger.info("Méthode choisie: OCR (aucune méthode traditionnelle efficace)")
                        else:
                            extraction_method = best_method
                            logger.info(f"Méthode choisie: {best_method} (score: {best_score:.2f})")
                    
                    # 3. Configurer les paramètres optimaux pour la méthode choisie
                    ocr_config = None
                    if extraction_method == "ocr":
                        ocr_config = {
                            "lang": self.tesseract_lang,
                            "enhance_image": True,
                            "deskew": True,
                            "preprocess_type": "thresh",
                            "force_grid": False  # Essayer d'abord sans force_grid
                        }
                    
                    # 4. Extraire les tableaux avec la méthode choisie
                    tables = await self.extract_tables(
                        file_path,
                        pages=pages,
                        extraction_method=extraction_method,
                        output_format=output_format,
                        ocr_config=ocr_config
                    )
                    
                    # 5. Si aucun tableau trouvé, essayer des approches alternatives
                    if not tables:
                        logger.info("Aucun tableau trouvé, essai de méthodes alternatives")
                        
                        # Si méthode précédente était OCR, essayer avec force_grid=True
                        if extraction_method == "ocr":
                            ocr_config["force_grid"] = True
                            tables = await self.extract_tables(
                                file_path,
                                pages=pages,
                                extraction_method="ocr",
                                output_format=output_format,
                                ocr_config=ocr_config
                            )
                            logger.info(f"OCR avec force_grid: {len(tables)} tableaux trouvés")
                        
                        # Si toujours rien, essayer une approche hybride
                        if not tables and hasattr(self, 'table_detector') and self.table_detector:
                            tables = await self.extract_tables(
                                file_path,
                                pages=pages,
                                extraction_method="hybrid",
                                output_format=output_format
                            )
                            logger.info(f"Méthode hybride: {len(tables)} tableaux trouvés")
                    
                    # 6. Sauvegarder les résultats si demandé
                    if save_to and tables:
                        # Déterminer le format de sauvegarde
                        if output_format == "pandas":
                            save_format = "csv"  # Par défaut sauvegarde en CSV si output_format est pandas
                        else:
                            save_format = output_format
                        
                        # Créer le répertoire de sortie si nécessaire
                        save_dir = Path(save_to)
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Nom de fichier de base
                        base_filename = Path(file_path).stem
                        
                        # Sauvegarder chaque tableau
                        for i, table in enumerate(tables):
                            table_data = table.get("data")
                            if table_data is not None:
                                # Le chemin du fichier de sortie
                                output_path = save_dir / f"{base_filename}_table_{i+1}.{save_format}"
                                
                                if save_format == "csv":
                                    if isinstance(table_data, pd.DataFrame):
                                        table_data.to_csv(output_path, index=False)
                                    elif isinstance(table_data, str) and save_format == "csv":
                                        with open(output_path, "w", encoding="utf-8") as f:
                                            f.write(table_data)
                                elif save_format == "json":
                                    if isinstance(table_data, pd.DataFrame):
                                        table_data.to_json(output_path, orient="records", indent=2)
                                    else:
                                        with open(output_path, "w", encoding="utf-8") as f:
                                            json.dump(table_data, f, indent=2)
                                elif save_format == "excel":
                                    if isinstance(table_data, pd.DataFrame):
                                        table_data.to_excel(output_path, index=False)
                                elif save_format == "html":
                                    if isinstance(table_data, pd.DataFrame):
                                        table_data.to_html(output_path, index=False)
                                    else:
                                        with open(output_path, "w", encoding="utf-8") as f:
                                            f.write(table_data)
                        
                        logger.info(f"Tableaux sauvegardés dans {save_dir}")
                    
                    return tables
                    
                finally:
                    # Nettoyage du fichier temporaire si nécessaire
                    if temp_file and file_path and os.path.exists(file_path):
                        try:
                            os.unlink(file_path)
                        except Exception as e:
                            logger.error(f"Erreur suppression fichier temporaire: {e}")
        
        except Exception as e:
            logger.error(f"Erreur extraction automatique des tableaux: {e}", exc_info=True)
            metrics.increment_counter("table_extraction_auto_errors")
            raise

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
            extraction_method: Méthode d'extraction ("auto", "tabula", "camelot", "pdfplumber", "ocr", "ai", "hybrid")
            output_format: Format de sortie ("pandas", "csv", "json", "html", "excel")
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
                    tables = await self._extract_with_ocr(file_path, pages, ocr_config)
                elif extraction_method == "ai" and hasattr(self, 'table_detector'):
                    # Utiliser la détection par IA si disponible
                    tables = await self._extract_with_ai(file_path, pages, ocr_config)
                elif extraction_method == "hybrid" and hasattr(self, 'table_detector'):
                    # Méthode hybride combinant l'IA et les méthodes traditionnelles
                    tables = await self._extract_with_hybrid(file_path, pages, ocr_config)
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
                    # Si méthode inconnue ou si l'IA est demandée mais non disponible
                    if extraction_method in ["ai", "hybrid"]:
                        logger.warning(f"Méthode {extraction_method} non disponible, utilisation de tabula")
                        tables = await loop.run_in_executor(
                            self.executor, 
                            lambda: self._extract_with_tabula(file_path, pages)
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
    
    async def _extract_with_ai(self, file_path: str, pages: Union[str, List[int]], ocr_config: Dict[str, Any]) -> List[pd.DataFrame]:
        """
        Extrait les tableaux en utilisant la détection par IA.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Pages à analyser
            ocr_config: Configuration OCR (utilisée si nécessaire)
            
        Returns:
            Liste de DataFrames pandas contenant les tableaux
        """
        try:
            logger.info(f"Extraction avec détection IA: {file_path}")
            
            # Convertir le PDF en images
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
            
            # Traiter chaque image
            for img_idx, img in enumerate(images):
                page_num = image_to_page.get(img_idx, img_idx + 1)
                
                # Convertir l'image PIL en format numpy
                img_np = np.array(img)
                
                # Utiliser l'IA pour détecter les tableaux
                detections = await self.table_detector.detect_tables(img_np)
                
                logger.info(f"Page {page_num}: {len(detections)} tableaux détectés par IA")
                
                # Pour chaque tableau détecté
                for i, detection in enumerate(detections):
                    try:
                        # Extraire les coordonnées
                        x, y, w, h = detection["x"], detection["y"], detection["width"], detection["height"]
                        
                        # Extraire la région d'intérêt
                        table_img = img_np[y:y+h, x:x+w]
                        
                        # Prétraiter l'image pour l'OCR
                        processed_img = await self._preprocess_image(
                            table_img, 
                            ocr_config.get("enhance_image", True),
                            ocr_config.get("deskew", True),
                            ocr_config.get("preprocess_type", "thresh")
                        )
                        
                        # Appliquer l'OCR pour extraire le tableau
                        df = await self._ocr_table_to_dataframe(
                            processed_img,
                            lang=ocr_config.get("lang", self.tesseract_lang),
                            psm=ocr_config.get("psm", 6),
                            force_grid=ocr_config.get("force_grid", False)
                        )
                        
                        # Ajouter à la liste si le DataFrame n'est pas vide
                        if df is not None and not df.empty and len(df.columns) > 1:
                            # Ajouter des colonnes d'information
                            df = df.copy()
                            df.insert(0, "_page", page_num)
                            df.insert(1, "_table", i + 1)
                            df.insert(2, "_confidence", detection["score"])
                            tables_list.append(df)
                            
                    except Exception as e:
                        logger.error(f"Erreur extraction tableau détecté par IA {i+1} sur page {page_num}: {e}")
            
            logger.info(f"Extraction IA terminée: {len(tables_list)} tableaux extraits")
            return tables_list
            
        except Exception as e:
            logger.error(f"Erreur extraction avec IA: {e}")
            return []
    
    async def _extract_with_hybrid(self, file_path: str, pages: Union[str, List[int]], ocr_config: Dict[str, Any]) -> List[pd.DataFrame]:
        """
        Méthode hybride combinant l'IA et les approches traditionnelles.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Pages à analyser
            ocr_config: Configuration OCR
            
        Returns:
            Liste de DataFrames pandas contenant les tableaux
        """
        try:
            logger.info(f"Extraction hybride: {file_path}")
            
            # D'abord tenter avec des méthodes traditionnelles
            loop = asyncio.get_event_loop()
            traditional_tables = await loop.run_in_executor(
                self.executor, 
                lambda: self._extract_with_tabula(file_path, pages)
            )
            
            if traditional_tables and len(traditional_tables) > 0:
                logger.info(f"Méthode traditionnelle réussie, {len(traditional_tables)} tableaux trouvés")
                return traditional_tables
            
            # Si échec, tenter avec l'IA
            ai_tables = await self._extract_with_ai(file_path, pages, ocr_config)
            
            if ai_tables and len(ai_tables) > 0:
                logger.info(f"Méthode IA réussie, {len(ai_tables)} tableaux trouvés")
                return ai_tables
            
            # Si toujours rien, tenter l'OCR
            ocr_tables = await self._extract_with_ocr(file_path, pages, ocr_config)
            
            if ocr_tables and len(ocr_tables) > 0:
                logger.info(f"Méthode OCR réussie, {len(ocr_tables)} tableaux trouvés")
                return ocr_tables
            
            # Si toutes les méthodes ont échoué
            logger.warning("Toutes les méthodes d'extraction ont échoué")
            return []
            
        except Exception as e:
            logger.error(f"Erreur extraction hybride: {e}")
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
        Méthode alternative de détection des tableaux basée sur le contenu structuré.
        
        Args:
            img: Image prétraitée
            
        Returns:
            Liste de coordonnées (x, y, w, h) des régions de tableau
        """
        # Dilater l'image pour connecter les caractères proches
        kernel = np.ones((5, 20), np.uint8)
        dilated = cv2.dilate(img, kernel, iterations=2)
        
        # Trouver les contours des blocs de texte
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer les contours trop petits
        min_area = 0.0005 * img.shape[0] * img.shape[1]  # 0.05% de l'image
        text_blocks = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                text_blocks.append((x, y, w, h))
        
        # Si pas assez de blocs de texte, considérer une grande partie de l'image
        if len(text_blocks) < 3:
            height, width = img.shape[:2]
            margin = width // 10
            return [(margin, margin, width - 2*margin, height - 2*margin)]
        
        # Trier les blocs par position verticale (y)
        text_blocks.sort(key=lambda b: b[1])
        
        # Grouper les blocs en lignes (possibles lignes d'un tableau)
        rows = []
        current_row = [text_blocks[0]]
        
        for block in text_blocks[1:]:
            _, prev_y, _, prev_h = current_row[-1]
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
            max_x = min(img.shape[1], max_x + margin)
            max_y = min(img.shape[0], max_y + margin)
            
            return [(min_x, min_y, max_x - min_x, max_y - min_y)]
        
        # Si tout échoue, retourner un rectangle qui couvre la majorité de l'image
        height, width = img.shape[:2]
        margin = width // 10
        return [(margin, margin, width - 2*margin, height - 2*margin)]
    
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
            
            # Sauvegarder l'image temporairement
            temp_img_path = os.path.join(self.temp_dir, f"table_{uuid.uuid4()}.png")
            cv2.imwrite(temp_img_path, img)
            
            # Configuration OCR
            custom_config = f'--oem 3 --psm {psm} -l {lang}'
            
            # Inverser l'image pour l'OCR (texte noir sur fond blanc)
            pil_img = Image.open(temp_img_path)
            
            # Utiliser TSV pour capturer la structure
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
                return None
            
            # Regrouper par ligne
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
            
            if not valid_lines:
                return None
                
            # Pour chaque ligne, trier les mots par position
            for line_num in valid_lines:
                grouped_by_line[line_num] = sorted(grouped_by_line[line_num], key=lambda w: w['left'])
            
            # Créer une grille pour le tableau
            if force_grid:
                # Analyser toutes les positions de mots pour déduire la structure de colonnes
                all_positions = []
                for line_num in valid_lines:
                    for word in grouped_by_line[line_num]:
                        all_positions.append(word['center_x'])
                
                # Regrouper les positions proches en colonnes (clustering simple)
                all_positions.sort()
                column_centers = []
                if all_positions:
                    current_center = all_positions[0]
                    current_count = 1
                    for pos in all_positions[1:]:
                        if abs(pos - current_center) < 20:  # 20 pixels de tolérance
                            current_center = (current_center * current_count + pos) / (current_count + 1)
                            current_count += 1
                        else:
                            column_centers.append(current_center)
                            current_center = pos
                            current_count = 1
                    column_centers.append(current_center)
                
                # Créer une grille de tableau basée sur ces colonnes
                n_columns = len(column_centers)
                if n_columns < 2:
                    n_columns = 2  # Au moins 2 colonnes
                
                table_data = []
                for ln in valid_lines:
                    row_data = [''] * n_columns
                    
                    for word in grouped_by_line[ln]:
                        # Déterminer à quelle colonne appartient ce mot
                        if not column_centers:
                            break
                            
                        distances = [abs(word['center_x'] - center) for center in column_centers]
                        col_idx = distances.index(min(distances))
                        
                        # Ajouter le texte à la colonne appropriée
                        if col_idx < len(row_data):
                            if row_data[col_idx]:
                                row_data[col_idx] += ' ' + word['text'].strip()
                            else:
                                row_data[col_idx] = word['text'].strip()
                    
                    table_data.append(row_data)
            else:
                # Version simplifiée : une ligne = une ligne du tableau
                table_data = []
                
                for ln in valid_lines:
                    row = [word['text'] for word in grouped_by_line[ln]]
                    if row:
                        table_data.append(row)
            
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_img_path):
                os.unlink(temp_img_path)
                
            # Créer le DataFrame
            if not table_data:
                return None
                
            df = pd.DataFrame(table_data)
            
            # Utiliser la première ligne comme en-tête si approprié
            if len(df) > 1 and self._is_header_row(df.iloc[0]):
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
            
            # Nettoyer le DataFrame
            df = df.apply(lambda x: x.str.strip() if x.dtype == object else x)
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur OCR tableau: {e}")
            return None
    
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
    
    async def get_tables_as_images(
        self, 
        file_path: Union[str, BytesIO], 
        pages: Union[str, List[int]] = "all"
    ) -> List[Dict[str, Any]]:
        """
        Extrait les images des tableaux d'un PDF.
        
        Args:
            file_path: Chemin du fichier PDF ou BytesIO
            pages: Pages à analyser
            
        Returns:
            Liste d'images encodées en base64
        """
        try:
            # Préparation du fichier
            temp_file = None
            if isinstance(file_path, BytesIO):
                temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                temp_file.write(file_path.getvalue())
                temp_file.close()
                file_path = temp_file.name
            
            # Convertir en liste de pages
            if pages == "all":
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    page_list = list(range(1, len(pdf.pages) + 1))
            elif isinstance(pages, str):
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
            images = pdf2image.convert_from_path(
                file_path,
                dpi=150,
                first_page=min(page_list),
                last_page=max(page_list)
            )
            
            # Créer un dictionnaire pour associer l'index de l'image à la page
            image_to_page = {}
            for i, page_num in enumerate(page_list):
                if i < len(images):
                    image_to_page[i] = page_num
            
            # Résultat
            result = []
            
            # Pour chaque image
            for img_idx, img in enumerate(images):
                img_np = np.array(img)
                
                # Détecter les tableaux
                table_regions = await self._detect_table_regions(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY))
                
                # Pour chaque tableau détecté
                for i, region in enumerate(table_regions):
                    x, y, w, h = region
                    
                    # Extraire la région du tableau
                    table_img = img_np[y:y+h, x:x+w]
                    
                    # Convertir en format PIL
                    pil_img = Image.fromarray(table_img)
                    
                    # Convertir en base64
                    buffer = BytesIO()
                    pil_img.save(buffer, format="PNG")
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    
                    # Ajouter au résultat
                    result.append({
                        "page": image_to_page.get(img_idx, img_idx + 1),
                        "table_index": i,
                        "data": img_str,
                        "width": w,
                        "height": h,
                        "position": {"x": x, "y": y}
                    })
            
            # Nettoyage du fichier temporaire
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur extraction images tableaux: {e}")
            return []
    
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
            
            # Si le détecteur IA est utilisé, le nettoyer
            if hasattr(self, 'table_detector'):
                await self.table_detector.cleanup()
            
            logger.info("Ressources du TableExtractor nettoyées")
        except Exception as e:
            logger.error(f"Erreur nettoyage TableExtractor: {e}")