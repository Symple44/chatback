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
from PIL import Image
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
    """Classe améliorée pour extraire des tableaux de fichiers PDF."""
    
    def __init__(self, cache_enabled: bool = True, table_detector: Optional[Any] = None):
        """
        Initialisation de l'extracteur de tableaux.
        
        Args:
            cache_enabled: Active ou désactive le cache
            table_detector: Instance du détecteur de tableaux IA (optionnel)
        """
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
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
    
    async def extract_tables_auto(
        self,
        file_source: Union[str, Path, BytesIO],
        output_format: str = "pandas",
        max_pages: Optional[int] = None,
        ocr_config: Optional[Dict[str, Any]] = None,
        save_to: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extrait automatiquement les tableaux d'un PDF avec détection intelligente.
        
        Cette méthode simplifiée détecte automatiquement le type de PDF (texte ou image) 
        et utilise la meilleure stratégie d'extraction sans configuration manuelle.
        """
        with metrics.timer("table_extraction_auto"):
            # Préparation du fichier source
            file_path = None
            file_content = None
            is_temp_file = False
            
            try:
                # Gestion des différents types d'entrée
                if isinstance(file_source, BytesIO):
                    file_content = file_source.getvalue()
                    async with temp_file_manager(file_content) as tmp_path:
                        file_path = tmp_path
                        is_temp_file = True
                        return await self._process_auto_extraction(
                            file_path, file_content, output_format, max_pages, ocr_config, save_to
                        )
                elif isinstance(file_source, (str, Path)):
                    file_path = str(file_source)
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    return await self._process_auto_extraction(
                        file_path, file_content, output_format, max_pages, ocr_config, save_to
                    )
                else:
                    raise ValueError("Source de fichier non prise en charge")
            
            except Exception as e:
                logger.error(f"Erreur extraction automatique: {e}", exc_info=True)
                metrics.increment_counter("table_extraction_auto_errors")
                raise

    async def _process_auto_extraction(
        self,
        file_path: str,
        file_content: bytes,
        output_format: str,
        max_pages: Optional[int],
        ocr_config: Optional[Dict[str, Any]],
        save_to: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Processus principal d'extraction automatique."""
        # Vérifier le cache si activé
        if self.cache_enabled:
            cache_key = self._generate_cache_key(file_content, output_format, max_pages, ocr_config)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Résultats trouvés en cache pour {file_path}")
                metrics.track_cache_operation(hit=True)
                return cached_result
            metrics.track_cache_operation(hit=False)
        
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
        extraction_method = await self._select_extraction_method(file_path, pages, is_scanned, confidence)
        
        # 3. Configurer les paramètres optimaux pour la méthode choisie
        ocr_config_final = self._prepare_ocr_config(ocr_config, extraction_method == "ocr")
        
        # 4. Extraire les tableaux avec la méthode choisie
        tables = await self.extract_tables(
            file_path,
            pages=pages,
            extraction_method=extraction_method,
            output_format=output_format,
            ocr_config=ocr_config_final
        )
        
        # 5. Si aucun tableau trouvé, essayer des approches alternatives
        if not tables:
            tables = await self._try_alternative_extraction_methods(
                file_path, pages, extraction_method, output_format, ocr_config_final
            )
        
        # 6. Sauvegarder les résultats si demandé
        if save_to and tables:
            self._save_tables_to_files(tables, save_to, output_format)
        
        # Ajouter au cache si activé
        if self.cache_enabled and tables:
            cache_key = self._generate_cache_key(file_content, output_format, max_pages, ocr_config)
            self.cache[cache_key] = tables
        
        return tables
    
    def _generate_cache_key(self, file_content: bytes, output_format: str, max_pages: Optional[int], ocr_config: Optional[Dict]) -> str:
        """Génère une clé de cache unique basée sur le contenu et les paramètres."""
        # Hacher le contenu du fichier
        file_hash = hashlib.md5(file_content).hexdigest()
        
        # Créer une représentation des paramètres
        params = f"{output_format}_{max_pages}_{str(ocr_config)}"
        params_hash = hashlib.md5(params.encode()).hexdigest()
        
        return f"{file_hash}_{params_hash}"
    
    async def _select_extraction_method(
        self, 
        file_path: str, 
        pages: str, 
        is_scanned: bool, 
        confidence: float
    ) -> str:
        """
        Sélectionne la méthode d'extraction optimale.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Pages à analyser
            is_scanned: Si le PDF est détecté comme scanné
            confidence: Niveau de confiance de la détection
            
        Returns:
            Nom de la méthode à utiliser
        """
        # Si c'est un PDF scanné avec haute confiance, utiliser directement OCR
        if is_scanned and confidence > 0.8:
            logger.info("Méthode choisie: OCR (document scanné avec haute confiance)")
            return "ocr"
        
        # Si c'est un PDF potentiellement scanné avec confiance moyenne, tester les méthodes traditionnelles
        if is_scanned and confidence > 0.5:
            # Tester rapidement si les méthodes traditionnelles trouvent quelque chose
            loop = asyncio.get_event_loop()
            
            # Test avec Camelot (limité à quelques pages)
            test_pages = pages
            if pages == "all" or "-" in pages:
                test_pages = "1-3"  # Limiter à 3 pages pour le test
                
            camelot_result = await loop.run_in_executor(
                self.executor, 
                functools.partial(self._test_camelot, file_path, test_pages)
            )
            
            if camelot_result.get("tables", 0) > 0 and camelot_result.get("score", 0) > 0.7:
                logger.info(f"Méthode choisie: camelot (tables trouvées malgré un PDF potentiellement scanné)")
                return "camelot"
                
            # Si Camelot ne trouve rien, utiliser OCR
            logger.info("Méthode choisie: OCR (document potentiellement scanné, aucune table trouvée avec méthodes traditionnelles)")
            return "ocr"
        
        # Pour les PDFs textuels, tester différentes méthodes et sélectionner la meilleure
        loop = asyncio.get_event_loop()
        
        # Test avec Camelot
        camelot_result = await loop.run_in_executor(
            self.executor, 
            functools.partial(self._test_camelot, file_path, "1-3" if pages == "all" else pages)
        )
        
        # Test avec Tabula
        tabula_result = await loop.run_in_executor(
            self.executor,
            functools.partial(self._test_tabula, file_path, "1-3" if pages == "all" else pages)
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
        
        logger.info(f"Scores des méthodes traditionnelles: {scores}")
        
        # Si aucune méthode n'est satisfaisante, essayer l'IA ou OCR
        if best_score < 1.0:
            # Vérifier si détection IA disponible
            if hasattr(self, 'table_detector'):
                logger.info("Méthode choisie: AI (aucune méthode traditionnelle efficace)")
                return "ai"
            else:
                logger.info("Méthode choisie: hybrid (aucune méthode traditionnelle efficace)")
                return "hybrid"
        else:
            logger.info(f"Méthode choisie: {best_method} (score: {best_score:.2f})")
            return best_method
    
    def _prepare_ocr_config(self, user_config: Optional[Dict[str, Any]], is_ocr_method: bool) -> Dict[str, Any]:
        """
        Prépare la configuration OCR optimale.
        
        Args:
            user_config: Configuration fournie par l'utilisateur
            is_ocr_method: Si la méthode d'extraction est OCR
            
        Returns:
            Configuration OCR optimisée
        """
        # Configuration par défaut
        default_config = {
            "lang": self.tesseract_lang,
            "enhance_image": True,
            "deskew": True,
            "preprocess_type": "thresh",
            "psm": 6,
            "force_grid": False
        }
        
        # Si la méthode est OCR, utiliser des paramètres optimisés
        if is_ocr_method:
            default_config["enhance_image"] = True
            default_config["deskew"] = True
            default_config["psm"] = 6  # Mode 6 est généralement meilleur pour les tableaux
        
        # Fusionner avec la configuration utilisateur si fournie
        if user_config:
            default_config.update(user_config)
            
        return default_config
    
    async def _try_alternative_extraction_methods(
        self,
        file_path: str,
        pages: str,
        primary_method: str,
        output_format: str,
        ocr_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Essaie des méthodes d'extraction alternatives si la première a échoué.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Pages à analyser
            primary_method: Méthode d'extraction principale
            output_format: Format de sortie
            ocr_config: Configuration OCR
            
        Returns:
            Liste des tableaux extraits
        """
        if primary_method == "ocr":
            # Essayer avec force_grid=True
            logger.info("Aucun tableau trouvé, essai avec force_grid=True")
            modified_config = ocr_config.copy()
            modified_config["force_grid"] = True
            
            tables = await self.extract_tables(
                file_path,
                pages=pages,
                extraction_method="ocr",
                output_format=output_format,
                ocr_config=modified_config
            )
            
            if tables:
                logger.info(f"OCR avec force_grid: {len(tables)} tableaux trouvés")
                return tables
        
        # Essayer la méthode hybride
        if hasattr(self, 'table_detector') and primary_method != "hybrid":
            logger.info("Essai avec méthode hybride")
            tables = await self.extract_tables(
                file_path,
                pages=pages,
                extraction_method="hybrid",
                output_format=output_format,
                ocr_config=ocr_config
            )
            
            if tables:
                logger.info(f"Méthode hybride: {len(tables)} tableaux trouvés")
                return tables
        
        # Si tout échoue et que la méthode principale n'était pas tabula, essayer tabula
        if primary_method not in ["tabula", "hybrid"]:
            logger.info("Dernier essai avec tabula")
            tables = await self.extract_tables(
                file_path,
                pages=pages,
                extraction_method="tabula",
                output_format=output_format
            )
            
            if tables:
                logger.info(f"Méthode tabula: {len(tables)} tableaux trouvés")
                return tables
        
        logger.warning("Toutes les méthodes d'extraction ont échoué, aucun tableau trouvé")
        return []
    
    def _save_tables_to_files(self, tables: List[Dict[str, Any]], save_dir: str, output_format: str):
        """
        Sauvegarde les tableaux extraits dans des fichiers.
        
        Args:
            tables: Liste des tableaux extraits
            save_dir: Répertoire de sauvegarde
            output_format: Format de sortie
        """
        try:
            # Créer le répertoire de sortie
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Format de sauvegarde
            if output_format == "pandas":
                save_format = "csv"  # Par défaut CSV pour pandas
            else:
                save_format = output_format
            
            # Sauvegarde de chaque tableau
            for i, table in enumerate(tables):
                table_data = table.get("data")
                if table_data is None:
                    continue
                
                # Nom du fichier
                filename = f"table_{i+1}.{save_format}"
                output_path = save_path / filename
                
                # Conversion en DataFrame si nécessaire
                if not isinstance(table_data, pd.DataFrame) and hasattr(table_data, '__iter__'):
                    table_data = pd.DataFrame(table_data)
                
                # Sauvegarde selon le format
                if save_format == "csv" and isinstance(table_data, pd.DataFrame):
                    table_data.to_csv(output_path, index=False)
                elif save_format == "json" and isinstance(table_data, pd.DataFrame):
                    table_data.to_json(output_path, orient="records", indent=2)
                elif save_format == "excel" and isinstance(table_data, pd.DataFrame):
                    table_data.to_excel(output_path, index=False)
                elif save_format == "html" and isinstance(table_data, pd.DataFrame):
                    table_data.to_html(output_path, index=False)
                elif isinstance(table_data, (str, bytes)):
                    # Si c'est déjà une chaîne formatée
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(str(table_data))
            
            logger.info(f"Tableaux sauvegardés dans {save_dir}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde des tableaux: {e}")

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
                    
                    # Conversion vers le format de sortie désiré avec gestion d'erreur améliorée
                    try:
                        return await self._convert_output_format(tables, output_format)
                    except Exception as e:
                        logger.error(f"Erreur conversion format: {e}")
                        # Tenter une conversion de secours
                        return await self._fallback_format_conversion(tables, output_format)
                        
                finally:
                    # Nettoyage du fichier temporaire si nécessaire
                    if is_temp_file and os.path.exists(file_source):
                        try:
                            os.unlink(file_source)
                        except Exception as e:
                            logger.error(f"Erreur suppression fichier temporaire: {e}")
                
        except Exception as e:
            logger.error(f"Erreur extraction tableaux: {e}")
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
        
        Cette version améliorée utilise une approche plus robuste pour la détection.
        
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
            # La valeur seuil peut être ajustée en fonction des résultats
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
                    "rows": len(table),
                    "columns": len(table.columns)
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
    
    async def _extract_with_ocr(
        self, 
        file_path: str, 
        pages: Union[str, List[int]], 
        ocr_config: Dict[str, Any]
    ) -> List[pd.DataFrame]:
        """
        Extrait les tableaux à partir d'un PDF scanné en utilisant l'OCR.
        
        Version améliorée avec meilleure gestion des images et optimisation du traitement.
        
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
            
            # Convertir le PDF en images avec mise en cache
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
            logger.error(f"Erreur extraction OCR: {e}")
            return []  # Retourner une liste vide en cas d'erreur
    
    async def _convert_pdf_to_images(self, file_path: str, page_list: List[int]) -> List[np.ndarray]:
        """
        Convertit les pages d'un PDF en images avec mise en cache.
        
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

    async def _detect_best_method(self, file_path: str, pages: Union[str, List[int]]) -> str:
        """
        Version améliorée pour déterminer la meilleure méthode d'extraction.
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