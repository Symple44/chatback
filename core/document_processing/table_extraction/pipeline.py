# core/document_processing/table_extraction/pipeline.py
from typing import List, Dict, Any, Optional, Union, BinaryIO, Tuple
from enum import Enum
import asyncio
import tempfile
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import fitz
import io
import base64
import hashlib
import time

from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings
from .strategies import (
    PDFTableStrategy,
    CamelotTableStrategy, 
    TabulaTableStrategy, 
    PDFPlumberTableStrategy, 
    OCRTableStrategy,
    AIDetectionTableStrategy,
    HybridTableStrategy
)
from .models import (
    ExtractionResult,
    TableData,
    TableRegion,
    PDFType,
    TableExtractionContext,
    ProcessedTable
)
from .image_processing import PDFImageProcessor
from .validators import TableValidator
from .optimizers import TableOptimizer
from .utils import (
    estimate_pdf_complexity,
    convert_pdf_to_images,
    temp_file_manager,
    get_extraction_cache_key
)

logger = get_logger("table_extraction_pipeline")

class TableExtractionPipeline:
    """
    Pipeline modulaire et extensible pour l'extraction de tableaux de PDFs.
    
    Cette classe coordonne l'ensemble du processus d'extraction en utilisant:
    1. Une phase de détection du type de document
    2. Une sélection intelligente des stratégies d'extraction
    3. Un traitement parallèle des stratégies avec vote de confiance
    4. Une phase de post-traitement et validation
    5. Une phase d'optimisation des résultats
    """
    
    def __init__(self, cache_manager=None, table_detector=None):
        """
        Initialise le pipeline d'extraction.
        
        Args:
            cache_manager: Gestionnaire de cache optionnel
            table_detector: Détecteur de tableaux par IA
        """
        self.cache_manager = cache_manager
        self.table_detector = table_detector
        self.image_processor = PDFImageProcessor()
        self.validator = TableValidator()
        self.optimizer = TableOptimizer()
        
        # Configurations depuis settings
        self.default_strategy = settings.table_extraction.DEFAULT_STRATEGY.value
        self.cache_ttl = settings.table_extraction.CACHE_TTL
        self.max_tables = settings.table_extraction.MAX_TABLES
        
        # Initialisation des stratégies disponibles
        self.strategies = {
            "camelot": CamelotTableStrategy(),
            "tabula": TabulaTableStrategy(),
            "pdfplumber": PDFPlumberTableStrategy(),
            "ocr": OCRTableStrategy(self.image_processor),
            "ai": AIDetectionTableStrategy(table_detector) if table_detector and 
                        hasattr(settings, 'table_extraction') and 
                        hasattr(settings.table_extraction, 'AI_DETECTION') and 
                        getattr(getattr(settings.table_extraction, 'AI_DETECTION', None), 'ENABLED', False) else None,
            "hybrid": HybridTableStrategy(table_detector) if table_detector else None,
            "enhanced": EnhancedHybridStrategy(table_detector) if table_detector else None
        }
        
    async def extract_tables(
        self,
        file_obj: Union[str, Path, BinaryIO],
        pages: Union[str, List[int]] = "all",
        strategy: str = "auto",
        output_format: str = "pandas",
        ocr_config: Optional[Dict] = None,
        use_cache: bool = True
    ) -> ExtractionResult:
        """
        Extrait les tableaux d'un fichier PDF en utilisant la stratégie la plus appropriée.
        
        Args:
            file_obj: Chemin du fichier PDF ou objet BytesIO
            pages: Pages à analyser ("all" ou liste de numéros de page)
            strategy: Stratégie d'extraction ("auto", "camelot", "tabula", "pdfplumber", "ocr", "ai", "hybrid")
            output_format: Format de sortie des données ("pandas", "json", "csv", "html")
            ocr_config: Configuration pour l'OCR
            use_cache: Utiliser le cache si disponible
            
        Returns:
            Résultat de l'extraction contenant les tableaux, métadonnées et statistiques
        """
        try:
            with metrics.timer("table_extraction_total"):
                # Préparation et validation du fichier source
                file_path, is_temp = await self._prepare_file(file_obj)
                
                try:
                    # Vérification du cache si activé
                    if use_cache and self.cache_manager:
                        cache_key = await get_extraction_cache_key(file_path, pages, strategy, output_format, ocr_config)
                        cached_result = await self.cache_manager.get(cache_key)
                        if cached_result:
                            logger.info(f"Résultats d'extraction trouvés en cache")
                            metrics.track_cache_operation(hit=True)
                            return ExtractionResult.from_json(cached_result)
                        metrics.track_cache_operation(hit=False)
                    
                    # Création du contexte d'extraction
                    context = await self._create_extraction_context(
                        file_path=file_path,
                        pages=pages,
                        output_format=output_format,
                        ocr_config=ocr_config
                    )
                    
                    # Sélection de la stratégie appropriée
                    if strategy == "auto":
                        # Détection automatique de la meilleure stratégie
                        strategy = await self._detect_best_strategy(context)
                        logger.info(f"Stratégie auto-détectée: {strategy}")
                    
                    # Exécution de la stratégie sélectionnée
                    extraction_strategy = self.strategies.get(strategy)
                    if not extraction_strategy:
                        logger.warning(f"Stratégie {strategy} non disponible, utilisation de tabula")
                        extraction_strategy = self.strategies["tabula"]
                    
                    # Extraction des tableaux avec la stratégie choisie
                    tables = await extraction_strategy.extract_tables(context)
                    
                    # Si aucun tableau trouvé, essayer une stratégie de secours
                    if not tables and strategy != "ocr":
                        logger.info(f"Aucun tableau trouvé avec {strategy}, tentative avec OCR")
                        tables = await self.strategies["ocr"].extract_tables(context)
                    
                    # Validation et optimisation des tableaux extraits
                    tables = await self.validator.validate_tables(tables, context)
                    tables = await self.optimizer.optimize_tables(tables, context)
                    
                    # Conversion au format souhaité
                    result_tables = await self._convert_to_output_format(tables, output_format)
                    
                    # Construction du résultat
                    extraction_time = metrics.get_timer_value("table_extraction_total")
                    result = ExtractionResult(
                        extraction_id=context.extraction_id,
                        filename=context.filename,
                        file_size=context.file_size,
                        tables_count=len(result_tables),
                        processing_time=extraction_time,
                        tables=result_tables,
                        extraction_method_used=strategy,
                        pdf_type=context.pdf_type.value,
                        ocr_used=(strategy == "ocr" or context.pdf_type == PDFType.SCANNED),
                        status="completed" if result_tables else "no_tables_found",
                        message="Extraction réussie" if result_tables else "Aucun tableau trouvé"
                    )
                    
                    # Mise en cache des résultats si activé
                    if use_cache and self.cache_manager and result_tables:
                        await self.cache_manager.set(cache_key, result.to_json(), 3600)
                    
                    return result
                    
                finally:
                    # Nettoyage du fichier temporaire si nécessaire
                    if is_temp and os.path.exists(file_path):
                        os.unlink(file_path)
        
        except Exception as e:
            logger.error(f"Erreur extraction tableaux: {e}", exc_info=True)
            metrics.increment_counter("table_extraction_errors")
            return ExtractionResult(
                extraction_id=str(time.time()),
                filename=getattr(file_obj, "filename", str(file_obj)) if hasattr(file_obj, "filename") else "unknown",
                file_size=0,
                tables_count=0,
                processing_time=0,
                tables=[],
                extraction_method_used=strategy,
                pdf_type="unknown",
                ocr_used=False,
                status="error",
                message=f"Erreur lors de l'extraction: {str(e)}"
            )
                
    async def _prepare_file(self, file_obj: Union[str, Path, BinaryIO]) -> Tuple[str, bool]:
        """
        Prépare le fichier pour l'extraction.
        
        Args:
            file_obj: Fichier à traiter (path ou BytesIO)
            
        Returns:
            Tuple contenant le chemin du fichier et un booléen indiquant s'il est temporaire
        """
        if isinstance(file_obj, (str, Path)):
            return str(file_obj), False
        else:
            # Création d'un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                # Position au début du fichier si c'est un BytesIO
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                
                # Lecture du contenu
                if hasattr(file_obj, 'read'):
                    content = file_obj.read()
                    if isinstance(content, bytes):
                        temp_file.write(content)
                    else:
                        temp_file.write(content.encode('utf-8'))
                else:
                    # Cas où file_obj est déjà le contenu
                    temp_file.write(file_obj)
                
                return temp_file.name, True
        
    async def _create_extraction_context(
        self,
        file_path: str,
        pages: Union[str, List[int]],
        output_format: str,
        ocr_config: Optional[Dict] = None
    ) -> TableExtractionContext:
        """
        Crée un contexte d'extraction avec toutes les informations nécessaires.
        
        Args:
            file_path: Chemin du fichier PDF
            pages: Pages à analyser
            output_format: Format de sortie
            ocr_config: Configuration OCR
            
        Returns:
            Contexte d'extraction
        """
        # Identification du type de PDF (scanné ou numérique)
        pdf_type, confidence = await self._identify_pdf_type(file_path)
        logger.info(f"Type de PDF détecté: {pdf_type.value} (confiance: {confidence:.2f})")
        
        # Analyse de la complexité du document
        complexity_score = await estimate_pdf_complexity(file_path)
        logger.info(f"Score de complexité du document: {complexity_score:.2f}")
        
        # Création du contexte
        return TableExtractionContext(
            file_path=file_path,
            filename=os.path.basename(file_path),
            file_size=os.path.getsize(file_path),
            pages=pages,
            output_format=output_format,
            ocr_config=ocr_config or {},
            pdf_type=pdf_type,
            pdf_type_confidence=confidence,
            extraction_id=hashlib.md5(f"{file_path}:{time.time()}".encode()).hexdigest(),
            complexity_score=complexity_score
        )
    
    async def _identify_pdf_type(self, file_path: str) -> Tuple[PDFType, float]:
        """
        Identifie si un PDF est scanné ou numérique par plusieurs heuristiques.
        
        Cette méthode utilise une approche multi-facteurs pour déterminer le type de PDF:
        1. Densité de texte sélectionnable
        2. Présence de fontes
        3. Taille du fichier par page
        4. Rapport texte/images
        
        Args:
            file_path: Chemin du fichier PDF
            
        Returns:
            Tuple contenant le type de PDF et un score de confiance (0-1)
        """
        try:
            # Analyse plus détaillée avec méthodes multiples
            with fitz.open(file_path) as doc:
                page_count = len(doc)
                if page_count == 0:
                    return PDFType.DIGITAL, 0.5
                
                # Analyser un échantillon de pages plus représentatif
                sample_size = min(max(3, page_count // 5), 10)  # Entre 3 et 10 pages
                # Choisir des pages du début, du milieu et de la fin pour un meilleur échantillonnage
                indices = [0]  # Première page
                if page_count > 1:
                    indices.append(page_count - 1)  # Dernière page
                # Pages du milieu
                for i in range(1, sample_size - 1):
                    idx = (i * page_count) // sample_size
                    if idx not in indices:
                        indices.append(idx)
                
                # Scores pour chaque dimension d'analyse
                text_scores = []
                font_scores = []
                image_scores = []
                line_counts = []
                
                for page_idx in indices:
                    page = doc[page_idx]
                    text = page.get_text()
                    fonts = page.get_fonts()
                    images = page.get_images()
                    drawings = page.get_drawings()
                    
                    # Analyse plus précise du texte
                    words = page.get_text("words")
                    chars = len("".join([w[4] for w in words]))
                    blocks = page.get_text("blocks")
                    
                    # Score basé sur la densité et qualité du texte
                    text_score = min(1.0, len(words) / 500) if len(words) > 0 else 0
                    if chars < 50:
                        text_score = 0.9  # Très peu de texte = probablement scanné
                    
                    # L'OCR produit souvent du texte avec fautes
                    if text and len(words) > 20:
                        from spellchecker import SpellChecker
                        try:
                            spell = SpellChecker()
                            word_samples = [w[4].lower() for w in words[:50] if len(w[4]) > 3]
                            misspelled = spell.unknown(word_samples)
                            misspelled_ratio = len(misspelled) / len(word_samples) if word_samples else 0
                            if misspelled_ratio > 0.4:  # Beaucoup de fautes = probablement OCR
                                text_score = min(0.8, text_score + 0.3)
                        except:
                            pass
                    
                    # Détection de la qualité des caractères
                    if fonts:
                        # Les PDFs scannés utilisent généralement moins de polices
                        font_score = 0.2 if len(fonts) > 3 else 0.7
                    else:
                        font_score = 0.9  # Pas de fonts = probablement scanné
                    
                    # Détection de la présence d'images
                    image_score = 0.8 if images else 0.2
                    
                    # Densité des lignes (plus élevée dans les PDFs scannés)
                    line_count = len([d for d in drawings if d["type"] == "l"])
                    line_ratio = min(1.0, line_count / 100)
                    
                    # Ajouter aux scores globaux
                    text_scores.append(text_score)
                    font_scores.append(font_score)
                    image_scores.append(image_score)
                    line_counts.append(line_ratio)
                
                # Moyenne des scores
                avg_text_score = sum(text_scores) / len(text_scores) if text_scores else 0.5
                avg_font_score = sum(font_scores) / len(font_scores) if font_scores else 0.5
                avg_image_score = sum(image_scores) / len(image_scores) if image_scores else 0.5
                avg_line_score = sum(line_counts) / len(line_counts) if line_counts else 0.5
                
                # Calcul de score avec poids optimisés
                weights = {
                    "text": 0.5,    # Texte sélectionnable est le meilleur indicateur
                    "font": 0.25,   # Police de caractères
                    "image": 0.15,  # Présence d'images
                    "line": 0.1     # Présence de lignes
                }
                
                combined_score = (
                    avg_text_score * weights["text"] +
                    avg_font_score * weights["font"] +
                    avg_image_score * weights["image"] +
                    avg_line_score * weights["line"]
                )
                
                # Classification avec hysteresis pour éviter les changements brusques
                if combined_score > 0.65:
                    return PDFType.SCANNED, combined_score
                elif combined_score < 0.35:
                    return PDFType.DIGITAL, 1.0 - combined_score
                else:
                    # Zone de doute - analyser plus en détail ou retourner HYBRID
                    return PDFType.HYBRID, 0.5
                    
        except Exception as e:
            logger.error(f"Erreur identification type PDF: {e}")
            return PDFType.DIGITAL, 0.5
    
    async def _detect_best_strategy(self, context: TableExtractionContext) -> str:
        """
        Détermine la meilleure stratégie d'extraction en fonction du contexte.
        
        Args:
            context: Contexte d'extraction
            
        Returns:
            Nom de la stratégie recommandée
        """
        # Si PDF scanné détecté avec haute confiance, utiliser OCR ou IA
        if context.pdf_type == PDFType.SCANNED and context.pdf_type_confidence > 0.8:
            # Vérifier si la stratégie améliorée est disponible
            if "enhanced" in self.strategies and self.strategies["enhanced"]:
                return "enhanced"
            return "ocr"
        
        # Pour les PDFs avec tableaux complexes, utiliser la méthode améliorée
        if context.complexity_score > 0.6:
            # Utiliser la stratégie améliorée si disponible
            if "enhanced" in self.strategies and self.strategies["enhanced"]:
                return "enhanced"
            return "camelot"
        
        # Pour les PDFs simples, tabula est plus rapide et généralement efficace
        if context.complexity_score < 0.3:
            return "tabula"
        
        # Utilisation de la détection IA pour les cas intermédiaires
        if self.table_detector is not None:
            # Utiliser la stratégie améliorée pour les cas intermédiaires
            if "enhanced" in self.strategies and self.strategies["enhanced"]:
                return "enhanced"
            return "ai"
        
        # Si l'IA n'est pas disponible, utiliser hybrid ou enhanced
        if "enhanced" in self.strategies and self.strategies["enhanced"]:
            return "enhanced"
        return "hybrid" if "hybrid" in self.strategies else "camelot"
    
    async def _convert_to_output_format(
        self, 
        tables: List[ProcessedTable],
        output_format: str
    ) -> List[Dict[str, Any]]:
        """
        Convertit les tables dans le format de sortie souhaité.
        
        Args:
            tables: Liste des tables traitées
            output_format: Format de sortie souhaité
            
        Returns:
            Liste de tables au format demandé
        """
        result = []
        for i, table in enumerate(tables):
            output_table = {
                "table_id": i + 1,
                "page": table.page,
                "rows": table.rows,
                "columns": table.columns,
                "extraction_method": table.method,
                "confidence": table.confidence
            }
            
            # Conversion selon le format demandé
            if output_format == "pandas":
                output_table["data"] = table.data
            elif output_format == "json":
                if hasattr(table.data, "to_dict"):
                    output_table["data"] = table.data.to_dict(orient="records")
                else:
                    output_table["data"] = table.data
            elif output_format == "csv":
                if hasattr(table.data, "to_csv"):
                    output_table["data"] = table.data.to_csv(index=False)
                else:
                    output_table["data"] = str(table.data)
            elif output_format == "html":
                if hasattr(table.data, "to_html"):
                    output_table["data"] = table.data.to_html(index=False)
                else:
                    output_table["data"] = str(table.data)
            else:
                output_table["data"] = table.data
                
            result.append(output_table)
            
        return result

    async def get_tables_as_images(
        self,
        file_obj: Union[str, BinaryIO],
        pages: Union[str, List[int]] = "all"
    ) -> List[Dict[str, Any]]:
        """
        Obtient les images des tableaux détectés.
        
        Args:
            file_obj: Fichier PDF à analyser
            pages: Pages à analyser
            
        Returns:
            Liste de dictionnaires contenant les images des tableaux
        """
        try:
            # Préparation du fichier
            file_path, is_temp = await self._prepare_file(file_obj)
            
            try:
                # Création du contexte minimal
                context = await self._create_extraction_context(
                    file_path=file_path,
                    pages=pages,
                    output_format="pandas",
                    ocr_config={}
                )
                
                # Conversion du PDF en images
                page_images = await convert_pdf_to_images(file_path, context.pages)
                
                result = []
                for page_idx, img in enumerate(page_images):
                    # Détection des régions de tableaux dans l'image
                    table_regions = []
                    
                    # Utiliser le détecteur IA si disponible
                    if self.table_detector:
                        detections = await self.table_detector.detect_tables(img)
                        if detections:
                            table_regions = [TableRegion(
                                x=d["x"], 
                                y=d["y"], 
                                width=d["width"], 
                                height=d["height"],
                                confidence=d["score"]
                            ) for d in detections]
                    
                    # Si pas de régions détectées ou pas de détecteur IA, utiliser heuristiques basiques
                    if not table_regions:
                        # Détection simple basée sur les lignes et bordures
                        import cv2
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY_INV, 11, 2)
                        
                        # Détection des lignes horizontales et verticales
                        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
                        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
                        
                        # Détecter les lignes horizontales et verticales
                        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
                        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
                        
                        # Combiner les lignes
                        table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
                        
                        # Dilater pour mieux connecter les lignes
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                        table_mask = cv2.dilate(table_mask, kernel, iterations=3)
                        
                        # Trouver les contours
                        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Filtrer les petits contours
                        min_area = 0.01 * img.shape[0] * img.shape[1]  # Au moins 1% de l'image
                        for cnt in contours:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if w * h > min_area:
                                table_regions.append(TableRegion(
                                    x=x, y=y, width=w, height=h, confidence=0.7
                                ))
                    
                    # Si toujours pas de régions, considérer toute l'image
                    if not table_regions:
                        table_regions = [TableRegion(
                            x=0, y=0, width=img.shape[1], height=img.shape[0], confidence=0.5
                        )]
                    
                    # Récupérer l'image de chaque région
                    for i, region in enumerate(table_regions):
                        # Extraire la région d'intérêt
                        table_img = img[region.y:region.y+region.height, region.x:region.x+region.width]
                        
                        # Convertir en base64
                        is_success, buffer = cv2.imencode(".png", table_img)
                        if is_success:
                            img_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Ajouter l'image au résultat
                            result.append({
                                "table_id": i + 1,
                                "page": context.get_page_number(page_idx),
                                "data": img_base64,
                                "mime_type": "image/png",
                                "width": region.width,
                                "height": region.height,
                                "position": {"x": region.x, "y": region.y, "width": region.width, "height": region.height},
                                "confidence": region.confidence
                            })
                            
                return result
                
            finally:
                # Nettoyage du fichier temporaire
                if is_temp and os.path.exists(file_path):
                    os.unlink(file_path)
                    
        except Exception as e:
            logger.error(f"Erreur récupération images tableaux: {e}")
            return []
            
    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            # Nettoyage des stratégies
            for strategy in self.strategies.values():
                if strategy and hasattr(strategy, 'cleanup'):
                    await strategy.cleanup()
                    
            # Nettoyage des autres ressources
            if hasattr(self.image_processor, 'cleanup'):
                self.image_processor.cleanup()
                
            # Nettoyage du détecteur IA si disponible
            if self.table_detector and hasattr(self.table_detector, 'cleanup'):
                await self.table_detector.cleanup()
                
        except Exception as e:
            logger.error(f"Erreur nettoyage pipeline: {e}")