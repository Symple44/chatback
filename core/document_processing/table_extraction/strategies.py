# core/document_processing/table_extraction/strategies.py
from typing import List, Dict, Any, Optional, Union, BinaryIO
from abc import ABC, abstractmethod
import torch
import asyncio
import pandas as pd
import numpy as np
import camelot
import tabula
import pdfplumber
from concurrent.futures import ThreadPoolExecutor
import os
import io
import cv2
from PIL import Image
import pytesseract
import fitz
import time

from core.config.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .models import (
    TableExtractionContext,
    ProcessedTable,
    TableRegion,
    PDFType
)
from .image_processing import PDFImageProcessor
from .utils import convert_pdf_to_images, parse_page_range

logger = get_logger("table_extraction_strategies")

class PDFTableStrategy(ABC):
    """Classe de base pour toutes les stratégies d'extraction de tableaux de PDFs."""
    
    def __init__(self):
        """Initialise la stratégie."""
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        
    @abstractmethod
    async def extract_tables(self, context: TableExtractionContext) -> List[ProcessedTable]:
        """
        Extrait les tableaux selon la stratégie spécifique.
        
        Args:
            context: Contexte d'extraction
            
        Returns:
            Liste de tableaux traités
        """
        pass
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie et optimise un DataFrame.
        
        Args:
            df: DataFrame à traiter
            
        Returns:
            DataFrame nettoyé
        """
        if df is None or df.empty:
            return df
            
        # Suppression des lignes et colonnes vides
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Nettoyage des valeurs
        df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
        
        # Ajout de types de données appropriés où c'est possible
        for col in df.columns:
            # Tentative de conversion numérique
            if pd.to_numeric(df[col], errors='coerce').notna().all():
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    
    async def cleanup(self):
        """Nettoie les ressources."""
        self.executor.shutdown(wait=False)

class CamelotTableStrategy(PDFTableStrategy):
    """Stratégie d'extraction utilisant Camelot."""
    
    async def extract_tables(self, context: TableExtractionContext) -> List[ProcessedTable]:
        """
        Extrait les tableaux avec Camelot.
        
        Args:
            context: Contexte d'extraction
            
        Returns:
            Liste de tableaux traités
        """
        try:
            with metrics.timer("camelot_extraction"):
                # Conversion des pages
                if isinstance(context.pages, list):
                    page_str = ",".join([str(p+1) for p in context.pages])  # Camelot utilise des indices 1-based
                else:
                    page_str = context.pages
                
                # Limiter les pages pour le test
                if page_str == "all":
                    with fitz.open(context.file_path) as pdf:
                        max_pages = min(len(pdf), 20)  # Limite à 20 pages max
                    page_str = f"1-{max_pages}"
                
                # Extraction avec les deux méthodes de Camelot
                tables = []
                
                loop = asyncio.get_event_loop()
                
                # Méthode lattice (pour tableaux avec bordures)
                try:
                    lattice_tables = await loop.run_in_executor(
                        self.executor,
                        lambda: camelot.read_pdf(
                            context.file_path, 
                            pages=page_str,
                            flavor="lattice",
                            suppress_stdout=True,
                            line_scale=40,  # Paramètre amélioré pour mieux détecter les lignes
                            copy_text=['v', 'h']  # Meilleure récupération du texte
                        )
                    )
                    
                    for i, table in enumerate(lattice_tables):
                        # Vérifier la qualité
                        if table.parsing_report['accuracy'] > 50:  # 50% d'accuracy minimum
                            df = table.df
                            df = self._process_dataframe(df)
                            
                            if not df.empty:
                                tables.append(ProcessedTable(
                                    data=df,
                                    page=table.page,
                                    rows=len(df),
                                    columns=len(df.columns),
                                    method="camelot-lattice",
                                    confidence=table.parsing_report['accuracy'] / 100.0,
                                    region=None  # Pas d'info de région avec Camelot
                                ))
                except Exception as e:
                    logger.warning(f"Erreur extraction Camelot lattice: {e}")
                
                # Méthode stream (pour tableaux sans bordures)
                try:
                    stream_tables = await loop.run_in_executor(
                        self.executor,
                        lambda: camelot.read_pdf(
                            context.file_path, 
                            pages=page_str,
                            flavor="stream",
                            suppress_stdout=True,
                            edge_tol=100,  # Plus tolérant pour les alignements
                            row_tol=10     # Meilleure détection des lignes
                        )
                    )
                    
                    for i, table in enumerate(stream_tables):
                        # Vérifier la qualité
                        if table.parsing_report['accuracy'] > 30:  # Stream est généralement moins précis
                            df = table.df
                            df = self._process_dataframe(df)
                            
                            if not df.empty:
                                # Vérifier si cette table est un doublon d'une table déjà extraite
                                is_duplicate = False
                                for existing_table in tables:
                                    if existing_table.page == table.page and existing_table.rows == len(df) and existing_table.columns == len(df.columns):
                                        # Comparer le contenu
                                        similarity = self._calculate_df_similarity(existing_table.data, df)
                                        if similarity > 0.7:  # 70% de similarité
                                            is_duplicate = True
                                            break
                                
                                if not is_duplicate:
                                    tables.append(ProcessedTable(
                                        data=df,
                                        page=table.page,
                                        rows=len(df),
                                        columns=len(df.columns),
                                        method="camelot-stream",
                                        confidence=table.parsing_report['accuracy'] / 100.0,
                                        region=None
                                    ))
                except Exception as e:
                    logger.warning(f"Erreur extraction Camelot stream: {e}")
                
                logger.info(f"Camelot: {len(tables)} tableaux extraits")
                return tables
                
        except Exception as e:
            logger.error(f"Erreur extraction Camelot: {e}")
            return []
    
    def _calculate_df_similarity(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """Calcule la similarité entre deux DataFrames."""
        if df1.shape != df2.shape:
            return 0.0
            
        df1_str = df1.astype(str)
        df2_str = df2.astype(str)
        
        # Nombre de cellules identiques
        identical_cells = (df1_str == df2_str).sum().sum()
        total_cells = df1.size
        
        return identical_cells / total_cells if total_cells > 0 else 0.0

class TabulaTableStrategy(PDFTableStrategy):
    """Stratégie d'extraction utilisant Tabula."""
    
    async def extract_tables(self, context: TableExtractionContext) -> List[ProcessedTable]:
        """
        Extrait les tableaux avec Tabula.
        
        Args:
            context: Contexte d'extraction
            
        Returns:
            Liste de tableaux traités
        """
        try:
            with metrics.timer("tabula_extraction"):
                # Conversion des pages au format attendu par tabula
                page_list = await parse_page_range(context.file_path, context.pages)
                if not page_list:
                    return []
                
                # Convertir en 1-based pour Tabula
                page_list = [p + 1 for p in page_list]
                
                # Extraction avec tabula
                loop = asyncio.get_event_loop()
                tables_raw = await loop.run_in_executor(
                    self.executor,
                    lambda: tabula.read_pdf(
                        context.file_path,
                        pages=page_list,
                        multiple_tables=True,
                        guess=True,
                        stream=True,
                        lattice=True,  # Essayer les deux méthodes
                        area=[0, 0, 100, 100]  # Toute la page
                    )
                )
                
                # Traitement des résultats
                tables = []
                for i, df in enumerate(tables_raw):
                    if df is not None and not df.empty:
                        # Associer la page
                        page_num = page_list[min(i // 2, len(page_list) - 1)]
                        
                        # Nettoyage du tableau
                        df = self._process_dataframe(df)
                        
                        if not df.empty:
                            tables.append(ProcessedTable(
                                data=df,
                                page=page_num,
                                rows=len(df),
                                columns=len(df.columns),
                                method="tabula",
                                confidence=0.7,  # Tabula ne fournit pas de score de confiance
                                region=None
                            ))
                
                logger.info(f"Tabula: {len(tables)} tableaux extraits")
                return tables
                
        except Exception as e:
            logger.error(f"Erreur extraction Tabula: {e}")
            return []

class PDFPlumberTableStrategy(PDFTableStrategy):
    """Stratégie d'extraction utilisant PDFPlumber."""
    
    async def extract_tables(self, context: TableExtractionContext) -> List[ProcessedTable]:
        """
        Extrait les tableaux avec PDFPlumber.
        
        Args:
            context: Contexte d'extraction
            
        Returns:
            Liste de tableaux traités
        """
        try:
            with metrics.timer("pdfplumber_extraction"):
                # Conversion des pages
                page_indices = await parse_page_range(context.file_path, context.pages)
                if not page_indices:
                    return []
                
                tables = []
                loop = asyncio.get_event_loop()
                
                # Personnalisation des paramètres d'extraction
                table_settings = {
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "intersection_tolerance": 5,
                    "snap_tolerance": 10,
                    "edge_min_length": 5,
                    "min_words_vertical": 2,
                    "min_words_horizontal": 1
                }
                
                # Extraction avec PDFPlumber
                with pdfplumber.open(context.file_path) as pdf:
                    for page_idx in page_indices:
                        if page_idx >= len(pdf.pages):
                            continue
                            
                        page = pdf.pages[page_idx]
                        
                        # Extraction des tableaux
                        extracted_tables = await loop.run_in_executor(
                            self.executor,
                            lambda: page.extract_tables(table_settings)
                        )
                        
                        for i, table_data in enumerate(extracted_tables):
                            if not table_data or len(table_data) < 2:  # Au moins une ligne d'en-tête et une ligne de données
                                continue
                                
                            # Conversion en DataFrame
                            try:
                                # Déterminer si la première ligne est un en-tête
                                is_header = self._is_valid_header(table_data[0], table_data[1:])
                                
                                if is_header:
                                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                                else:
                                    columns = [f"Col_{j+1}" for j in range(len(table_data[0]))]
                                    df = pd.DataFrame(table_data, columns=columns)
                                
                                # Nettoyage
                                df = self._process_dataframe(df)
                                
                                if not df.empty:
                                    tables.append(ProcessedTable(
                                        data=df,
                                        page=page_idx + 1,  # 1-based
                                        rows=len(df),
                                        columns=len(df.columns),
                                        method="pdfplumber",
                                        confidence=0.6,  # Score fixe
                                        region=None
                                    ))
                            except Exception as table_error:
                                logger.warning(f"Erreur conversion tableau PDFPlumber: {table_error}")
                
                logger.info(f"PDFPlumber: {len(tables)} tableaux extraits")
                return tables
                
        except Exception as e:
            logger.error(f"Erreur extraction PDFPlumber: {e}")
            return []
    
    def _is_valid_header(self, header_row: List, data_rows: List[List]) -> bool:
        """Vérifie si une ligne peut être considérée comme un en-tête."""
        if not data_rows:
            return False
            
        # Vérifier si l'en-tête contient des chaînes numériques
        header_numeric = sum(1 for cell in header_row if cell.replace('.', '').isdigit() if isinstance(cell, str))
        
        # Échantillon des lignes de données pour vérifier si elles sont numériques
        sample_size = min(5, len(data_rows))
        sample_rows = data_rows[:sample_size]
        
        numeric_counts = []
        for row in sample_rows:
            numeric_count = sum(1 for cell in row if isinstance(cell, str) and cell.replace('.', '').isdigit())
            numeric_counts.append(numeric_count)
        
        # Calculer le pourcentage moyen de cellules numériques dans les données
        avg_numeric_percent = sum(numeric_counts) / (sample_size * len(header_row)) if sample_size > 0 else 0
        
        # Si l'en-tête contient significativement moins de numériques que les données, c'est probablement un en-tête
        header_numeric_percent = header_numeric / len(header_row)
        
        return header_numeric_percent < avg_numeric_percent * 0.5
    
    async def _ocr_table_grid_approach(
        self, 
        table_img: np.ndarray, 
        lang: str = "fra+eng", 
        psm: int = 6
    ) -> Optional[pd.DataFrame]:
        """
        Utilise une approche grille pour OCR.
        
        Args:
            table_img: Image du tableau
            lang: Langue pour l'OCR
            psm: Mode de segmentation Tesseract
            
        Returns:
            DataFrame ou None si échec
        """
        try:
            # Détecter les lignes de la grille
            if len(table_img.shape) > 2 and table_img.shape[2] == 3:
                gray = cv2.cvtColor(table_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = table_img
            
            # Binarisation adaptative
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Détection des lignes
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combiner les lignes
            grid = cv2.bitwise_or(horizontal_lines, vertical_lines)
            
            # Inversion pour l'affichage (lignes en noir)
            grid_inverted = 255 - grid
            
            # Trouver les intersections (croisements des lignes)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
            
            h_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_DILATE, h_kernel)
            v_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_DILATE, v_kernel)
            
            # Points d'intersection = points où lignes H et V se rencontrent
            intersections = cv2.bitwise_and(h_lines, v_lines)
            
            # Trouver les coordonnées des intersections
            intersection_points = np.where(intersections > 0)
            y_coords = intersection_points[0]
            x_coords = intersection_points[1]
            
            # Extraire les coordonnées uniques (approximativement)
            tolerance = 5  # En pixels
            unique_x = []
            unique_y = []
            
            # Clustering des coordonnées x
            x_sorted = sorted(x_coords)
            if x_sorted:
                current_cluster = [x_sorted[0]]
                for x in x_sorted[1:]:
                    if x - current_cluster[-1] <= tolerance:
                        current_cluster.append(x)
                    else:
                        unique_x.append(int(sum(current_cluster) / len(current_cluster)))
                        current_cluster = [x]
                if current_cluster:
                    unique_x.append(int(sum(current_cluster) / len(current_cluster)))
            
            # Clustering des coordonnées y
            y_sorted = sorted(y_coords)
            if y_sorted:
                current_cluster = [y_sorted[0]]
                for y in y_sorted[1:]:
                    if y - current_cluster[-1] <= tolerance:
                        current_cluster.append(y)
                    else:
                        unique_y.append(int(sum(current_cluster) / len(current_cluster)))
                        current_cluster = [y]
                if current_cluster:
                    unique_y.append(int(sum(current_cluster) / len(current_cluster)))
            
            # Si pas assez de lignes/colonnes, retourner None
            if len(unique_x) < 2 or len(unique_y) < 2:
                return None
            
            # Trier les coordonnées
            unique_x.sort()
            unique_y.sort()
            
            # Initialiser un tableau vide
            num_rows = len(unique_y) - 1
            num_cols = len(unique_x) - 1
            
            if num_rows < 1 or num_cols < 1:
                return None
                
            table_data = [["" for _ in range(num_cols)] for _ in range(num_rows)]
            
            # Configuration OCR
            loop = asyncio.get_event_loop()
            custom_config = f'--oem 3 --psm 7 -l {lang}'  # PSM 7: traiter comme une ligne de texte
            
            # Pour chaque cellule
            for i in range(num_rows):
                for j in range(num_cols):
                    # Coordonnées de la cellule
                    x1, y1 = unique_x[j], unique_y[i]
                    x2, y2 = unique_x[j+1], unique_y[i+1]
                    
                    # Marge pour éviter de prendre les bordures
                    margin = 3
                    cell_img = gray[y1+margin:y2-margin, x1+margin:x2-margin]
                    
                    if cell_img.size == 0:
                        continue
                    
                    # OCR sur la cellule
                    try:
                        cell_text = await loop.run_in_executor(
                            self.executor,
                            lambda: pytesseract.image_to_string(
                                cell_img, config=custom_config
                            ).strip()
                        )
                        table_data[i][j] = cell_text
                    except Exception as cell_error:
                        logger.warning(f"Erreur OCR cellule ({i},{j}): {cell_error}")
            
            # Création du DataFrame
            if table_data:
                # Vérifier si la première ligne est un en-tête
                if self._is_valid_header(table_data[0], table_data[1:]):
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                else:
                    columns = [f'Col_{j+1}' for j in range(num_cols)]
                    df = pd.DataFrame(table_data, columns=columns)
                
                return self._process_dataframe(df)
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur approche grille: {e}")
            return None

class AIDetectionTableStrategy(PDFTableStrategy):
    """Stratégie d'extraction utilisant la détection IA des tableaux."""
    
    def __init__(self, table_detector=None):
        """
        Initialise la stratégie IA.
        
        Args:
            table_detector: Détecteur de tableaux par IA
        """
        super().__init__()
        self.table_detector = table_detector
        self.ocr_strategy = OCRTableStrategy()
    
    async def extract_tables(self, context: TableExtractionContext) -> List[ProcessedTable]:
        """
        Extrait les tableaux avec détection IA.
        
        Args:
            context: Contexte d'extraction
            
        Returns:
            Liste de tableaux traités
        """
        try:
            with metrics.timer("ai_detection_extraction"):
                if not self.table_detector:
                    logger.warning("Détecteur IA non disponible, utilisant OCR")
                    return await self.ocr_strategy.extract_tables(context)
                
                # Conversion du PDF en images
                page_indices = await parse_page_range(context.file_path, context.pages)
                if not page_indices:
                    return []
                
                page_images = await convert_pdf_to_images(context.file_path, page_indices)
                
                # Extraction des tableaux avec IA
                tables = []
                for idx, img in enumerate(page_images):
                    # Numéro de page réel (1-based)
                    page_num = page_indices[idx] + 1 if idx < len(page_indices) else idx + 1
                    
                    # Détecter les régions de tableaux avec IA
                    detections = await self.table_detector.detect_tables(img)
                    
                    if detections:
                        # Création de régions à partir des détections
                        regions = [TableRegion(
                            x=d["x"],
                            y=d["y"],
                            width=d["width"],
                            height=d["height"],
                            confidence=d["score"]
                        ) for d in detections]
                        
                        # Pour chaque région détectée
                        for i, region in enumerate(regions):
                            try:
                                # Découper la région de l'image
                                table_img = img[region.y:region.y+region.height, region.x:region.x+region.width]
                                
                                # 1. Essai avec méthode OCR améliorée
                                df = await self.ocr_strategy._ocr_table_to_dataframe(
                                    table_img,
                                    lang=context.ocr_config.get("lang", "fra+eng"),
                                    psm=context.ocr_config.get("psm", 6)
                                )
                                
                                # 2. Si échec, essai avec approche grille
                                if df is None or df.empty or len(df.columns) <= 1:
                                    df = await self.ocr_strategy._ocr_table_grid_approach(
                                        table_img,
                                        lang=context.ocr_config.get("lang", "fra+eng"),
                                        psm=context.ocr_config.get("psm", 6)
                                    )
                                
                                # Si un DataFrame valide a été créé
                                if df is not None and not df.empty and len(df.columns) > 1:
                                    tables.append(ProcessedTable(
                                        data=df,
                                        page=page_num,
                                        rows=len(df),
                                        columns=len(df.columns),
                                        method="ai-detection",
                                        confidence=region.confidence,
                                        region=region
                                    ))
                            except Exception as e:
                                logger.warning(f"Erreur extraction région {i}: {e}")
                
                logger.info(f"IA: {len(tables)} tableaux extraits")
                return tables
                
        except Exception as e:
            logger.error(f"Erreur extraction IA: {e}")
            return []
            
    async def cleanup(self):
        """Nettoie les ressources."""
        await super().cleanup()
        if self.ocr_strategy:
            await self.ocr_strategy.cleanup()

class HybridTableStrategy(PDFTableStrategy):
    """Stratégie d'extraction hybride combinant plusieurs approches."""
    
    def __init__(self, table_detector=None):
        """
        Initialise la stratégie hybride.
        
        Args:
            table_detector: Détecteur de tableaux par IA
        """
        super().__init__()
        self.table_detector = table_detector
        
        # Initialiser les sous-stratégies
        self.ocr_strategy = OCRTableStrategy()
        self.tabula_strategy = TabulaTableStrategy()
        self.camelot_strategy = CamelotTableStrategy()
        
    async def extract_tables(self, context: TableExtractionContext) -> List[ProcessedTable]:
        """
        Extrait les tableaux avec une approche hybride.
        
        Args:
            context: Contexte d'extraction
            
        Returns:
            Liste de tableaux traités
        """
        try:
            with metrics.timer("hybrid_extraction"):
                all_tables = []
                
                # 1. Déterminer l'ordre des stratégies en fonction du type de PDF
                if context.pdf_type == PDFType.SCANNED:
                    # PDF scanné: privilégier OCR puis IA
                    strategies = [
                        (self.ocr_strategy, "ocr", 1.0),
                        (self.tabula_strategy, "tabula", 0.6)
                    ]
                    
                    # Ajouter la stratégie IA si disponible
                    if self.table_detector:
                        ai_strategy = AIDetectionTableStrategy(self.table_detector)
                        strategies.insert(0, (ai_strategy, "ai-detection", 1.1))
                else:
                    # PDF numérique: d'abord tabula, puis camelot, puis OCR
                    strategies = [
                        (self.tabula_strategy, "tabula", 1.0),
                        (self.camelot_strategy, "camelot", 0.9),
                        (self.ocr_strategy, "ocr", 0.7)
                    ]
                
                # 2. Exécuter les stratégies en parallèle
                tasks = []
                for strategy, name, weight in strategies:
                    tasks.append(self._run_strategy_with_weight(strategy, context, name, weight))
                
                # Attendre tous les résultats
                results = await asyncio.gather(*tasks)
                
                # 3. Fusionner les résultats en évitant les doublons
                all_tables = await self._merge_tables_results(results)
                
                logger.info(f"Hybride: {len(all_tables)} tableaux extraits")
                return all_tables
                
        except Exception as e:
            logger.error(f"Erreur extraction hybride: {e}")
            return []
    
    async def _run_strategy_with_weight(
        self, 
        strategy: PDFTableStrategy, 
        context: TableExtractionContext,
        name: str,
        weight: float
    ) -> List[ProcessedTable]:
        """Exécute une stratégie et ajuste les scores de confiance."""
        try:
            result = await strategy.extract_tables(context)
            
            # Ajuster les scores et le nom de la méthode
            for table in result:
                table.confidence *= weight
                table.method = f"hybrid-{name}"
                
            return result
        except Exception as e:
            logger.warning(f"Erreur exécution stratégie {name}: {e}")
            return []
    
    async def _merge_tables_results(self, results: List[List[ProcessedTable]]) -> List[ProcessedTable]:
        """
        Fusionne les résultats de plusieurs stratégies en évitant les doublons.
        
        Args:
            results: Liste de listes de tableaux
            
        Returns:
            Liste unifiée de tableaux
        """
        # Aplatir les résultats
        all_tables = [table for sublist in results for table in sublist]
        
        # Trier par page puis par confiance
        all_tables.sort(key=lambda x: (x.page, -x.confidence))
        
        # Détecter et supprimer les doublons
        unique_tables = []
        seen_signatures = set()
        
        for table in all_tables:
            # Créer une signature basée sur la page et la structure
            signature = (table.page, table.rows, table.columns)
            
            # Si tableau similaire déjà vu et confiance inférieure, ignorer
            if signature in seen_signatures:
                # Vérifier si c'est réellement un doublon en comparant le contenu
                is_duplicate = False
                for existing in unique_tables:
                    if (existing.page == table.page and 
                        existing.rows == table.rows and 
                        existing.columns == table.columns):
                        
                        # Comparer le contenu
                        similarity = await self._calculate_table_similarity(existing, table)
                        if similarity > 0.7:  # 70% de similarité
                            is_duplicate = True
                            # Garder la méthode avec la plus grande confiance
                            if table.confidence > existing.confidence:
                                existing.method = table.method
                                existing.confidence = table.confidence
                            break
                
                if is_duplicate:
                    continue
            
            seen_signatures.add(signature)
            unique_tables.append(table)
        
        # Trier par page et position dans la page (si disponible)
        unique_tables.sort(key=lambda x: (x.page, x.region.y if x.region else 0))
        
        return unique_tables
    
    async def _calculate_table_similarity(self, table1: ProcessedTable, table2: ProcessedTable) -> float:
        """Calcule la similarité entre deux tableaux."""
        # Si la structure est différente, similarité nulle
        if table1.rows != table2.rows or table1.columns != table2.columns:
            return 0.0
        
        try:
            # Convertir en DataFrames si ce n'est pas déjà le cas
            df1 = table1.data if isinstance(table1.data, pd.DataFrame) else pd.DataFrame(table1.data)
            df2 = table2.data if isinstance(table2.data, pd.DataFrame) else pd.DataFrame(table2.data)
            
            # Conversion en chaînes pour comparaison
            df1_str = df1.astype(str)
            df2_str = df2.astype(str)
            
            # Nombre de cellules identiques
            identical_cells = (df1_str == df2_str).sum().sum()
            total_cells = df1.size
            
            return identical_cells / total_cells if total_cells > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Erreur calcul similarité: {e}")
            return 0.0
            
    async def cleanup(self):
        """Nettoie les ressources."""
        await super().cleanup()
        
        # Nettoyage des sous-stratégies
        strategies = [self.ocr_strategy, self.tabula_strategy, self.camelot_strategy]
        for strategy in strategies:
            if strategy and hasattr(strategy, 'cleanup'):
                await strategy.cleanup()

class OCRTableStrategy(PDFTableStrategy):
    """Stratégie d'extraction de tableaux par OCR."""
    
    def __init__(self, image_processor=None):
        """
        Initialise la stratégie OCR.
        
        Args:
            image_processor: Processeur d'image optionnel
        """
        super().__init__()
        self.image_processor = image_processor or PDFImageProcessor()
        
        # Configuration depuis settings
        ocr_config = settings.table_extraction.OCR
        self.tesseract_cmd = ocr_config.TESSERACT_CMD
        self.tesseract_lang = ocr_config.TESSERACT_LANG
        self.dpi = ocr_config.OCR_DPI
        
        # Initialisation de Tesseract
        if os.path.exists(self.tesseract_cmd) and os.access(self.tesseract_cmd, os.X_OK):
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
    
    async def extract_tables(self, context: TableExtractionContext) -> List[ProcessedTable]:
        """
        Extrait les tableaux avec OCR.
        
        Args:
            context: Contexte d'extraction
            
        Returns:
            Liste de tableaux traités
        """
        try:
            with metrics.timer("ocr_extraction"):
                # Récupération de la config OCR
                ocr_config = context.ocr_config
                lang = ocr_config.get("lang", "fra+eng")
                enhance_image = ocr_config.get("enhance_image", True)
                deskew = ocr_config.get("deskew", True)
                preprocess_type = ocr_config.get("preprocess_type", "thresh")
                psm = ocr_config.get("psm", 6)
                force_grid = ocr_config.get("force_grid", False)
                
                # Conversion des pages en images
                page_indices = await parse_page_range(context.file_path, context.pages)
                if not page_indices:
                    return []
                
                page_images = await convert_pdf_to_images(context.file_path, page_indices)
                
                # Traitement de chaque image de page
                tasks = []
                for idx, img in enumerate(page_images):
                    # Association de l'index de page
                    page_num = page_indices[idx] + 1 if idx < len(page_indices) else idx + 1
                    tasks.append(self._process_page_image(
                        img, page_num, lang, enhance_image, deskew, preprocess_type, psm, force_grid
                    ))
                
                # Exécution parallèle
                results = await asyncio.gather(*tasks)
                
                # Consolidation des résultats
                tables = [table for page_tables in results for table in page_tables if table]
                logger.info(f"OCR: {len(tables)} tableaux extraits")
                return tables
                
        except Exception as e:
            logger.error(f"Erreur extraction OCR: {e}")
            return []
    
    async def _process_page_image(
        self, 
        image: np.ndarray,
        page_num: int,
        lang: str,
        enhance_image: bool,
        deskew: bool,
        preprocess_type: str,
        psm: int,
        force_grid: bool
    ) -> List[ProcessedTable]:
        """
        Traite une image de page pour détecter et extraire les tableaux.
        
        Args:
            image: Image de page (numpy array)
            page_num: Numéro de la page
            lang: Langue pour l'OCR
            enhance_image: Si True, améliore l'image
            deskew: Si True, redresse l'image
            preprocess_type: Type de prétraitement ('thresh', 'adaptive', 'blur')
            psm: Mode de segmentation Tesseract
            force_grid: Si True, force l'approche grille
            
        Returns:
            Liste de tableaux extraits
        """
        try:
            # Prétraitement de l'image
            processed_img = await self._preprocess_image(image, enhance_image, deskew, preprocess_type)
            
            # Détection des régions de tableau
            table_regions = await self._detect_table_regions(processed_img)
            
            # Extraction des tableaux
            tables = []
            for i, region in enumerate(table_regions):
                try:
                    # Extraire la région d'intérêt
                    table_img = processed_img[region.y:region.y+region.height, region.x:region.x+region.width]
                    
                    # Amélioration de l'image du tableau
                    table_img = await self._enhance_table_image(table_img)
                    
                    # Extraction du tableau par OCR
                    if force_grid:
                        df = await self._ocr_table_grid_approach(table_img, lang, psm)
                    else:
                        df = await self._ocr_table_to_dataframe(table_img, lang, psm)
                    
                    if df is not None and not df.empty and len(df.columns) > 1:
                        tables.append(ProcessedTable(
                            data=df,
                            page=page_num,
                            rows=len(df),
                            columns=len(df.columns),
                            method="ocr",
                            confidence=0.6,  # Score fixe pour OCR
                            region=region
                        ))
                except Exception as e:
                    logger.warning(f"Erreur extraction tableau OCR {i}: {e}")
            
            return tables
            
        except Exception as e:
            logger.error(f"Erreur traitement image page {page_num}: {e}")
            return []
    
    async def _preprocess_image(
        self, 
        image: np.ndarray, 
        enhance: bool, 
        deskew: bool, 
        preprocess_type: str
    ) -> np.ndarray:
        """
        Prétraite une image pour améliorer la qualité pour l'OCR.
        
        Args:
            image: Image à prétraiter
            enhance: Si True, améliore le contraste
            deskew: Si True, redresse l'image
            preprocess_type: Type de prétraitement
            
        Returns:
            Image prétraitée
        """
        try:
            # Conversion en PIL Image pour certains traitements
            pil_image = Image.fromarray(image)
            
            # Redressement si demandé
            if deskew:
                loop = asyncio.get_event_loop()
                pil_image = await loop.run_in_executor(
                    self.executor,
                    self.image_processor.deskew_image,
                    pil_image
                )
            
            # Amélioration du contraste
            if enhance:
                from PIL import ImageEnhance
                
                # Augmenter le contraste
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.5)
                
                # Augmenter la netteté
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.5)
            
            # Reconversion en numpy array
            image = np.array(pil_image)
            
            # Conversion en niveaux de gris
            if len(image.shape) > 2 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Application du prétraitement selon le type
            if preprocess_type == "thresh":
                # Binarisation Otsu
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return binary
            elif preprocess_type == "adaptive":
                # Binarisation adaptative
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
                return binary
            elif preprocess_type == "blur":
                # Réduction du bruit
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return binary
            else:
                return gray
                
        except Exception as e:
            logger.error(f"Erreur prétraitement image: {e}")
            return image
    
    async def _detect_table_regions(self, image: np.ndarray) -> List[TableRegion]:
        """
        Détecte les régions de tableau dans une image.
        
        Args:
            image: Image prétraitée
            
        Returns:
            Liste de régions de tableau
        """
        try:
            # Si l'image est en couleur, convertir en niveaux de gris
            if len(image.shape) > 2 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Détection des lignes
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Détection des lignes horizontales et verticales
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combiner les lignes
            table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
            
            # Dilater pour connecter les composants
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            table_mask = cv2.dilate(table_mask, kernel, iterations=4)
            
            # Trouver les contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrer les petits contours
            min_area = 0.01 * image.shape[0] * image.shape[1]  # Au moins 1% de l'image
            regions = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > min_area:
                    # Vérifier les proportions
                    aspect_ratio = w / h
                    if 0.2 < aspect_ratio < 5:  # Pas trop allongé
                        regions.append(TableRegion(
                            x=x, y=y, width=w, height=h, confidence=0.7
                        ))
            
            # Si aucune région détectée, considérer toute l'image
            if not regions:
                regions = [TableRegion(
                    x=0, y=0, width=image.shape[1], height=image.shape[0], confidence=0.5
                )]
            
            return regions
            
        except Exception as e:
            logger.error(f"Erreur détection régions de tableau: {e}")
            # En cas d'erreur, retourner toute l'image
            return [TableRegion(
                x=0, y=0, width=image.shape[1], height=image.shape[0], confidence=0.5
            )]
    
    async def _enhance_table_image(self, image: np.ndarray) -> np.ndarray:
        """
        Améliore une image de tableau pour l'OCR.
        
        Args:
            image: Image du tableau
            
        Returns:
            Image améliorée
        """
        try:
            # Conversion en PIL Image
            pil_image = Image.fromarray(image)
            
            from PIL import ImageEnhance
            
            # Augmenter le contraste
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(2.0)
            
            # Augmenter la netteté
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(2.0)
            
            # Reconversion en numpy array
            enhanced = np.array(pil_image)
            
            # Si l'image est en couleur, conversion en niveaux de gris
            if len(enhanced.shape) > 2 and enhanced.shape[2] == 3:
                gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            else:
                gray = enhanced
            
            # Débruitage
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # Amélioration du contraste adaptatif
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(denoised)
            
        except Exception as e:
            logger.error(f"Erreur amélioration image: {e}")
            return image
    
    async def _ocr_table_to_dataframe(
        self, 
        table_img: np.ndarray, 
        lang: str = "fra+eng", 
        psm: int = 6
    ) -> Optional[pd.DataFrame]:
        """
        Convertit une image de tableau en DataFrame par OCR.
        
        Args:
            table_img: Image du tableau
            lang: Langue pour l'OCR
            psm: Mode de segmentation Tesseract
            
        Returns:
            DataFrame ou None si échec
        """
        try:
            # Prétraitement amélioré
            enhanced_img = await self._enhance_table_image(table_img, high_contrast=True)
            
            # Application d'un preprocessing spécifique aux tableaux
            binary = cv2.adaptiveThreshold(
                enhanced_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Détection des lignes
            horizontal = self._detect_lines(binary, is_horizontal=True)
            vertical = self._detect_lines(binary, is_horizontal=False)
            
            # Combiner pour obtenir la grille
            grid = cv2.bitwise_or(horizontal, vertical)
            
            # Si une grille est détectée, utiliser l'approche par cellules
            if np.sum(grid) > 100:  # Seuil arbitraire pour détecter une grille
                # Détection des cellules par intersection des lignes
                cells = self._detect_cells(grid, binary)
                if cells and len(cells) > 4:  # Au moins quelques cellules
                    return await self._process_cells_to_dataframe(cells, enhanced_img, lang)
            
            # Si pas de grille ou échec, utiliser l'OCR normal
            loop = asyncio.get_event_loop()
            
            # Configuration OCR optimisée
            custom_config = f'--oem 3 --psm {psm} -l {lang} --dpi 300'
            
            # Exécuter l'OCR
            text = await loop.run_in_executor(
                self.executor,
                lambda: pytesseract.image_to_string(enhanced_img, config=custom_config)
            )
            
            if not text.strip():
                logger.warning("Aucun texte détecté par OCR")
                return None
            
            # Traitement amélioré du texte OCR
            return await self._parse_ocr_text_to_dataframe(text)
            
        except Exception as e:
            logger.error(f"Erreur OCR tableau amélioré: {e}")
            return None
    
    def _detect_separator(self, lines: List[str]) -> str:
        """Détecte le séparateur probable dans le texte."""
        separators = {'|': 0, ',': 0, ';': 0, '\t': 0}
        
        for line in lines:
            for sep, count in separators.items():
                separators[sep] += line.count(sep)
        
        max_sep = max(separators.items(), key=lambda x: x[1])
        
        if max_sep[1] > 0:
            return max_sep[0]
        
        return 'whitespace'
    
    def _guess_columns(self, line: str) -> List[str]:
        """Devine les colonnes dans une ligne sans séparateur clair."""
        parts = []
        current_part = ""
        space_count = 0
        
        for char in line:
            if char == ' ':
                space_count += 1
                if space_count >= 3:  # 3 espaces = séparateur
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
        
        # Si pas de colonnes identifiées, diviser par espaces simples
        if len(parts) <= 1:
            parts = line.split()
        
        return [p.strip() for p in parts]
    
    def _is_valid_header(self, header_row: List[str], data_rows: List[List[str]]) -> bool:
        """Vérifie si une ligne est un en-tête valide."""
        if not data_rows:
            return False
        
class EnhancedHybridStrategy(PDFTableStrategy):
    """Stratégie hybride améliorée avec meilleure pondération des résultats."""
    
    def __init__(self, table_detector=None):
        super().__init__()
        self.table_detector = table_detector
        
        # Initialiser les stratégies avec paramètres optimisés
        self.camelot_strategy = CamelotTableStrategy()
        self.tabula_strategy = TabulaTableStrategy()
        self.pdfplumber_strategy = PDFPlumberTableStrategy()
        self.ocr_strategy = OCRTableStrategy()
        
    async def extract_tables(self, context: TableExtractionContext) -> List[ProcessedTable]:
        try:
            with metrics.timer("enhanced_hybrid_extraction"):
                # Initialiser les pondérations selon le type de PDF
                if context.pdf_type == PDFType.SCANNED:
                    strategies = [
                        (self.ocr_strategy, "ocr", 1.2)
                    ]
                    
                    # Ajouter IA si disponible
                    if self.table_detector:
                        ai_strategy = AIDetectionTableStrategy(self.table_detector)
                        strategies.insert(0, (ai_strategy, "ai-detection", 1.3))
                elif context.pdf_type == PDFType.HYBRID:
                    # Pour PDF hybride, essayer plusieurs approches
                    strategies = [
                        (self.camelot_strategy, "camelot", 1.0),
                        (self.tabula_strategy, "tabula", 0.9),
                        (self.pdfplumber_strategy, "pdfplumber", 0.8)
                    ]
                    # Ajouter OCR avec faible pondération
                    strategies.append((self.ocr_strategy, "ocr", 0.6))
                else:
                    # PDF digital standard
                    strategies = [
                        (self.camelot_strategy, "camelot", 1.1),
                        (self.tabula_strategy, "tabula", 1.0),
                        (self.pdfplumber_strategy, "pdfplumber", 0.8)
                    ]
                
                # Exécuter les stratégies en parallèle
                tasks = []
                for strategy, name, weight in strategies:
                    tasks.append(self._run_strategy_with_weight(strategy, context, name, weight))
                
                # Attendre tous les résultats
                results = await asyncio.gather(*tasks)
                
                # Filtrer les résultats vides
                results = [tables for tables in results if tables]
                
                if not results:
                    logger.warning("Aucun résultat trouvé avec les stratégies hybrides, tentative avec approche simple")
                    # Fallback avec stratégie simple
                    return await self._run_fallback_extraction(context)
                
                # Fusionner les résultats avec meilleure gestion des tableaux
                all_tables = await self._enhanced_merge_tables_results(results, context)
                
                # Appliquer une validation finale
                validated_tables = []
                for table in all_tables:
                    # Validation personnalisée pour éviter les faux positifs
                    if self._is_valid_table(table):
                        validated_tables.append(table)
                
                logger.info(f"Hybride amélioré: {len(validated_tables)} tableaux extraits")
                return validated_tables
                
        except Exception as e:
            logger.error(f"Erreur extraction hybride améliorée: {e}")
            return []
    
    async def _run_strategy_with_weight(
        self, 
        strategy: PDFTableStrategy, 
        context: TableExtractionContext,
        name: str,
        weight: float
    ) -> List[ProcessedTable]:
        """Exécute une stratégie et ajuste les scores de confiance."""
        try:
            result = await strategy.extract_tables(context)
            
            # Ajuster les scores et le nom de la méthode
            for table in result:
                table.confidence *= weight
                table.method = f"enhanced-hybrid-{name}"
                
            return result
        except Exception as e:
            logger.warning(f"Erreur exécution stratégie {name}: {e}")
            return []
    
    async def _run_fallback_extraction(self, context: TableExtractionContext) -> List[ProcessedTable]:
        """Méthode de secours si aucune stratégie ne donne de résultat."""
        try:
            # Essayer avec PDFPlumber qui est souvent plus tolérant
            tables = await self.pdfplumber_strategy.extract_tables(context)
            
            if not tables:
                # Si toujours rien, essayer OCR comme dernier recours
                tables = await self.ocr_strategy.extract_tables(context)
                
            return tables
        except Exception as e:
            logger.error(f"Erreur extraction de secours: {e}")
            return []
    
    async def _enhanced_merge_tables_results(
        self, 
        results: List[List[ProcessedTable]],
        context: TableExtractionContext
    ) -> List[ProcessedTable]:
        """
        Fusion améliorée des résultats avec gestion des doublons et règles de priorité.
        """
        # Aplatir les résultats
        all_tables = [table for sublist in results for table in sublist]
        
        # Trier par page puis par confiance
        all_tables.sort(key=lambda x: (x.page, -x.confidence))
        
        # Collecte intelligente de tableaux uniques
        unique_tables = []
        seen_regions = {}  # {page: [regions]}
        
        for table in all_tables:
            page = table.page
            
            # Initialiser la liste des régions pour cette page si nécessaire
            if page not in seen_regions:
                seen_regions[page] = []
            
            # Vérifier si la région du tableau chevauche une région existante
            is_duplicate = False
            for region_idx, existing_region in enumerate(seen_regions[page]):
                # Si le tableau a une région définie
                if table.region and existing_region:
                    # Calculer le chevauchement
                    overlap = self._calculate_region_overlap(table.region, existing_region)
                    
                    # Si chevauchement significatif, c'est un doublon potentiel
                    if overlap > 0.6:  # 60% de chevauchement
                        # Trouver le tableau correspondant pour comparaison
                        existing_table = next((t for t in unique_tables 
                                              if t.page == page and t.region == existing_region), None)
                        
                        if existing_table:
                            # Comparer sur contenu et qualité
                            similarity = await self._calculate_table_similarity(existing_table, table)
                            
                            # Si tables similaires, garder celle avec la meilleure confiance
                            if similarity > 0.7:  # 70% de similitude
                                is_duplicate = True
                                
                                # Remplacer si la nouvelle table est meilleure
                                if table.confidence > existing_table.confidence:
                                    # Trouver l'index du tableau existant
                                    table_idx = unique_tables.index(existing_table)
                                    unique_tables[table_idx] = table
                                    seen_regions[page][region_idx] = table.region
                                break
            
            # Si ce n'est pas un doublon, l'ajouter
            if not is_duplicate:
                if table.region:
                    seen_regions[page].append(table.region)
                unique_tables.append(table)
        
        # Vérification finale de la qualité
        final_tables = []
        for table in unique_tables:
            # Limiter le nombre de tableaux par page
            page_tables = [t for t in final_tables if t.page == table.page]
            if len(page_tables) >= 5:  # Max 5 tableaux par page
                # Ne garder que si meilleure confiance qu'un des tableaux existants
                min_confidence_table = min(page_tables, key=lambda t: t.confidence)
                if table.confidence > min_confidence_table.confidence:
                    final_tables.remove(min_confidence_table)
                    final_tables.append(table)
            else:
                final_tables.append(table)
        
        # Trier par page et position dans la page (si disponible)
        final_tables.sort(key=lambda x: (x.page, x.region.y if x.region else 0))
        
        return final_tables
    
    def _calculate_region_overlap(self, region1: TableRegion, region2: TableRegion) -> float:
        """Calcule le chevauchement entre deux régions."""
        # Calcul des coordonnées des régions
        r1_x1, r1_y1 = region1.x, region1.y
        r1_x2, r1_y2 = region1.x + region1.width, region1.y + region1.height
        
        r2_x1, r2_y1 = region2.x, region2.y
        r2_x2, r2_y2 = region2.x + region2.width, region2.y + region2.height
        
        # Calculer l'intersection
        x_overlap = max(0, min(r1_x2, r2_x2) - max(r1_x1, r2_x1))
        y_overlap = max(0, min(r1_y2, r2_y2) - max(r1_y1, r2_y1))
        intersection = x_overlap * y_overlap
        
        # Calculer l'union
        r1_area = region1.width * region1.height
        r2_area = region2.width * region2.height
        union = r1_area + r2_area - intersection
        
        # Retourner le rapport intersection/union (IoU)
        return intersection / union if union > 0 else 0
    
    async def _calculate_table_similarity(self, table1: ProcessedTable, table2: ProcessedTable) -> float:
        """Calcule la similarité entre deux tableaux."""
        # Si la structure est différente, similarité nulle
        if table1.rows != table2.rows or table1.columns != table2.columns:
            return 0.0
        
        try:
            # Convertir en DataFrames si ce n'est pas déjà le cas
            df1 = table1.data if isinstance(table1.data, pd.DataFrame) else pd.DataFrame(table1.data)
            df2 = table2.data if isinstance(table2.data, pd.DataFrame) else pd.DataFrame(table2.data)
            
            # Vérifier que les deux DataFrames ont au moins une ligne
            if len(df1) == 0 or len(df2) == 0:
                return 0.0
            
            # Conversion en chaînes pour comparaison
            df1_str = df1.fillna('').astype(str)
            df2_str = df2.fillna('').astype(str)
            
            # Nombre de cellules identiques
            identical_cells = 0
            total_cells = 0
            
            # Comparer les cellules en tenant compte de variations mineures
            for col in df1_str.columns:
                if col in df2_str.columns:
                    for idx in range(min(len(df1_str), len(df2_str))):
                        if idx >= len(df1_str) or idx >= len(df2_str):
                            continue
                            
                        val1 = df1_str.iloc[idx][col]
                        val2 = df2_str.iloc[idx][col]
                        
                        # Nettoyage léger pour la comparaison
                        val1 = val1.strip().lower()
                        val2 = val2.strip().lower()
                        
                        # Identique ou très similaire
                        if val1 == val2 or self._strings_very_similar(val1, val2):
                            identical_cells += 1
                        
                        total_cells += 1
            
            # Retourner le ratio de similarité
            return identical_cells / total_cells if total_cells > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Erreur calcul similarité: {e}")
            return 0.0
    
    def _strings_very_similar(self, s1: str, s2: str, threshold: float = 0.85) -> bool:
        """Détermine si deux chaînes sont très similaires (tolérance aux erreurs OCR)."""
        if not s1 and not s2:
            return True
        if not s1 or not s2:
            return False
            
        # Si l'une est contenue dans l'autre
        if s1 in s2 or s2 in s1:
            return True
            
        # Calcul de similarité simple
        # On pourrait utiliser Levenshtein mais pour la simplicité, on utilise une approche basique
        shorter = s1 if len(s1) <= len(s2) else s2
        longer = s2 if len(s1) <= len(s2) else s1
        
        # Si trop de différence de longueur
        if len(shorter) == 0:
            return len(longer) == 0
            
        if len(longer) / len(shorter) > 2:
            return False
            
        # Compte de caractères communs
        s1_chars = set(s1)
        s2_chars = set(s2)
        common_chars = s1_chars.intersection(s2_chars)
        
        # Ratio de caractères communs
        ratio = len(common_chars) / max(len(s1_chars), len(s2_chars))
        
        return ratio >= threshold
    
    def _is_valid_table(self, table: ProcessedTable) -> bool:
        """Validation améliorée des tableaux."""
        if table.rows < 2 or table.columns < 2:
            return False
            
        # Vérifier que ce n'est pas juste du texte formaté
        df = table.data
        
        # Si plus de 80% des cellules sont vides, c'est probablement un faux positif
        empty_ratio = df.isna().sum().sum() / df.size
        if empty_ratio > 0.8:
            return False
            
        # Vérifier la cohérence des données
        # Si toutes les colonnes ont des types de données différents, c'est suspect
        try:
            num_cols = len(df.select_dtypes(include=[np.number]).columns)
            str_cols = len(df.select_dtypes(include=['object']).columns)
            
            # Un vrai tableau a généralement une structure cohérente
            if (num_cols == 1 and str_cols > 3) or (str_cols == 1 and num_cols > 3):
                # Structure déséquilibrée
                column_value_counts = {col: df[col].nunique() for col in df.columns}
                
                # Vérifier si une seule colonne contient toutes les valeurs différentes
                max_unique = max(column_value_counts.values()) if column_value_counts else 0
                if max_unique == len(df) and len(df) > 4:
                    # C'est probablement une liste et non un tableau
                    return False
        except:
            # Erreur dans l'analyse, on accepte le tableau par défaut
            pass
            
        return True
        
    async def cleanup(self):
        """Nettoie les ressources."""
        await super().cleanup()
        
        # Nettoyage des sous-stratégies
        strategies = [self.camelot_strategy, self.tabula_strategy, 
                      self.pdfplumber_strategy, self.ocr_strategy]
        for strategy in strategies:
            if strategy and hasattr(strategy, 'cleanup'):
                await strategy.cleanup()