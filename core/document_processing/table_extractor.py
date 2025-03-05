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

from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings

logger = get_logger("table_extractor")

class TableExtractor:
    """Classe pour extraire des tableaux de fichiers PDF."""
    
    def __init__(self):
        """Initialisation de l'extracteur de tableaux."""
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def extract_tables(
        self, 
        file_path: Union[str, Path, BytesIO], 
        pages: Union[str, List[int]] = "all",
        extraction_method: str = "auto",
        output_format: str = "pandas"
    ) -> List[Dict[str, Any]]:
        """
        Extrait les tableaux d'un fichier PDF.
        
        Args:
            file_path: Chemin du fichier PDF ou objet BytesIO
            pages: Pages à analyser ("all" ou liste de numéros de page)
            extraction_method: Méthode d'extraction ("auto", "tabula", "camelot", "pdfplumber")
            output_format: Format de sortie ("pandas", "csv", "json", "html")
            
        Returns:
            Liste de tables extraites dans le format demandé
        """
        try:
            with metrics.timer("table_extraction"):
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
                
                # Détection automatique de la meilleure méthode
                if extraction_method == "auto":
                    extraction_method = await self._detect_best_method(file_path, pages)
                
                # Extraction des tableaux selon la méthode choisie
                loop = asyncio.get_event_loop()
                if extraction_method == "tabula":
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
    
    async def get_tables_as_images(
        self, 
        file_path: Union[str, Path, BytesIO],
        pages: Union[str, List[int]] = "all",
        dpi: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Extrait les tableaux sous forme d'images.
        
        Args:
            file_path: Chemin du fichier PDF ou objet BytesIO
            pages: Pages à analyser
            dpi: Résolution des images en DPI
            
        Returns:
            Liste de dictionnaires contenant les images des tableaux
        """
        try:
            from pdf2image import convert_from_path, convert_from_bytes
            import cv2
            import numpy as np
            from PIL import Image
            
            # Gérer les entrées BytesIO
            temp_file = None
            if isinstance(file_path, BytesIO):
                temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                temp_file.write(file_path.getvalue())
                temp_file.close()
                file_path = temp_file.name
            
            # Conversion du chemin en str si c'est un Path
            if isinstance(file_path, Path):
                file_path = str(file_path)
            
            # Extraire les tableaux avec Camelot pour obtenir les coordonnées
            if pages == "all":
                # Obtenir le nombre de pages
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    pages = f"1-{len(pdf.pages)}"
            
            tables = camelot.read_pdf(
                file_path, 
                pages=pages,
                flavor="lattice",
                suppress_stdout=True
            )
            
            # Si aucun tableau trouvé avec lattice, essayer stream
            if len(tables) == 0:
                tables = camelot.read_pdf(
                    file_path, 
                    pages=pages,
                    flavor="stream",
                    suppress_stdout=True
                )
            
            # Convertir les pages en images
            images = convert_from_path(file_path, dpi=dpi) if isinstance(file_path, str) else \
                     convert_from_bytes(file_path.read(), dpi=dpi)
            
            result = []
            
            for table in tables:
                try:
                    # Récupérer l'index de page (0-based)
                    page_idx = int(table.page) - 1
                    if page_idx < 0 or page_idx >= len(images):
                        continue
                    
                    # Récupérer l'image de la page
                    img = np.array(images[page_idx])
                    
                    # Récupérer les coordonnées du tableau (normalisées entre 0 et 1)
                    coords = table._bbox
                    
                    # Convertir les coordonnées en pixels
                    height, width = img.shape[:2]
                    x1 = int(coords[0] * width)
                    y1 = int((1 - coords[3]) * height)  # Note: PDF coordinates start from bottom
                    x2 = int(coords[2] * width)
                    y2 = int((1 - coords[1]) * height)
                    
                    # Découper l'image
                    table_img = img[y1:y2, x1:x2]
                    
                    # Vérifier que l'image n'est pas vide
                    if table_img.size == 0:
                        continue
                    
                    # Convertir l'image en base64
                    pil_img = Image.fromarray(table_img)
                    buffer = BytesIO()
                    pil_img.save(buffer, format="PNG")
                    buffer.seek(0)
                    img_str = base64.b64encode(buffer.read()).decode("utf-8")
                    
                    # Ajouter au résultat
                    result.append({
                        "table_id": len(result) + 1,
                        "page": table.page,
                        "bbox": [x1, y1, x2, y2],
                        "accuracy": table.parsing_report["accuracy"],
                        "image": img_str,
                        "format": "base64"
                    })
                    
                except Exception as e:
                    logger.warning(f"Erreur extraction image du tableau: {e}")
            
            # Nettoyage
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur extraction tableaux en images: {e}")
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            return []
    
    async def cleanup(self):
        """Nettoie les ressources."""
        self.executor.shutdown(wait=False)