# core/utils/pdf_table_utils.py
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import asyncio
import pandas as pd
import numpy as np
import io
import os
import json
import tempfile
import uuid
from datetime import datetime

from core.document_processing.table_extractor import TableExtractor
from core.utils.logger import get_logger

logger = get_logger("pdf_table_utils")

async def extract_tables_from_pdf(
    pdf_file: Union[str, Path, bytes, io.BytesIO],
    output_format: str = "dataframe",
    max_pages: Optional[int] = None,
    save_to: Optional[str] = None,
    ocr_language: str = "fra+eng",
    enhance_ocr: bool = True
) -> Union[List[pd.DataFrame], List[Dict[str, Any]], str]:
    """
    Fonction utilitaire simple pour extraire des tableaux de n'importe quel PDF.
    
    Cette fonction est le point d'entrée principal pour l'extraction de tableaux.
    Elle détecte automatiquement le type de PDF et utilise la meilleure méthode.
    
    Args:
        pdf_file: Chemin du fichier PDF, bytes ou BytesIO
        output_format: Format de sortie ('dataframe', 'dict', 'csv', 'json', 'html', 'excel')
        max_pages: Nombre maximum de pages à traiter (None pour toutes)
        save_to: Dossier où sauvegarder les résultats (None pour ne pas sauvegarder)
        ocr_language: Langages pour l'OCR (fra+eng par défaut)
        enhance_ocr: Si True, améliore la qualité de l'OCR
        
    Returns:
        Selon le format demandé:
        - 'dataframe': Liste de pandas DataFrames
        - 'dict': Liste de dictionnaires au format JSON
        - 'csv'/'json'/'html'/'excel': Liste de tableaux dans le format demandé
    """
    try:
        # Créer une instance de TableExtractor
        extractor = TableExtractor()
        
        # Convertir les formats d'entrée
        file_source = pdf_file
        
        # Convertir les bytes en BytesIO
        if isinstance(pdf_file, bytes):
            file_source = io.BytesIO(pdf_file)
        
        # Convertir le format de sortie interne
        internal_format = "pandas"
        if output_format == "dict":
            internal_format = "json"
        elif output_format in ["csv", "json", "html", "excel"]:
            internal_format = output_format
        
        # Configuration OCR (utilisée seulement si nécessaire)
        ocr_config = {
            "lang": ocr_language,
            "enhance_image": enhance_ocr,
            "deskew": enhance_ocr,
            "preprocess_type": "thresh" if enhance_ocr else "adaptive"
        }
        
        # Extraire les tableaux
        tables = await extractor.extract_tables_auto(
            file_source=file_source,
            output_format=internal_format,
            max_pages=max_pages,
            save_to=save_to,
            ocr_config=ocr_config
        )
        
        # Post-traitement selon le format de sortie demandé
        if output_format == "dataframe":
            # Retourner directement les DataFrames
            return [table["data"] for table in tables if "data" in table]
        else:
            # Retourner le format demandé
            return tables
            
    except Exception as e:
        logger.error(f"Erreur extraction des tableaux: {e}")
        # En cas d'erreur, retourner une liste vide
        return []

def extract_tables_sync(
    pdf_file: Union[str, Path, bytes, io.BytesIO],
    output_format: str = "dataframe",
    max_pages: Optional[int] = None,
    save_to: Optional[str] = None,
    ocr_language: str = "fra+eng",
    enhance_ocr: bool = True
) -> Union[List[pd.DataFrame], List[Dict[str, Any]], str]:
    """
    Version synchrone de extract_tables_from_pdf pour une utilisation plus simple.
    
    Args: Identiques à extract_tables_from_pdf
    Returns: Identiques à extract_tables_from_pdf
    """
    try:
        # Créer et exécuter une boucle d'événements si nécessaire
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Si aucune boucle d'événements n'existe dans ce thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Exécuter la fonction asynchrone
        return loop.run_until_complete(extract_tables_from_pdf(
            pdf_file=pdf_file,
            output_format=output_format,
            max_pages=max_pages,
            save_to=save_to,
            ocr_language=ocr_language,
            enhance_ocr=enhance_ocr
        ))
    
    except Exception as e:
        logger.error(f"Erreur extraction synchrone des tableaux: {e}")
        return []

def save_tables_to_file(
    tables: Union[List[pd.DataFrame], List[Dict[str, Any]]],
    output_dir: str,
    base_filename: str = "table",
    output_format: str = "csv",
    sheet_name: Optional[str] = None
) -> List[str]:
    """
    Sauvegarde les tableaux extraits dans des fichiers.
    
    Args:
        tables: Tableaux à sauvegarder (DataFrames ou dictionnaires)
        output_dir: Répertoire de sortie
        base_filename: Nom de base pour les fichiers
        output_format: Format de sortie ('csv', 'json', 'excel', 'html')
        sheet_name: Nom de la feuille pour Excel (si format='excel')
        
    Returns:
        Liste des chemins de fichiers créés
    """
    try:
        # Créer le répertoire de sortie si nécessaire
        os.makedirs(output_dir, exist_ok=True)
        
        # Vérifier que les tableaux ne sont pas vides
        if not tables:
            logger.warning("Aucun tableau à sauvegarder")
            return []
        
        # Liste pour les chemins de fichiers
        file_paths = []
        
        # Si output_format est 'excel' et qu'il y a plusieurs tableaux, les mettre dans un seul fichier
        if output_format == 'excel' and len(tables) > 1:
            excel_path = os.path.join(output_dir, f"{base_filename}.xlsx")
            with pd.ExcelWriter(excel_path) as writer:
                for i, table in enumerate(tables):
                    # Convertir en DataFrame si nécessaire
                    df = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
                    # Déterminer le nom de la feuille
                    sheet = f"{sheet_name or 'Table'}_{i+1}"
                    if len(sheet) > 31:  # Excel a une limite de 31 caractères
                        sheet = sheet[:31]
                    # Écrire le tableau
                    df.to_excel(writer, sheet_name=sheet, index=False)
            file_paths.append(excel_path)
        else:
            # Traiter chaque tableau individuellement
            for i, table in enumerate(tables):
                # Déterminer le chemin de sortie
                filename = f"{base_filename}_{i+1}.{output_format}"
                if output_format == 'excel':
                    filename = f"{base_filename}_{i+1}.xlsx"
                output_path = os.path.join(output_dir, filename)
                
                # Convertir en DataFrame si nécessaire
                df = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
                
                # Sauvegarder selon le format
                if output_format == 'csv':
                    df.to_csv(output_path, index=False)
                elif output_format == 'json':
                    df.to_json(output_path, orient='records', indent=2)
                elif output_format == 'excel':
                    df.to_excel(output_path, sheet_name=sheet_name or 'Table', index=False)
                elif output_format == 'html':
                    df.to_html(output_path, index=False)
                
                file_paths.append(output_path)
        
        return file_paths
        
    except Exception as e:
        logger.error(f"Erreur sauvegarde des tableaux: {e}")
        return []

def analyze_tables(
    tables: List[pd.DataFrame]
) -> List[Dict[str, Any]]:
    """
    Analyse une liste de tableaux et retourne des statistiques.
    
    Args:
        tables: Liste de DataFrames pandas
        
    Returns:
        Liste de dictionnaires avec les statistiques pour chaque tableau
    """
    try:
        results = []
        
        for i, df in enumerate(tables):
            # Statistiques de base
            stats = {
                "table_index": i + 1,
                "rows": len(df),
                "columns": len(df.columns),
                "empty_cells": df.isna().sum().sum(),
                "empty_cells_pct": round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2) if len(df) * len(df.columns) > 0 else 0,
                "column_stats": []
            }
            
            # Statistiques par colonne
            for col in df.columns:
                col_stats = {
                    "name": str(col),
                    "dtype": str(df[col].dtype),
                    "non_null": int(df[col].count()),
                    "null": int(df[col].isna().sum())
                }
                
                # Ajouter des statistiques spécifiques selon le type
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Statistiques numériques
                    non_na = df[col].dropna()
                    if len(non_na) > 0:
                        col_stats.update({
                            "type": "numeric",
                            "min": float(non_na.min()),
                            "max": float(non_na.max()),
                            "mean": float(non_na.mean()),
                            "median": float(non_na.median())
                        })
                elif pd.api.types.is_string_dtype(df[col]):
                    # Statistiques textuelles
                    non_na = df[col].dropna()
                    if len(non_na) > 0:
                        col_stats.update({
                            "type": "text",
                            "unique_values": int(non_na.nunique()),
                            "avg_length": float(non_na.str.len().mean()) if len(non_na) > 0 else 0
                        })
                
                stats["column_stats"].append(col_stats)
            
            results.append(stats)
            
        return results
        
    except Exception as e:
        logger.error(f"Erreur analyse des tableaux: {e}")
        return []

def merge_tables(
    tables: List[pd.DataFrame],
    method: str = "vertical",
    key_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Fusionne plusieurs tableaux en un seul.
    
    Args:
        tables: Liste de DataFrames à fusionner
        method: Méthode de fusion ('vertical', 'horizontal', 'join')
        key_column: Colonne clé pour la jointure (uniquement pour method='join')
        
    Returns:
        DataFrame fusionné
    """
    try:
        if not tables:
            return pd.DataFrame()
        
        if method == "vertical":
            # Concaténation verticale (empiler)
            return pd.concat(tables, ignore_index=True)
        elif method == "horizontal":
            # Concaténation horizontale (côte à côte)
            return pd.concat(tables, axis=1)
        elif method == "join" and key_column:
            # Jointure sur une colonne clé
            result = tables[0]
            for df in tables[1:]:
                result = result.merge(df, on=key_column, how='outer')
            return result
        else:
            # Par défaut, concaténation verticale
            return pd.concat(tables, ignore_index=True)
            
    except Exception as e:
        logger.error(f"Erreur fusion des tableaux: {e}")
        return pd.DataFrame()