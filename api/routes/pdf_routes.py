# api/routes/pdf_routes.py
from fastapi import APIRouter, UploadFile, File, Form, Query, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Union
import io
from datetime import datetime
import uuid
import tempfile
import os
import shutil

from ..dependencies import get_components
from ..models.responses import ErrorResponse
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings

logger = get_logger("pdf_routes")
router = APIRouter(prefix="/pdf", tags=["pdf"])

@router.post("/extract-tables", response_model=Dict[str, Any])
async def extract_tables_from_pdf(
    file: UploadFile = File(...),
    pages: str = Form("all"),
    extraction_method: str = Form("auto"),
    output_format: str = Form("json"),
    include_images: bool = Form(False),
    components=Depends(get_components)
):
    """
    Extrait les tableaux d'un fichier PDF.
    
    Args:
        file: Fichier PDF à traiter
        pages: Pages à analyser ("all" ou liste de numéros séparés par des virgules)
        extraction_method: Méthode d'extraction ("auto", "tabula", "camelot", "pdfplumber")
        output_format: Format de sortie ("json", "csv", "html", "excel")
        include_images: Si True, inclut les images des tableaux
        
    Returns:
        Liste des tableaux extraits
    """
    try:
        metrics.increment_counter("pdf_table_extractions")
        start_time = datetime.utcnow()
        
        # Vérification du fichier
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Le fichier doit être au format PDF"
            )
            
        # Lecture du fichier en mémoire
        contents = await file.read()
        file_obj = io.BytesIO(contents)
        file_size = len(contents)
        
        # Log pour debugging
        logger.info(f"Fichier reçu: {file.filename}, taille: {file_size / 1024:.2f} Ko")
        
        # Vérification de la taille du fichier
        max_size = 50 * 1024 * 1024  # 50 Mo
        if file_size > max_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Fichier trop volumineux. Taille maximum: 50 Mo"
            )
            
        # Parse les pages
        page_list = None
        if pages != "all":
            try:
                # Gestion des formats comme "1,3,5-7"
                page_parts = []
                for part in pages.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        page_parts.extend(range(start, end + 1))
                    else:
                        page_parts.append(int(part))
                page_list = page_parts
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail="Format de pages invalide. Utilisez 'all' ou des numéros séparés par des virgules (ex: '1,3,5-7')"
                )
                
        # Initialiser l'extracteur de tableaux s'il n'est pas déjà présent
        if not hasattr(components, 'table_extractor'):
            from core.document_processing.table_extractor import TableExtractor
            components._components["table_extractor"] = TableExtractor()
            logger.info("Extracteur de tableaux initialisé")
            
        # Extraire les tableaux
        tables = await components.table_extractor.extract_tables(
            file_obj,
            pages=page_list or "all",
            extraction_method=extraction_method,
            output_format=output_format
        )
        
        # Si demandé, extraire aussi les images des tableaux
        table_images = []
        if include_images:
            # Retour au début du fichier
            file_obj.seek(0)
            
            # Extraction des images
            table_images = await components.table_extractor.get_tables_as_images(
                file_obj,
                pages=page_list or "all"
            )
            
        # Calcul du temps de traitement
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Extraction terminée en {processing_time:.2f}s: {len(tables)} tableaux trouvés")
        
        # Préparation de la réponse
        response = {
            "filename": file.filename,
            "file_size": file_size,
            "tables_count": len(tables),
            "processing_time": processing_time,
            "tables": tables
        }
        
        # Ajouter les images si demandé
        if include_images:
            response["images"] = table_images
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur extraction tableaux: {e}", exc_info=True)
        metrics.increment_counter("pdf_extraction_errors")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'extraction des tableaux: {str(e)}"
        )
    finally:
        # Nettoyage
        await file.close()
        
@router.post("/analyze-table", response_model=Dict[str, Any])
async def analyze_table_from_pdf(
    file: UploadFile = File(...),
    page: int = Form(1),
    table_index: int = Form(0),
    components=Depends(get_components)
):
    """
    Analyse un tableau spécifique d'un PDF et retourne des statistiques.
    
    Args:
        file: Fichier PDF à traiter
        page: Numéro de page contenant le tableau
        table_index: Index du tableau dans la page (commence à 0)
        
    Returns:
        Statistiques et informations sur le tableau
    """
    try:
        metrics.increment_counter("pdf_table_analysis")
        start_time = datetime.utcnow()
        
        # Vérification du fichier
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Le fichier doit être au format PDF"
            )
            
        # Lecture du fichier en mémoire
        contents = await file.read()
        file_obj = io.BytesIO(contents)
        
        # Initialiser l'extracteur de tableaux si nécessaire
        if not hasattr(components, 'table_extractor'):
            from core.document_processing.table_extractor import TableExtractor
            components._components["table_extractor"] = TableExtractor()
            logger.info("Extracteur de tableaux initialisé")
            
        # Extraction des tableaux de la page spécifiée
        tables = await components.table_extractor.extract_tables(
            file_obj,
            pages=[page],
            extraction_method="auto",
            output_format="pandas"
        )
        
        # Vérification de l'existence du tableau
        if not tables or table_index >= len(tables):
            raise HTTPException(
                status_code=404, 
                detail=f"Aucun tableau à l'index {table_index} trouvé sur la page {page}"
            )
            
        # Récupération du tableau
        table_data = tables[table_index]
        table = table_data["data"]
        
        # Analyse du tableau
        analysis = await _analyze_table(table)
        
        # Ajout des métadonnées
        response = {
            "filename": file.filename,
            "page": page,
            "table_index": table_index,
            "processing_time": (datetime.utcnow() - start_time).total_seconds(),
            "table_rows": len(table),
            "table_columns": len(table.columns),
            "analysis": analysis
        }
        
        # Ajout de l'aperçu du tableau
        preview_rows = min(5, len(table))
        response["preview"] = table.head(preview_rows).to_dict(orient="records")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur analyse tableau: {e}", exc_info=True)
        metrics.increment_counter("pdf_analysis_errors")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse du tableau: {str(e)}"
        )
    finally:
        await file.close()

@router.post("/convert-table", response_model=Dict[str, Any])
async def convert_table_format(
    file: UploadFile = File(...),
    page: int = Form(1),
    table_index: int = Form(0),
    output_format: str = Form("json"),
    components=Depends(get_components)
):
    """
    Convertit un tableau d'un PDF vers un format spécifique.
    
    Args:
        file: Fichier PDF à traiter
        page: Numéro de page contenant le tableau
        table_index: Index du tableau dans la page (commence à 0)
        output_format: Format de sortie désiré ("json", "csv", "html", "excel")
        
    Returns:
        Tableau converti dans le format demandé
    """
    try:
        metrics.increment_counter("pdf_table_conversions")
        
        # Lecture du fichier en mémoire
        contents = await file.read()
        file_obj = io.BytesIO(contents)
        
        # Initialiser l'extracteur de tableaux si nécessaire
        if not hasattr(components, 'table_extractor'):
            from core.document_processing.table_extractor import TableExtractor
            components._components["table_extractor"] = TableExtractor()
        
        # Extraction des tableaux
        tables = await components.table_extractor.extract_tables(
            file_obj,
            pages=[page],
            extraction_method="auto",
            output_format=output_format
        )
        
        # Vérification de l'existence du tableau
        if not tables or table_index >= len(tables):
            raise HTTPException(
                status_code=404, 
                detail=f"Aucun tableau à l'index {table_index} trouvé sur la page {page}"
            )
        
        table_data = tables[table_index]
        
        # Préparer la réponse selon le format demandé
        response = {
            "filename": file.filename,
            "page": page,
            "table_index": table_index,
            "format": output_format,
            "rows": table_data["rows"],
            "columns": table_data["columns"],
            "data": table_data["data"]
        }
        
        # Pour les formats spéciaux, configure le type de réponse approprié
        headers = {}
        if output_format == "csv":
            headers = {"Content-Disposition": f"attachment; filename=table_p{page}_{table_index}.csv"}
        elif output_format == "excel":
            headers = {"Content-Disposition": f"attachment; filename=table_p{page}_{table_index}.xlsx"}
        elif output_format == "html":
            headers = {"Content-Type": "text/html"}
        
        return JSONResponse(content=response, headers=headers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur conversion tableau: {e}", exc_info=True)
        metrics.increment_counter("pdf_conversion_errors")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la conversion du tableau: {str(e)}"
        )
    finally:
        await file.close()

@router.post("/search-tables", response_model=Dict[str, Any])
async def search_in_pdf_tables(
    file: UploadFile = File(...),
    query: str = Form(...),
    pages: str = Form("all"),
    case_sensitive: bool = Form(False),
    components=Depends(get_components)
):
    """
    Recherche une chaîne dans tous les tableaux d'un PDF.
    
    Args:
        file: Fichier PDF à traiter
        query: Texte à rechercher
        pages: Pages à analyser ("all" ou liste de numéros séparés par des virgules)
        case_sensitive: Si True, la recherche est sensible à la casse
        
    Returns:
        Résultats de la recherche
    """
    try:
        metrics.increment_counter("pdf_table_searches")
        
        # Vérification des paramètres
        if not query.strip():
            raise HTTPException(
                status_code=400, 
                detail="La requête de recherche ne peut pas être vide"
            )
        
        # Lecture du fichier en mémoire
        contents = await file.read()
        file_obj = io.BytesIO(contents)
        
        # Parse les pages
        page_list = None
        if pages != "all":
            try:
                # Gestion des formats comme "1,3,5-7"
                page_parts = []
                for part in pages.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        page_parts.extend(range(start, end + 1))
                    else:
                        page_parts.append(int(part))
                page_list = page_parts
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail="Format de pages invalide"
                )
        
        # Initialiser l'extracteur de tableaux si nécessaire
        if not hasattr(components, 'table_extractor'):
            from core.document_processing.table_extractor import TableExtractor
            components._components["table_extractor"] = TableExtractor()
        
        # Extraction des tableaux
        tables = await components.table_extractor.extract_tables(
            file_obj,
            pages=page_list or "all",
            extraction_method="auto",
            output_format="pandas"
        )
        
        # Recherche dans les tableaux
        results = []
        import pandas as pd
        
        for i, table_data in enumerate(tables):
            table = table_data["data"]
            
            # Recherche dans le tableau
            if not case_sensitive:
                # Convertir tout en minuscules pour une recherche insensible à la casse
                mask = table.astype(str).apply(
                    lambda x: x.str.lower().str.contains(query.lower()), axis=0
                )
            else:
                mask = table.astype(str).apply(
                    lambda x: x.str.contains(query), axis=0
                )
            
            # Si des correspondances sont trouvées
            if mask.any().any():
                # Trouver les indices des lignes avec des correspondances
                matching_rows = mask.any(axis=1)
                row_indices = matching_rows[matching_rows].index.tolist()
                
                # Préparation des résultats
                matches = []
                for row_idx in row_indices:
                    # Trouver les colonnes avec des correspondances dans cette ligne
                    cols = mask.iloc[row_idx]
                    matching_cols = cols[cols].index.tolist()
                    
                    # Ajouter chaque cellule correspondante
                    for col in matching_cols:
                        cell_value = str(table.iloc[row_idx][col])
                        matches.append({
                            "row": int(row_idx),
                            "column": str(col),
                            "value": cell_value
                        })
                
                # Ajouter le résultat
                results.append({
                    "table_id": i + 1,
                    "page": table_data.get("page", "unknown"),
                    "matches_count": len(matches),
                    "matches": matches
                })
        
        return {
            "filename": file.filename,
            "query": query,
            "case_sensitive": case_sensitive,
            "tables_scanned": len(tables),
            "matching_tables": len(results),
            "total_matches": sum(r["matches_count"] for r in results),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur recherche tableaux: {e}", exc_info=True)
        metrics.increment_counter("pdf_search_errors")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la recherche dans les tableaux: {str(e)}"
        )
    finally:
        await file.close()

async def _analyze_table(table):
    """Analyse un DataFrame et retourne des statistiques."""
    import pandas as pd
    import numpy as np
    
    try:
        # Statistiques de base
        stats = {
            "row_count": len(table),
            "column_count": len(table.columns),
            "empty_cells": table.isna().sum().sum(),
            "empty_cells_percent": round(table.isna().sum().sum() / (len(table) * len(table.columns)) * 100, 2),
            "columns": []
        }
        
        # Analyse par colonne
        for col in table.columns:
            col_data = table[col]
            col_stats = {
                "name": str(col),
                "dtype": str(col_data.dtype),
                "non_null_count": col_data.count(),
                "null_count": col_data.isna().sum()
            }
            
            # Détecter le type de données
            if pd.api.types.is_numeric_dtype(col_data):
                # Statistiques numériques
                col_stats.update({
                    "data_type": "numeric",
                    "min": col_data.min() if col_data.count() > 0 else None,
                    "max": col_data.max() if col_data.count() > 0 else None,
                    "mean": round(col_data.mean(), 2) if col_data.count() > 0 else None,
                    "median": round(col_data.median(), 2) if col_data.count() > 0 else None
                })
            elif pd.api.types.is_datetime64_dtype(col_data):
                # Statistiques de dates
                col_stats.update({
                    "data_type": "datetime",
                    "min": col_data.min().isoformat() if col_data.count() > 0 else None,
                    "max": col_data.max().isoformat() if col_data.count() > 0 else None
                })
            else:
                # Statistiques de texte
                col_stats.update({
                    "data_type": "text",
                    "unique_values": col_data.nunique(),
                    "most_common": col_data.value_counts().head(3).to_dict() if col_data.count() > 0 else {},
                    "avg_length": round(col_data.astype(str).str.len().mean(), 2) if col_data.count() > 0 else 0
                })
                
            stats["columns"].append(col_stats)
            
        return stats
        
    except Exception as e:
        logger.error(f"Erreur analyse statistique: {e}")
        return {"error": str(e)}