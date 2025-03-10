# api/routes/pdf_routes.py
from fastapi import APIRouter, UploadFile, File, Form, Query, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import io
from datetime import datetime
import uuid
import tempfile
import os
import shutil
import json
import asyncio
import traceback

from ..dependencies import get_components
from ..models.responses import ErrorResponse
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings

# Modèles de données pour les réponses
class TableData(BaseModel):
    table_id: int = Field(..., description="Identifiant unique du tableau")
    page: Optional[int] = Field(None, description="Numéro de page du tableau")
    rows: int = Field(..., description="Nombre de lignes")
    columns: int = Field(..., description="Nombre de colonnes")
    data: Any = Field(..., description="Contenu du tableau dans le format spécifié")
    extraction_method: Optional[str] = Field(None, description="Méthode utilisée pour extraire ce tableau")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score de confiance de l'extraction")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Métadonnées supplémentaires")

class ImageData(BaseModel):
    data: str = Field(..., description="Image encodée en base64")
    mime_type: str = Field(..., description="Type MIME de l'image")
    width: Optional[int] = Field(None, description="Largeur de l'image")
    height: Optional[int] = Field(None, description="Hauteur de l'image")
    page: Optional[int] = Field(None, description="Numéro de page")
    table_id: Optional[int] = Field(None, description="ID du tableau associé")

class SearchMatch(BaseModel):
    row: int = Field(..., description="Numéro de ligne")
    column: str = Field(..., description="Nom de colonne")
    value: str = Field(..., description="Valeur correspondante")

class SearchTableResult(BaseModel):
    table_id: int = Field(..., description="ID du tableau")
    page: Optional[int] = Field(None, description="Numéro de page")
    matches_count: int = Field(..., description="Nombre de correspondances")
    matches: List[SearchMatch] = Field(..., description="Détails des correspondances")

class SearchResult(BaseModel):
    query: str = Field(..., description="Requête de recherche")
    total_matches: int = Field(..., description="Nombre total de correspondances")
    matching_tables: int = Field(..., description="Nombre de tableaux contenant des correspondances")
    results: List[SearchTableResult] = Field(..., description="Résultats par tableau")

class TableExtractionResponse(BaseModel):
    extraction_id: str = Field(..., description="Identifiant unique de l'extraction")
    filename: str = Field(..., description="Nom du fichier traité")
    file_size: int = Field(..., description="Taille du fichier en octets")
    tables_count: int = Field(..., description="Nombre de tableaux extraits")
    processing_time: float = Field(..., description="Temps de traitement en secondes")
    tables: List[Dict[str, Any]] = Field(default=[], description="Tableaux extraits")
    extraction_method_used: str = Field(..., description="Méthode d'extraction utilisée")
    ocr_used: bool = Field(default=False, description="OCR utilisé pour l'extraction")
    images: Optional[List[Dict[str, Any]]] = Field(None, description="Images des tableaux extraits")
    search_results: Optional[SearchResult] = Field(None, description="Résultats de recherche")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Analyse des tableaux")
    status: str = Field(default="completed", description="Statut de l'extraction")
    message: Optional[str] = Field(None, description="Message d'information")

    model_config = {
        "json_schema_extra": {
            "example": {
                "extraction_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "filename": "document.pdf",
                "file_size": 1256437,
                "tables_count": 3,
                "processing_time": 2.45,
                "tables": [
                    {
                        "table_id": 1,
                        "page": 1,
                        "rows": 5,
                        "columns": 3,
                        "data": [
                            {"col1": "val1", "col2": "val2", "col3": "val3"},
                            {"col1": "val4", "col2": "val5", "col3": "val6"}
                        ],
                        "extraction_method": "auto",
                        "confidence": 0.95
                    }
                ],
                "extraction_method_used": "auto",
                "ocr_used": False,
                "status": "completed"
            }
        }
    }

class TaskStatus(BaseModel):
    task_id: str = Field(..., description="Identifiant de la tâche")
    status: str = Field(..., description="Statut de la tâche (processing, completed, error)")
    message: str = Field(..., description="Message informatif")
    results: Optional[Dict[str, Any]] = Field(None, description="Résultats de la tâche si terminée")
    progress: Optional[float] = Field(None, description="Pourcentage de progression (0-100)")

logger = get_logger("pdf_routes")
router = APIRouter(prefix="/pdf", tags=["pdf"])

@router.post("/extract-tables-auto", response_model=Union[TableExtractionResponse, Dict[str, Any]])
async def extract_tables_auto(
    file: UploadFile = File(...),
    # Format de sortie pour les données dans la réponse JSON
    output_format: str = Form("json", description="Format des données dans la réponse JSON (json, pandas, csv, html)"),
    pages: str = Form("all", description="Pages à analyser (all, 1,3,5-7, etc.)"),
    ocr_enabled: bool = Form(False, description="Activer l'OCR pour les PDF scannés"),
    ocr_language: str = Form("fra+eng", description="Langues pour l'OCR (fra+eng, eng, etc.)"),
    ocr_enhance_image: bool = Form(True, description="Améliorer l'image avant OCR"),
    ocr_deskew: bool = Form(True, description="Redresser l'image avant OCR"),
    include_images: bool = Form(False, description="Inclure les images des tableaux extraits"),
    analyze: bool = Form(False, description="Effectuer une analyse des tableaux"),
    search_query: Optional[str] = Form(None, description="Texte à rechercher dans les tableaux"),
    force_grid: bool = Form(False, description="Forcer l'approche grille pour l'OCR"),
    # Format d'export pour le téléchargement
    download_format: Optional[str] = Form(None, description="Format à télécharger (csv, excel, json, html)"),
    background_processing: bool = Form(False, description="Traiter en arrière-plan (pour les grands PDF)"),
    auto_ocr: bool = Form(True, description="Activer automatiquement l'OCR si nécessaire"),
    components=Depends(get_components)
):
    """
    Endpoint unifié pour extraire, analyser et rechercher des tableaux dans les PDFs.
    
    Cette route utilise la détection automatique des méthodes d'extraction optimales
    en fonction du type de PDF (scanné, texte) et de son contenu.
    
    Args:
        file: Fichier PDF à traiter
        output_format: Format des données dans la réponse JSON ("json", "csv", "html", "pandas")
        pages: Pages à analyser ("all" ou numéros séparés par des virgules comme "1,3,5-7")
        ocr_enabled: Force l'OCR même pour les PDFs textuels
        ocr_language: Langues pour l'OCR (ex: "fra+eng" pour français et anglais)
        ocr_enhance_image: Applique l'amélioration d'image pour l'OCR
        ocr_deskew: Corrige l'inclinaison de l'image avant l'OCR
        include_images: Inclut les images des tableaux dans la réponse
        analyze: Effectue une analyse détaillée des tableaux extraits
        search_query: Recherche du texte dans les tableaux
        force_grid: Force l'extraction basée sur une grille pour l'OCR
        download_format: Format d'export pour le téléchargement (remplace export_format)
        background_processing: Traitement en arrière-plan pour les PDF volumineux
        auto_ocr: Activer automatiquement l'OCR si aucun tableau n'est trouvé avec les méthodes textuelles
        
    Returns:
        Tableaux extraits avec analyse ou résultats de recherche optionnels
    """
    try:
        # Mesure des performances
        extraction_id = str(uuid.uuid4())
        metrics.start_request_tracking(extraction_id)
        metrics.increment_counter("pdf_table_extractions")
        start_time = datetime.utcnow()
        
        # Vérification du fichier
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Le fichier doit être au format PDF"
            )
        
        # Vérification de la taille du fichier
        file_content = await file.read()
        file_size = len(file_content)
        file_obj = io.BytesIO(file_content)
        
        max_size = 50 * 1024 * 1024  # 50 Mo
        if file_size > max_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Fichier trop volumineux. Taille maximum: 50 Mo"
            )
        
        # Vérification des paramètres
        valid_formats = ["json", "csv", "html", "excel", "pandas"]
        if output_format not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Format de sortie invalide. Formats supportés: {', '.join(valid_formats)}"
            )
        
        # Validation du format de téléchargement
        valid_download_formats = [None, "csv", "excel", "json", "html"]
        if download_format not in valid_download_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Format de téléchargement invalide. Formats supportés: {', '.join([f for f in valid_download_formats if f])}"
            )
        
        # Ensuite, utilisez download_format au lieu de export_format dans le reste du code
        
        # Vérification et traitement d'une demande de traitement en arrière-plan
        if background_processing and file_size > 5 * 1024 * 1024:  # > 5 Mo
            # Enregistrer le fichier temporaire et démarrer le traitement en arrière-plan
            task_id = str(uuid.uuid4())
            temp_dir = os.path.join("temp", "pdf_tasks")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"{task_id}.pdf")
            
            with open(temp_path, "wb") as f:
                f.write(file_content)
            
            # Lancer la tâche en arrière-plan
            background_tasks = BackgroundTasks()
            background_tasks.add_task(
                process_pdf_in_background,
                task_id=task_id,
                file_path=temp_path,
                original_filename=file.filename,
                params={
                    "output_format": output_format,
                    "pages": pages,
                    "ocr_enabled": ocr_enabled,
                    "ocr_language": ocr_language,
                    "ocr_enhance_image": ocr_enhance_image,
                    "ocr_deskew": ocr_deskew,
                    "include_images": include_images,
                    "analyze": analyze,
                    "search_query": search_query,
                    "force_grid": force_grid,
                    "auto_ocr": auto_ocr
                }
            )
            
            return JSONResponse(
                content={
                    "task_id": task_id,
                    "status": "processing",
                    "message": "Traitement démarré en arrière-plan, veuillez vérifier le statut ultérieurement",
                    "check_status_url": f"/api/pdf/task-status/{task_id}",
                    "extraction_id": extraction_id,
                    "filename": file.filename,
                    "file_size": file_size
                },
                background=background_tasks
            )
        
        # Vérification du cache
        cache_key = None
        if hasattr(components, 'cache_manager'):
            import hashlib
            # Créer une clé unique basée sur le contenu du fichier et les paramètres
            params_str = f"{pages}:{ocr_enabled}:{ocr_language}:{output_format}:{force_grid}"
            file_hash = hashlib.md5(file_content).hexdigest()
            cache_key = f"pdf:tables:{file_hash}:{params_str}"
            
            # Vérifier si les résultats sont en cache
            cached_results = await components.cache_manager.get(cache_key)
            if cached_results:
                logger.info(f"Résultats d'extraction trouvés en cache: {file.filename}")
                metrics.track_cache_operation(hit=True)
                metrics.finish_request_tracking(extraction_id)
                return json.loads(cached_results)
            else:
                metrics.track_cache_operation(hit=False)
        
        # Initialiser l'extracteur de tableaux s'il n'est pas déjà présent
        if not hasattr(components, 'table_extractor'):
            from core.document_processing.table_extractor import TableExtractor
            
            # Vérifier si le détecteur de tableaux est disponible
            table_detector = None
            if hasattr(components, 'table_detector'):
                table_detector = components.table_detector
            
            components._components["table_extractor"] = TableExtractor(
                cache_enabled=True,
                table_detector=table_detector
            )
            logger.info("Extracteur de tableaux initialisé")
        
        # Configuration OCR si activée
        ocr_config = None
        if ocr_enabled:
            ocr_config = {
                "lang": ocr_language,
                "enhance_image": ocr_enhance_image,
                "deskew": ocr_deskew,
                "preprocess_type": "thresh",
                "psm": 6,
                "force_grid": force_grid
            }
            
        # Détection si le PDF est scanné
        is_scanned = False
        if auto_ocr and not ocr_enabled:
            is_scanned, confidence = await components.table_extractor._is_scanned_pdf(file_obj, pages)
            logger.info(f"Détection automatique du type de PDF: {'scanné' if is_scanned else 'textuel'} (confiance: {confidence:.2f})")
            
            # Activer automatiquement l'OCR si le PDF est scanné
            if is_scanned and confidence > 0.7:
                ocr_enabled = True
                ocr_config = {
                    "lang": ocr_language,
                    "enhance_image": ocr_enhance_image,
                    "deskew": ocr_deskew,
                    "preprocess_type": "thresh",
                    "psm": 6,
                    "force_grid": force_grid
                }
                logger.info("OCR activé automatiquement pour PDF scanné")
                
        # Remise du curseur au début du fichier si nécessaire
        file_obj.seek(0)
        
        # Extraire les tableaux avec la méthode automatique
        tables = await components.table_extractor.extract_tables(
            file_obj,
            pages=pages,
            extraction_method="auto",
            output_format=output_format,
            ocr_config=ocr_config
        )
        
        # Essayer avec OCR si aucun tableau trouvé et auto_ocr activé
        if not tables and auto_ocr and not ocr_enabled:
            logger.info("Aucun tableau trouvé, nouvel essai avec OCR")
            ocr_config = {
                "lang": ocr_language,
                "enhance_image": ocr_enhance_image,
                "deskew": ocr_deskew,
                "preprocess_type": "thresh",
                "psm": 6,
                "force_grid": force_grid
            }
            file_obj.seek(0)
            tables = await components.table_extractor.extract_tables(
                file_obj,
                pages=pages,
                extraction_method="ocr",
                output_format=output_format,
                ocr_config=ocr_config
            )
            ocr_enabled = True
        
        # Si demandé, extraire aussi les images des tableaux
        table_images = []
        if include_images and tables:
            # Retour au début du fichier
            file_obj.seek(0)
            
            # Extraction des images
            table_images = await components.table_extractor.get_tables_as_images(
                file_obj,
                pages=pages
            )
        
        # Si search_query est spécifié, filtrer les tableaux
        search_results = None
        if search_query and search_query.strip() and tables:
            search_results = await search_in_tables(tables, search_query)
        
        # Si analyze est True, effectuer une analyse détaillée
        analysis_results = None
        if analyze and tables:
            if output_format == "pandas":
                analysis_results = analyze_pandas_tables([table["data"] for table in tables if "data" in table])
            else:
                # Pour les formats autres que pandas, convertir d'abord
                analysis_results = {"error": "L'analyse détaillée nécessite output_format='pandas'"}
        
        # Si download_format est spécifié, générer le fichier à télécharger
        if download_format and tables:
            export_result = await export_tables(tables, download_format, file.filename)
            if export_result.get("download_url"):
                return StreamingResponse(
                    io.BytesIO(export_result["content"]),
                    media_type=export_result["media_type"],
                    headers={"Content-Disposition": f"attachment; filename={export_result['filename']}"}
                )
        
        # Calcul du temps de traitement
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Extraction de la méthode utilisée
        extraction_method_used = "auto"
        for table in tables:
            if table.get("extraction_method"):
                extraction_method_used = table.get("extraction_method")
                break
        
        # Préparation de la réponse
        response = TableExtractionResponse(
            extraction_id=extraction_id,
            filename=file.filename,
            file_size=file_size,
            tables_count=len(tables),
            processing_time=processing_time,
            tables=tables,
            extraction_method_used=extraction_method_used,
            ocr_used=ocr_enabled,
            images=table_images if include_images else None,
            search_results=search_results,
            analysis=analysis_results,
            status="completed",
            message="Extraction réussie"
        )
        
        # Message informatif si PDF scanné mais OCR non activé initialement
        if is_scanned and not tables and not ocr_enabled:
            response.message = "Aucun tableau trouvé. Ce document semble être scanné, l'activation de l'OCR pourrait donner de meilleurs résultats."
        
        # Mettre en cache si possible
        if cache_key and hasattr(components, 'cache_manager'):
            # Stocker en cache pendant 1 heure (3600 secondes)
            await components.cache_manager.set(cache_key, json.dumps(response.model_dump()), 3600)
        
        # Enregistrer les métriques
        metrics.finish_request_tracking(extraction_id)
        try:
            # S'assurer que la méthode d'extraction est une chaîne de caractères
            method_str = str(extraction_method_used) if isinstance(extraction_method_used, (str, float, int)) else "auto"
            metrics.track_search_operation(
                method=method_str,
                success=len(tables) > 0,
                processing_time=processing_time,
                results_count=len(tables)
            )
        except Exception as e:
            logger.error(f"Erreur tracking métriques: {e}", exc_info=True)
        
        return response
        
    except HTTPException:
        metrics.increment_counter("pdf_extraction_errors")
        raise
    except Exception as e:
        # Enregistrer l'erreur complète
        logger.error(f"Erreur extraction tableaux: {str(e)}", exc_info=True)
        metrics.increment_counter("pdf_extraction_errors")
        
        # Calculer le temps de traitement même en cas d'erreur
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        metrics.finish_request_tracking(extraction_id)
        
        # S'assurer que la méthode est une chaîne
        extraction_method_str = "auto"
        
        # Tenter d'enregistrer les métriques de recherche même en cas d'erreur
        try:
            metrics.track_search_operation(
                method=extraction_method_str,
                success=False,
                processing_time=processing_time,
                results_count=0
            )
        except Exception as metric_error:
            logger.error(f"Erreur tracking métriques: {metric_error}")
        
        # Retourner une réponse d'erreur structurée
        return TableExtractionResponse(
            extraction_id=extraction_id,
            filename=file.filename if file else "unknown",
            file_size=file_size if 'file_size' in locals() else 0,
            tables_count=0,
            processing_time=processing_time,
            tables=[],
            extraction_method_used="auto",
            ocr_used=ocr_enabled,
            status="error",
            message=f"Erreur lors de l'extraction des tableaux: {str(e)}"
        )
    finally:
        # Nettoyage
        await file.close()

@router.get("/task-status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    Vérifie le statut d'une tâche d'extraction en arrière-plan.
    
    Args:
        task_id: ID de la tâche
        
    Returns:
        Statut de la tâche et résultats si disponibles
    """
    try:
        result_path = os.path.join("temp", "pdf_tasks", f"{task_id}_result.json")
        progress_path = os.path.join("temp", "pdf_tasks", f"{task_id}_progress.json")
        
        # Vérifier si la tâche est terminée
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)
            return TaskStatus(
                task_id=task_id,
                status=result.get("status", "completed"),
                message=result.get("message", "Traitement terminé"),
                results=result,
                progress=100.0
            )
        
        # Vérifier la progression si disponible
        if os.path.exists(progress_path):
            with open(progress_path, "r") as f:
                progress_data = json.load(f)
            return TaskStatus(
                task_id=task_id,
                status="processing",
                message=progress_data.get("message", "Traitement en cours"),
                progress=progress_data.get("progress", 0.0)
            )
        
        # Vérifier si la tâche est en cours
        if os.path.exists(os.path.join("temp", "pdf_tasks", f"{task_id}.pdf")):
            return TaskStatus(
                task_id=task_id,
                status="processing",
                message="Traitement en cours",
                progress=0.0
            )
        
        return TaskStatus(
            task_id=task_id,
            status="not_found",
            message="Tâche non trouvée",
            progress=None
        )
    except Exception as e:
        logger.error(f"Erreur vérification statut tâche: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la vérification du statut: {str(e)}"
        )

@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    Télécharge un fichier généré.
    
    Args:
        filename: Nom du fichier à télécharger
        
    Returns:
        Fichier à télécharger
    """
    try:
        file_path = os.path.join("temp", "pdf_exports", filename)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Fichier non trouvé: {filename}"
            )
        
        # Déterminer le type MIME
        media_type = "application/octet-stream"
        if filename.endswith(".xlsx"):
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif filename.endswith(".csv"):
            media_type = "text/csv"
        elif filename.endswith(".json"):
            media_type = "application/json"
        elif filename.endswith(".html"):
            media_type = "text/html"
        elif filename.endswith(".zip"):
            media_type = "application/zip"
        
        # Lire le fichier et le retourner
        with open(file_path, "rb") as f:
            content = f.read()
        
        return StreamingResponse(
            io.BytesIO(content),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur téléchargement fichier: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du téléchargement: {str(e)}"
        )

# Fonctions auxiliaires

async def search_in_tables(tables: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """
    Recherche un texte dans les tableaux.
    
    Args:
        tables: Liste de tableaux
        query: Texte à rechercher
        
    Returns:
        Résultats de recherche
    """
    results = []
    total_matches = 0
    
    for i, table in enumerate(tables):
        if "data" not in table:
            continue
            
        data = table["data"]
        matches = []
        
        # Recherche dans les données du tableau
        if isinstance(data, list):
            # Pour les formats JSON/dict
            for row_idx, row in enumerate(data):
                for col_key, cell_value in row.items():
                    if isinstance(cell_value, (str, int, float)) and query.lower() in str(cell_value).lower():
                        matches.append({
                            "row": row_idx,
                            "column": col_key,
                            "value": str(cell_value)
                        })
        elif hasattr(data, 'astype') and hasattr(data, 'apply'):
            # Pour les pandas DataFrames
            import pandas as pd
            mask = data.astype(str).apply(
                lambda x: x.str.lower().str.contains(query.lower()), axis=0
            )
            
            for row_idx, row in enumerate(mask.values):
                for col_idx, match in enumerate(row):
                    if match:
                        cell_value = data.iloc[row_idx, col_idx]
                        matches.append({
                            "row": row_idx,
                            "column": str(data.columns[col_idx]),
                            "value": str(cell_value)
                        })
        
        if matches:
            results.append({
                "table_id": i + 1,
                "page": table.get("page", "unknown"),
                "matches_count": len(matches),
                "matches": matches
            })
            total_matches += len(matches)
    
    return SearchResult(
        query=query,
        total_matches=total_matches,
        matching_tables=len(results),
        results=results
    ).model_dump()

async def export_tables(tables: List[Dict[str, Any]], format: str, filename: str) -> Dict[str, Any]:
    """
    Exporte les tableaux dans un format spécifique.
    
    Args:
        tables: Liste de tableaux
        format: Format d'export ('csv', 'excel', 'json', 'html')
        filename: Nom du fichier original
        
    Returns:
        Informations sur le fichier exporté
    """
    try:
        import pandas as pd
        import numpy as np
        from io import BytesIO
        
        # Créer le répertoire d'export s'il n'existe pas
        export_dir = os.path.join("temp", "pdf_exports")
        os.makedirs(export_dir, exist_ok=True)
        
        # Fonction pour convertir les objets non-JSON-sérialisables
        def json_serializable(obj):
            if isinstance(obj, (pd.Series, pd.Index)):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            else:
                return str(obj)
        
        if format == "excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for i, table_data in enumerate(tables):
                    if "data" not in table_data:
                        continue
                        
                    # Convertir en DataFrame si nécessaire
                    if isinstance(table_data["data"], pd.DataFrame):
                        df = table_data["data"]
                    else:
                        df = pd.DataFrame(table_data["data"])
                    
                    # Remplacer les NaN par None pour éviter les problèmes d'exportation
                    df = df.replace({np.nan: None})
                    
                    sheet_name = f"Table_{i+1}"
                    if len(sheet_name) > 31:  # Excel limite le nom des feuilles à 31 caractères
                        sheet_name = sheet_name[:31]
                    
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Ajuster la largeur des colonnes
                    worksheet = writer.sheets[sheet_name]
                    for j, col in enumerate(df.columns):
                        # Définir la largeur en fonction du contenu
                        max_len = max(
                            df[col].astype(str).map(len).max() if len(df) > 0 else 0,
                            len(str(col))
                        )
                        # Ajouter un peu d'espace
                        worksheet.set_column(j, j, max_len + 2)
            
            output.seek(0)
            export_filename = f"{os.path.splitext(filename)[0]}_tables.xlsx"
            export_path = os.path.join(export_dir, export_filename)
            
            # Sauvegarder le fichier
            with open(export_path, "wb") as f:
                f.write(output.getvalue())
            
            return {
                "content": output.getvalue(),
                "filename": export_filename,
                "media_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "download_url": f"/api/pdf/download/{export_filename}"
            }
        elif format == "csv":
            import zipfile
            
            output = BytesIO()
            with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for i, table_data in enumerate(tables):
                    if "data" not in table_data:
                        continue
                        
                    # Convertir en DataFrame si nécessaire
                    if isinstance(table_data["data"], pd.DataFrame):
                        df = table_data["data"]
                    else:
                        df = pd.DataFrame(table_data["data"])
                    
                    # Remplacer les NaN par None
                    df = df.replace({np.nan: None})
                    
                    # Créer un CSV en mémoire
                    csv_buffer = BytesIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    # Ajouter au ZIP
                    zipf.writestr(f"table_{i+1}.csv", csv_buffer.getvalue())
            
            output.seek(0)
            export_filename = f"{os.path.splitext(filename)[0]}_tables.zip"
            export_path = os.path.join(export_dir, export_filename)
            
            # Sauvegarder le fichier
            with open(export_path, "wb") as f:
                f.write(output.getvalue())
            
            return {
                "content": output.getvalue(),
                "filename": export_filename,
                "media_type": "application/zip",
                "download_url": f"/api/pdf/download/{export_filename}"
            }
        elif format == "json":
            output = BytesIO()
            result = []
            
            for i, table_data in enumerate(tables):
                if "data" not in table_data:
                    continue
                    
                # Préparation des données pour la sérialisation JSON
                table_result = {
                    "table_id": i + 1,
                    "page": table_data.get("page", i + 1),
                }
                
                # Convertir en DataFrame si nécessaire
                if isinstance(table_data["data"], pd.DataFrame):
                    df = table_data["data"]
                    # Convertir le DataFrame en dictionnaire en remplaçant NaN par None
                    records = df.replace({np.nan: None}).to_dict(orient="records")
                    table_result["rows"] = len(records)
                    table_result["columns"] = len(df.columns)
                    table_result["data"] = records
                else:
                    # Déjà sous forme de liste ou dictionnaire
                    data = table_data["data"]
                    table_result["rows"] = len(data) if isinstance(data, list) else 0
                    table_result["columns"] = len(data[0]) if isinstance(data, list) and len(data) > 0 else 0
                    table_result["data"] = data
                
                result.append(table_result)
            
            # Utiliser json.dumps avec un convertisseur personnalisé
            import json
            json_str = json.dumps(result, default=json_serializable, indent=2)
            output.write(json_str.encode('utf-8'))
            output.seek(0)
            export_filename = f"{os.path.splitext(filename)[0]}_tables.json"
            export_path = os.path.join(export_dir, export_filename)
            
            # Sauvegarder le fichier
            with open(export_path, "wb") as f:
                f.write(output.getvalue())
            
            return {
                "content": output.getvalue(),
                "filename": export_filename,
                "media_type": "application/json",
                "download_url": f"/api/pdf/download/{export_filename}"
            }
        elif format == "html":
            output = BytesIO()
            html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Tables Export</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        h2 { color: #333; }
    </style>
</head>
<body>
"""
            
            for i, table_data in enumerate(tables):
                if "data" not in table_data:
                    continue
                    
                # Convertir en DataFrame si nécessaire
                if isinstance(table_data["data"], pd.DataFrame):
                    df = table_data["data"]
                    # Remplacer les NaN par des chaînes vides pour l'affichage HTML
                    df = df.fillna("")
                    table_html = df.to_html(index=False, classes='data-table', border=1)
                else:
                    df = pd.DataFrame(table_data["data"])
                    df = df.fillna("")
                    table_html = df.to_html(index=False, classes='data-table', border=1)
                
                html_content += f"<h2>Tableau {i+1} (Page {table_data.get('page', 'N/A')})</h2>{table_html}"
            
            html_content += "</body></html>"
            output.write(html_content.encode('utf-8'))
            output.seek(0)
            export_filename = f"{os.path.splitext(filename)[0]}_tables.html"
            export_path = os.path.join(export_dir, export_filename)
            
            # Sauvegarder le fichier
            with open(export_path, "wb") as f:
                f.write(output.getvalue())
            
            return {
                "content": output.getvalue(),
                "filename": export_filename,
                "media_type": "text/html",
                "download_url": f"/api/pdf/download/{export_filename}"
            }
        else:
            raise ValueError(f"Format non supporté: {format}")
    
    except Exception as e:
        logger.error(f"Erreur export: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'export: {str(e)}"
        )

# Fonction d'analyse de tableaux pandas
def analyze_pandas_tables(tables: List[Any]) -> Dict[str, Any]:
    """
    Analyse une liste de tables pandas pour produire des statistiques.
    
    Args:
        tables: Liste de DataFrames pandas
        
    Returns:
        Dictionnaire contenant des statistiques et métriques sur les tableaux
    """
    try:
        import pandas as pd
        import numpy as np
        
        if not tables or len(tables) == 0:
            return {"error": "Aucun tableau à analyser"}
        
        result = {
            "table_count": len(tables),
            "tables": []
        }
        
        total_rows = 0
        total_cols = 0
        total_cells = 0
        non_null_cells = 0
        
        for i, df in enumerate(tables):
            if not isinstance(df, pd.DataFrame):
                continue
                
            # Statistiques par tableau
            rows = len(df)
            cols = len(df.columns)
            cells = rows * cols
            non_null = df.count().sum()
            null_percentage = ((cells - non_null) / cells * 100) if cells > 0 else 0
            
            # Types de données
            dtypes = df.dtypes.astype(str).to_dict()
            
            # Détection de colonnes numériques
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            # Statistiques numériques de base
            numeric_stats = {}
            for col in numeric_cols:
                numeric_stats[str(col)] = {
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else None
                }
            
            # Ajouter aux totaux
            total_rows += rows
            total_cols += cols
            total_cells += cells
            non_null_cells += non_null
            
            # Résultats pour ce tableau
            table_result = {
                "table_id": i + 1,
                "rows": rows,
                "columns": cols,
                "column_names": list(df.columns),
                "null_percentage": round(null_percentage, 2),
                "dtypes": {str(k): v for k, v in dtypes.items()},
                "numeric_columns": numeric_cols,
                "numeric_columns_count": len(numeric_cols),
                "numeric_stats": numeric_stats
            }
            
            result["tables"].append(table_result)
        
        # Statistiques globales
        result["total_rows"] = total_rows
        result["total_columns"] = total_cols
        result["average_row_count"] = round(total_rows / len(tables), 2) if len(tables) > 0 else 0
        result["total_cells"] = total_cells
        result["null_percentage"] = round(((total_cells - non_null_cells) / total_cells * 100), 2) if total_cells > 0 else 0
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur analyse tableaux: {e}")
        return {"error": str(e)}

async def process_pdf_in_background(
    task_id: str, 
    file_path: str, 
    original_filename: str,
    params: Dict[str, Any]
):
    """
    Traite un PDF en arrière-plan.
    
    Args:
        task_id: ID de la tâche
        file_path: Chemin du fichier PDF
        original_filename: Nom original du fichier
        params: Paramètres d'extraction
    """
    try:
        from core.document_processing.table_extractor import TableExtractor
        from core.document_processing.table_detection import TableDetectionModel
        
        # Créer le fichier de progression
        progress_path = os.path.join("temp", "pdf_tasks", f"{task_id}_progress.json")
        
        # Fonction pour mettre à jour la progression
        def update_progress(percent, message):
            with open(progress_path, "w") as f:
                json.dump({
                    "progress": percent,
                    "message": message
                }, f)
        
        # Mettre à jour l'état initial
        update_progress(0, "Initialisation de l'extraction")
        
        # Créer un détecteur de tableaux si possible
        table_detector = None
        try:
            table_detector = TableDetectionModel()
            await table_detector.initialize()
        except Exception as e:
            logger.warning(f"Impossible d'initialiser le détecteur de tableaux: {e}")
        
        # Créer un nouvel extracteur pour cette tâche
        extractor = TableExtractor(table_detector=table_detector)
        
        # Mise à jour de la progression
        update_progress(10, "Préparation du traitement")
        
        # Configuration OCR si activée
        ocr_config = None
        if params.get("ocr_enabled"):
            ocr_config = {
                "lang": params.get("ocr_language", "fra+eng"),
                "enhance_image": params.get("ocr_enhance_image", True),
                "deskew": params.get("ocr_deskew", True),
                "preprocess_type": "thresh",
                "psm": 6,
                "force_grid": params.get("force_grid", False)
            }
        
        start_time = datetime.utcnow()
        
        # Mise à jour de la progression
        update_progress(20, "Extraction des tableaux")
        
        # Extraire les tableaux avec la méthode automatique
        with open(file_path, 'rb') as f:
            file_size = os.path.getsize(file_path)
            file_obj = io.BytesIO(f.read())
        
        # Extraction des tableaux
        tables = await extractor.extract_tables(
            file_obj,
            pages=params.get("pages", "all"),
            extraction_method="auto",
            output_format=params.get("output_format", "json"),
            ocr_config=ocr_config
        )
        
        # Mise à jour de la progression
        update_progress(70, "Préparation des résultats")
        
        # Si demandé, extraire aussi les images des tableaux
        table_images = []
        if params.get("include_images") and tables:
            update_progress(75, "Extraction des images")
            file_obj.seek(0)
            table_images = await extractor.get_tables_as_images(
                file_obj,
                pages=params.get("pages", "all")
            )
        
        # Si search_query est spécifié, filtrer les tableaux
        search_results = None
        if params.get("search_query") and tables:
            update_progress(85, "Recherche dans les tableaux")
            search_results = await search_in_tables(tables, params["search_query"])
        
        # Si analyze est True, effectuer une analyse détaillée
        analysis_results = None
        if params.get("analyze") and tables:
            update_progress(90, "Analyse des tableaux")
            if params.get("output_format") == "pandas":
                analysis_results = analyze_pandas_tables([table["data"] for table in tables if "data" in table])
            else:
                analysis_results = {"error": "L'analyse détaillée nécessite output_format='pandas'"}
        
        # Calcul du temps de traitement
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Mise à jour de la progression finale
        update_progress(95, "Finalisation des résultats")
        
        # Extraction de la méthode utilisée
        extraction_method_used = "auto"
        for table in tables:
            if table.get("extraction_method"):
                extraction_method_used = table.get("extraction_method")
                break
        
        # Préparation du résultat
        result = {
            "extraction_id": task_id,
            "filename": original_filename,
            "file_size": file_size,
            "tables_count": len(tables),
            "processing_time": processing_time,
            "tables": tables,
            "extraction_method_used": extraction_method_used,
            "ocr_used": params.get("ocr_enabled", False),
            "status": "completed",
            "message": "Extraction réussie"
        }
        
        if params.get("include_images"):
            result["images"] = table_images
        
        if search_results:
            result["search_results"] = search_results
        
        if analysis_results:
            result["analysis"] = analysis_results
        
        # Sauvegarder le résultat
        with open(os.path.join("temp", "pdf_tasks", f"{task_id}_result.json"), "w") as f:
            json.dump(result, f)
        
        # Mise à jour finale
        update_progress(100, "Traitement terminé")
        
        # Nettoyer le fichier PDF temporaire
        os.remove(file_path)
        
        # Nettoyer le détecteur de tableaux si créé
        if table_detector:
            await table_detector.cleanup()
        
    except Exception as e:
        logger.error(f"Erreur traitement en arrière-plan: {e}", exc_info=True)
        # Sauvegarder l'erreur
        with open(os.path.join("temp", "pdf_tasks", f"{task_id}_result.json"), "w") as f:
            json.dump({
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "message": f"Erreur lors du traitement: {str(e)}"
            }, f)
        
        # Nettoyer le fichier PDF temporaire
        if os.path.exists(file_path):
            os.remove(file_path)