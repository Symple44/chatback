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
    status: Optional[str] = Field(None, description="Statut de l'extraction")
    message: Optional[str] = Field(None, description="Message d'information")

    class Config:
        schema_extra = {
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
                "ocr_used": False
            }
        }

logger = get_logger("pdf_routes")
router = APIRouter(prefix="/pdf", tags=["pdf"])

@router.post("/extract-tables-auto", response_model=TableExtractionResponse)
async def extract_tables_auto(
    file: UploadFile = File(...),
    output_format: str = Form("json"),
    pages: str = Form("all"),
    ocr_enabled: bool = Form(False),
    ocr_language: str = Form("fra+eng"),
    ocr_enhance_image: bool = Form(True),
    ocr_deskew: bool = Form(True),
    include_images: bool = Form(False),
    analyze: bool = Form(False),
    search_query: Optional[str] = Form(None),
    force_grid: bool = Form(False),
    export_format: Optional[str] = Form(None),
    background_processing: bool = Form(False),
    components=Depends(get_components)
):
    """
    Endpoint unifié pour extraire, analyser et rechercher des tableaux dans les PDFs.
    
    Cette route utilise la détection automatique des méthodes d'extraction optimales
    en fonction du type de PDF (scanné, texte) et de son contenu.
    
    Args:
        file: Fichier PDF à traiter
        output_format: Format de sortie ("json", "csv", "html", "excel", "pandas")
        pages: Pages à analyser ("all" ou numéros séparés par des virgules comme "1,3,5-7")
        ocr_enabled: Force l'OCR même pour les PDFs textuels
        ocr_language: Langues pour l'OCR (ex: "fra+eng" pour français et anglais)
        ocr_enhance_image: Applique l'amélioration d'image pour l'OCR
        ocr_deskew: Corrige l'inclinaison de l'image avant l'OCR
        include_images: Inclut les images des tableaux dans la réponse
        analyze: Effectue une analyse détaillée des tableaux extraits
        search_query: Recherche du texte dans les tableaux
        force_grid: Force l'extraction basée sur une grille pour l'OCR
        export_format: Format d'export optionnel (génère un fichier à télécharger)
        background_processing: Traitement en arrière-plan pour les PDF volumineux
        
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
        
        # Vérification et traitement d'une demande de traitement en arrière-plan
        if background_processing and file_size > 10 * 1024 * 1024:  # > 10 Mo
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
                    "force_grid": force_grid
                }
            )
            
            return JSONResponse(
                content={
                    "task_id": task_id,
                    "status": "processing",
                    "message": "Traitement démarré en arrière-plan, veuillez vérifier le statut ultérieurement",
                    "check_status_url": f"/api/pdf/task-status/{task_id}"
                },
                background=background_tasks
            )
        
        # Vérification du cache
        cache_key = None
        if hasattr(components, 'cache'):
            import hashlib
            # Créer une clé unique basée sur le contenu du fichier et les paramètres
            params_str = f"{pages}:{ocr_enabled}:{ocr_language}:{output_format}:{force_grid}"
            file_hash = hashlib.md5(file_content).hexdigest()
            cache_key = f"pdf:tables:{file_hash}:{params_str}"
            
            # Vérifier si les résultats sont en cache
            cached_results = await components.cache.get(cache_key)
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
            components._components["table_extractor"] = TableExtractor()
            logger.info("Extracteur de tableaux initialisé")
        
        # Configuration OCR si activée
        ocr_config = None
        if ocr_enabled:
            ocr_config = {
                "lang": ocr_language,
                "enhance_image": ocr_enhance_image,
                "deskew": ocr_deskew,
                "preprocess_type": "thresh",
                "force_grid": force_grid
            }
        
        # Utiliser la méthode automatique d'extraction
        tables = await components.table_extractor.extract_tables_auto(
            file_obj,
            output_format=output_format,
            max_pages=None if pages == "all" else len(pages.split(",")),
            ocr_config=ocr_config,
        )
        
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
        
        # Si export_format est spécifié, générer le fichier d'export
        if export_format and tables:
            export_result = await export_tables(tables, export_format, file.filename)
            if export_result.get("download_url"):
                return StreamingResponse(
                    io.BytesIO(export_result["content"]),
                    media_type=export_result["media_type"],
                    headers={"Content-Disposition": f"attachment; filename={export_result['filename']}"}
                )
        
        # Calcul du temps de traitement
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Préparation de la réponse
        response = {
            "extraction_id": extraction_id,
            "filename": file.filename,
            "file_size": file_size,
            "tables_count": len(tables),
            "processing_time": processing_time,
            "tables": tables,
            "extraction_method_used": "auto",
            "ocr_used": ocr_enabled
        }
        
        # Ajouter les résultats supplémentaires si disponibles
        if include_images:
            response["images"] = table_images
        
        if search_results:
            response["search_results"] = search_results
        
        if analysis_results:
            response["analysis"] = analysis_results
        
        # Mettre en cache si possible
        if cache_key and hasattr(components, 'cache'):
            # Stocker en cache pendant 1 heure (3600 secondes)
            await components.cache.set(cache_key, json.dumps(response), 3600)
        
        # Enregistrer les métriques
        metrics.finish_request_tracking(extraction_id)
        metrics.track_search_operation(
            method="auto",
            success=len(tables) > 0,
            processing_time=processing_time,
            results_count=len(tables)
        )
        
        return response
        
    except HTTPException:
        metrics.increment_counter("pdf_extraction_errors")
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

@router.get("/task-status/{task_id}")
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
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)
            return result
        
        # Vérifier si la tâche est en cours
        if os.path.exists(os.path.join("temp", "pdf_tasks", f"{task_id}.pdf")):
            return {
                "task_id": task_id,
                "status": "processing",
                "message": "Traitement en cours"
            }
        
        return {
            "task_id": task_id,
            "status": "not_found",
            "message": "Tâche non trouvée"
        }
    except Exception as e:
        logger.error(f"Erreur vérification statut tâche: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la vérification du statut: {str(e)}"
        )

# Fonctions auxiliaires

async def search_in_tables(tables, query: str) -> Dict[str, Any]:
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
    
    return {
        "query": query,
        "total_matches": total_matches,
        "matching_tables": len(results),
        "results": results
    }

async def export_tables(tables, format: str, filename: str) -> Dict[str, Any]:
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
        from io import BytesIO
        
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
                    
                    # Créer un CSV en mémoire
                    csv_buffer = BytesIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    # Ajouter au ZIP
                    zipf.writestr(f"table_{i+1}.csv", csv_buffer.getvalue())
            
            output.seek(0)
            export_filename = f"{os.path.splitext(filename)[0]}_tables.zip"
            
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
                    
                # Convertir en DataFrame si nécessaire
                if isinstance(table_data["data"], pd.DataFrame):
                    df = table_data["data"]
                    table_json = json.loads(df.replace({pd.NA: None}).to_json(orient="records"))
                else:
                    table_json = table_data["data"]
                
                result.append({
                    "table_id": i + 1,
                    "rows": len(table_json),
                    "columns": len(table_json[0]) if table_json and len(table_json) > 0 else 0,
                    "data": table_json
                })
            
            output.write(json.dumps(result, indent=2).encode('utf-8'))
            output.seek(0)
            export_filename = f"{os.path.splitext(filename)[0]}_tables.json"
            
            return {
                "content": output.getvalue(),
                "filename": export_filename,
                "media_type": "application/json",
                "download_url": f"/api/pdf/download/{export_filename}"
            }
        elif format == "html":
            output = BytesIO()
            html_content = "<html><head><style>table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid #ddd; padding: 8px;}</style></head><body>"
            
            for i, table_data in enumerate(tables):
                if "data" not in table_data:
                    continue
                    
                # Convertir en DataFrame si nécessaire
                if isinstance(table_data["data"], pd.DataFrame):
                    df = table_data["data"]
                    table_html = df.to_html(index=False)
                else:
                    df = pd.DataFrame(table_data["data"])
                    table_html = df.to_html(index=False)
                
                html_content += f"<h2>Tableau {i+1}</h2>{table_html}"
            
            html_content += "</body></html>"
            output.write(html_content.encode('utf-8'))
            output.seek(0)
            export_filename = f"{os.path.splitext(filename)[0]}_tables.html"
            
            return {
                "content": output.getvalue(),
                "filename": export_filename,
                "media_type": "text/html",
                "download_url": f"/api/pdf/download/{export_filename}"
            }
        else:
            raise ValueError(f"Format non supporté: {format}")
    
    except Exception as e:
        logger.error(f"Erreur export: {e}")
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
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            
            # Statistiques numériques de base
            numeric_stats = {}
            for col in df.select_dtypes(include=[np.number]).columns:
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
                "numeric_columns_count": numeric_cols,
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

async def process_pdf_in_background(task_id: str, file_path: str, params: Dict[str, Any]):
    """
    Traite un PDF en arrière-plan.
    
    Args:
        task_id: ID de la tâche
        file_path: Chemin du fichier PDF
        params: Paramètres d'extraction
    """
    try:
        from core.document_processing.table_extractor import TableExtractor
        
        # Créer un nouvel extracteur pour cette tâche
        extractor = TableExtractor()
        
        # Configuration OCR si activée
        ocr_config = None
        if params.get("ocr_enabled"):
            ocr_config = {
                "lang": params.get("ocr_language", "fra+eng"),
                "enhance_image": params.get("ocr_enhance_image", True),
                "deskew": params.get("ocr_deskew", True),
                "preprocess_type": "thresh",
                "force_grid": params.get("force_grid", False)
            }
        
        start_time = datetime.utcnow()
        
        # Extraire les tableaux avec la méthode auto
        with open(file_path, 'rb') as f:
            file_obj = io.BytesIO(f.read())
        
        # Utiliser la méthode automatique d'extraction
        tables = await extractor.extract_tables_auto(
            file_obj,
            output_format=params.get("output_format", "json"),
            max_pages=None if params.get("pages", "all") == "all" else len(params.get("pages", "all").split(",")),
            ocr_config=ocr_config
        )
        
        # Si demandé, extraire aussi les images des tableaux
        table_images = []
        if params.get("include_images"):
            file_obj.seek(0)
            table_images = await extractor.get_tables_as_images(
                file_obj,
                pages=params.get("pages", "all")
            )
        
        # Calcul du temps de traitement
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Préparation du résultat
        result = {
            "task_id": task_id,
            "status": "completed",
            "tables_count": len(tables),
            "processing_time": processing_time,
            "tables": tables,
            "extraction_method_used": "auto",
            "ocr_used": params.get("ocr_enabled", False)
        }
        
        if params.get("include_images"):
            result["images"] = table_images
        
        # Sauvegarder le résultat
        with open(os.path.join("temp", "pdf_tasks", f"{task_id}_result.json"), "w") as f:
            json.dump(result, f)
        
        # Nettoyer le fichier PDF temporaire
        os.remove(file_path)
        
    except Exception as e:
        logger.error(f"Erreur traitement en arrière-plan: {e}", exc_info=True)
        # Sauvegarder l'erreur
        with open(os.path.join("temp", "pdf_tasks", f"{task_id}_result.json"), "w") as f:
            json.dump({
                "task_id": task_id,
                "status": "error",
                "error": str(e)
            }, f)
        
        # Nettoyer le fichier PDF temporaire
        if os.path.exists(file_path):
            os.remove(file_path)