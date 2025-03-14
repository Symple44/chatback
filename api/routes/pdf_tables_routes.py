# api/routes/pdf_tables_routes.py
from fastapi import APIRouter, UploadFile, File, Form, Query, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import io
import uuid
import tempfile
import os
import asyncio
import json
import time

from ..dependencies import get_components
from ..models.responses import ErrorResponse
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings
from core.document_processing.table_extraction.pipeline import TableExtractionPipeline
from core.document_processing.table_extraction.models import ExtractionResult, TableData, ImageData, PDFType
from core.document_processing.table_extraction.invoice_processor import InvoiceProcessor

logger = get_logger("pdf_tables_routes")
router = APIRouter(prefix="/pdf/tables", tags=["pdf"])

# Modèles de données pour les réponses
class TableExtractionResponse(BaseModel):
    extraction_id: str = Field(..., description="Identifiant unique de l'extraction")
    filename: str = Field(..., description="Nom du fichier traité")
    file_size: int = Field(..., description="Taille du fichier en octets")
    tables_count: int = Field(..., description="Nombre de tableaux extraits")
    processing_time: float = Field(..., description="Temps de traitement en secondes")
    tables: List[Dict[str, Any]] = Field(default=[], description="Tableaux extraits")
    extraction_method_used: str = Field(..., description="Méthode d'extraction utilisée")
    pdf_type: str = Field(..., description="Type de PDF (scanné/numérique)")
    ocr_used: bool = Field(default=False, description="OCR utilisé pour l'extraction")
    images: Optional[List[Dict[str, Any]]] = Field(None, description="Images des tableaux extraits")
    search_results: Optional[Dict[str, Any]] = Field(None, description="Résultats de recherche")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Analyse des tableaux")
    status: str = Field(default="completed", description="Statut de l'extraction")
    message: Optional[str] = Field(None, description="Message d'information")
    structured_data: Optional[Dict[str, Any]] = Field(None, description="Données structurées après post-traitement")

class TaskStatus(BaseModel):
    task_id: str = Field(..., description="Identifiant de la tâche")
    status: str = Field(..., description="Statut de la tâche (processing, completed, error)")
    message: str = Field(..., description="Message informatif")
    results: Optional[Dict[str, Any]] = Field(None, description="Résultats de la tâche si terminée")
    progress: Optional[float] = Field(None, description="Pourcentage de progression (0-100)")

@router.post("/extract", response_model=Union[TableExtractionResponse, Dict[str, Any]])
async def extract_tables(
    file: UploadFile = File(...),
    # Paramètres d'extraction
    strategy: str = Form(settings.table_extraction.DEFAULT_STRATEGY.value, description="Stratégie d'extraction"),
    output_format: str = Form("json", description="Format des données dans la réponse"),
    pages: str = Form("all", description="Pages à analyser"),
    # Type de document
    document_type: str = Form("generic", description="Type de document à extraire (generic, invoice, devis, receipt, etc.)"),
    # Paramètres OCR
    ocr_enabled: bool = Form(False, description="Activer l'OCR pour les PDF scannés"),
    ocr_language: str = Form(settings.table_extraction.OCR.TESSERACT_LANG, description="Langues pour l'OCR"),
    ocr_enhance_image: bool = Form(settings.table_extraction.OCR.ENHANCE_IMAGE, description="Améliorer l'image avant OCR"),
    ocr_deskew: bool = Form(settings.table_extraction.OCR.DESKEW, description="Redresser l'image avant OCR"),
    # Options supplémentaires
    include_images: bool = Form(False, description="Inclure les images des tableaux extraits"),
    analyze: bool = Form(False, description="Effectuer une analyse des tableaux"),
    search_query: Optional[str] = Form(None, description="Texte à rechercher dans les tableaux"),
    # Format d'export et traitement
    download_format: Optional[str] = Form(None, description="Format à télécharger (csv, excel, json, html)"),
    background_processing: bool = Form(False, description="Traiter en arrière-plan (pour les grands PDF)"),
    use_cache: bool = Form(True, description="Utiliser le cache si disponible"),
    # Traiter les cases à cocher
    extract_checkboxes: bool = Form(False, description="Extraire également les cases à cocher"),
    components = Depends(get_components)
):
    """
    Extrait les tableaux d'un fichier PDF.
    
    Cette API utilise une nouvelle architecture modulaire qui sélectionne intelligemment 
    la meilleure méthode d'extraction en fonction du contenu du PDF.
    
    - Détection automatique du type de PDF (scanné ou numérique)
    - Choix intelligent entre différentes méthodes d'extraction
    - Support OCR avancé pour les PDF scannés
    - Validation et optimisation des tableaux extraits
    - Cache configurable pour les extractions répétées
    - Support pour l'extraction des cases à cocher
    """
    try:
        # Mesure des performances
        extraction_id = str(uuid.uuid4())
        metrics.start_request_tracking(extraction_id)
        metrics.increment_counter("pdf_table_extractions")
        start_time = time.time()
        
        # Vérification du fichier
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Le fichier doit être au format PDF"
            )
        
        # Lecture du contenu
        file_content = await file.read()
        file_size = len(file_content)
        file_obj = io.BytesIO(file_content)
        
        # Extraction des cases à cocher si demandé
        checkbox_results = None
        if extract_checkboxes:
            try:
                # Créer une copie du fichier pour l'extraction des cases à cocher
                checkbox_file = io.BytesIO(file_content)
                
                # Extraire les cases à cocher
                if not hasattr(components, 'checkbox_extractor'):
                    from core.document_processing.table_extraction.checkbox_extractor import CheckboxExtractor
                    components._components["checkbox_extractor"] = CheckboxExtractor()
                
                checkbox_extractor = components.checkbox_extractor
                
                # Convertir la plage de pages
                if pages == "all":
                    page_range = None
                else:
                    page_range = []
                    try:
                        for part in pages.split(','):
                            if '-' in part:
                                start, end = map(int, part.split('-'))
                                page_range.extend(range(start, end + 1))
                            else:
                                page_range.append(int(part))
                    except:
                        page_range = None
                
                metrics.increment_counter("checkbox_extractions")
                checkbox_results = await checkbox_extractor.extract_checkboxes_from_pdf(checkbox_file, page_range)
                logger.info(f"Extraction des cases à cocher terminée: {len(checkbox_results.get('checkboxes', []))} cases trouvées")
            except Exception as e:
                logger.error(f"Erreur extraction cases à cocher: {e}")
                metrics.increment_counter("checkbox_extraction_errors")
                checkbox_results = {"error": str(e)}
        
        # Vérification de la taille
        if file_size > settings.table_extraction.MAX_FILE_SIZE:
            max_mb = settings.table_extraction.MAX_FILE_SIZE / (1024 * 1024)
            raise HTTPException(
                status_code=400,
                detail=f"Fichier trop volumineux. Taille maximum: {max_mb} Mo"
            )
        
        # Vérification des formats
        if output_format not in settings.table_extraction.SUPPORTED_OUTPUT_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Format de sortie invalide. Formats supportés: {', '.join(settings.table_extraction.SUPPORTED_OUTPUT_FORMATS)}"
        )
        
        # Vérification traitement en arrière-plan
        if background_processing and file_size > 5 * 1024 * 1024:  # > 5 Mo
            task_id = str(uuid.uuid4())
            temp_dir = os.path.join("temp", "pdf_tasks")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"{task_id}.pdf")
            
            # Sauvegarder le fichier
            with open(temp_path, "wb") as f:
                f.write(file_content)
            
            # Démarrer le traitement en arrière-plan
            background_tasks = BackgroundTasks()
            background_tasks.add_task(
                process_pdf_in_background,
                task_id=task_id,
                file_path=temp_path,
                original_filename=file.filename,
                params={
                    "strategy": strategy,
                    "output_format": output_format,
                    "pages": pages,
                    "ocr_config": {
                        "lang": ocr_language,
                        "enhance_image": ocr_enhance_image,
                        "deskew": ocr_deskew
                    },
                    "include_images": include_images,
                    "analyze": analyze,
                    "search_query": search_query,
                    "use_cache": use_cache,
                    "extract_checkboxes": extract_checkboxes
                },
                components=components
            )
            
            return JSONResponse(
                content={
                    "task_id": task_id,
                    "status": "processing",
                    "message": "Traitement démarré en arrière-plan, veuillez vérifier le statut ultérieurement",
                    "check_status_url": f"/api/pdf/tables/tasks/{task_id}",
                    "extraction_id": extraction_id,
                    "filename": file.filename,
                    "file_size": file_size
                },
                background=background_tasks
            )
        
        # Initialisation du pipeline et de l'InvoiceProcessor si nécessaire
        if not hasattr(components, 'table_extraction_pipeline'):
            table_detector = components.table_detector if hasattr(components, 'table_detector') else None
            cache_manager = components.cache_manager if hasattr(components, 'cache_manager') else None
            components._components["table_extraction_pipeline"] = TableExtractionPipeline(
                cache_manager=cache_manager,
                table_detector=table_detector
            )
        
        if not hasattr(components, 'invoice_processor'):
            components._components["invoice_processor"] = InvoiceProcessor()
        
        # Détection automatique du type de document si demandé
        auto_detect = document_type.lower() == "auto"
        detected_type = "generic"
        
        if auto_detect:
            # Utiliser la méthode de détection intégrée à l'InvoiceProcessor
            invoice_processor = components.invoice_processor
            doc_types = invoice_processor.detect_document_type(file_obj)
            dominant_type = max(doc_types.items(), key=lambda x: x[1])
            detected_type = dominant_type[0]
            detected_confidence = dominant_type[1]
            
            logger.info(f"Type de document auto-détecté: {detected_type} (confiance: {detected_confidence:.2f})")
            document_type = detected_type
        
        # Configuration OCR
        ocr_config = {
            "lang": ocr_language,
            "enhance_image": ocr_enhance_image,
            "deskew": ocr_deskew,
            "preprocess_type": "thresh",
            "psm": 6
        } if ocr_enabled else None
        
        # Forcer la stratégie OCR si demandé
        if ocr_enabled and strategy == "auto":
            strategy = "ocr"
        
        # Extraction des tableaux
        pipeline = components.table_extraction_pipeline
        result = await pipeline.extract_tables(
            file_obj=file_obj,
            pages=pages,
            strategy=strategy,
            output_format=output_format,
            ocr_config=ocr_config,
            use_cache=use_cache
        )

        # Post-traitement spécifique selon le type de document
        structured_data = None
        
        if document_type.lower() in ["devis", "technical_invoice", "devis_technique"]:
            try:
                # Appliquer le post-traitement pour les devis techniques
                invoice_processor = components.invoice_processor
                structured_data = invoice_processor.process_technical_invoice(result.tables)
                logger.info(f"Post-traitement de devis technique appliqué avec succès")
                
                # Ajouter des métadonnées d'extraction
                structured_data["extraction_id"] = extraction_id
                structured_data["extraction_method"] = result.extraction_method_used
                structured_data["pdf_type"] = result.pdf_type
                structured_data["ocr_used"] = result.ocr_used
                structured_data["processing_time"] = time.time() - start_time
                
                # Si auto-détection, ajouter l'information
                if auto_detect:
                    structured_data["auto_detected"] = True
                    structured_data["detection_confidence"] = detected_confidence
                
                # Ajouter les résultats des cases à cocher si disponibles
                if checkbox_results and extract_checkboxes:
                    structured_data["checkboxes"] = checkbox_results
                
                return structured_data
                
            except Exception as e:
                logger.error(f"Erreur lors du post-traitement de devis technique: {e}")
                structured_data = {"error": str(e)}
                
        elif document_type.lower() == "invoice":
            try:
                # Appliquer le post-traitement pour les factures
                invoice_processor = components.invoice_processor
                structured_data = invoice_processor.process(result.tables)
                logger.info(f"Post-traitement de facture appliqué avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du post-traitement de facture: {e}")
                structured_data = {"error": str(e)}
        
        # Si download_format est spécifié, générer le fichier à télécharger
        if download_format and result.tables:
            export_result = await export_tables(result.tables, download_format, file.filename, components)
            if export_result.get("download_url"):
                return StreamingResponse(
                    io.BytesIO(export_result["content"]),
                    media_type=export_result["media_type"],
                    headers={"Content-Disposition": f"attachment; filename={export_result['filename']}"}
                )
        
        # Si include_images demandé, obtenir les images des tableaux
        if include_images and not result.images:
            file_obj.seek(0)
            images = await pipeline.get_tables_as_images(file_obj, pages)
            result.images = images
        
        # Si search_query est spécifié, rechercher dans les tableaux
        if search_query and search_query.strip() and result.tables:
            search_results = await search_in_tables(result.tables, search_query)
            result.search_results = search_results
        
        # Si analyze est demandé, analyser les tableaux
        if analyze and result.tables:
            analysis_results = await analyze_tables(result.tables, output_format)
            result.analysis = analysis_results

        # Post-traitement spécifique selon le type de document
        structured_data = None
        if document_type.lower() == "invoice" and result.tables:
            try:
                # Appliquer le post-traitement pour les factures
                invoice_processor = InvoiceProcessor()
                structured_data = invoice_processor.process(result.tables)
                logger.info(f"Post-traitement de facture appliqué avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du post-traitement de facture: {e}")
                structured_data = {"error": str(e)}
        
        # Terminer le suivi des métriques
        processing_time = time.time() - start_time
        metrics.finish_request_tracking(extraction_id)
        
        try:
            metrics.track_search_operation(
                method=result.extraction_method_used,
                success=result.tables_count > 0,
                processing_time=processing_time,
                results_count=result.tables_count
            )
        except Exception as e:
            logger.error(f"Erreur tracking métriques: {e}")
        
        # Ajouter les données structurées à la réponse si disponibles
        if structured_data:
            # Créer une copie du résultat pour ajouter les données structurées
            result_dict = result.to_dict()
            result_dict["structured_data"] = structured_data
            
            # Ajouter les résultats des cases à cocher si disponibles
            if checkbox_results and extract_checkboxes:
                result_dict["checkboxes"] = checkbox_results
                
            return result_dict
            
        # Retour du résultat standard avec les cases à cocher si demandé
        result_dict = result.to_dict()
        
        # Ajouter les résultats des cases à cocher si disponibles
        if checkbox_results and extract_checkboxes:
            result_dict["checkboxes"] = checkbox_results
            
        return result_dict
        
    except HTTPException:
        metrics.increment_counter("pdf_extraction_errors")
        raise
    except Exception as e:
        logger.error(f"Erreur extraction tableaux: {str(e)}", exc_info=True)
        metrics.increment_counter("pdf_extraction_errors")
        
        # Calculer le temps de traitement même en cas d'erreur
        processing_time = time.time() - start_time
        metrics.finish_request_tracking(extraction_id)
        
        # Retourner une réponse d'erreur structurée
        return TableExtractionResponse(
            extraction_id=extraction_id,
            filename=file.filename if file else "unknown",
            file_size=file_size if 'file_size' in locals() else 0,
            tables_count=0,
            processing_time=processing_time,
            tables=[],
            extraction_method_used=strategy,
            pdf_type="unknown",
            ocr_used=ocr_enabled,
            status="error",
            message=f"Erreur lors de l'extraction des tableaux: {str(e)}"
        )
    finally:
        # Nettoyage
        await file.close()

@router.get("/tasks/{task_id}", response_model=TaskStatus)
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
        
        # Tâche non trouvée
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
        
        # Lire et retourner le fichier
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

# Fonctions utilitaires 

async def process_pdf_in_background(
    task_id: str, 
    file_path: str, 
    original_filename: str,
    params: Dict[str, Any],
    components
):
    """
    Traite un PDF en arrière-plan.
    
    Args:
        task_id: ID de la tâche
        file_path: Chemin du fichier PDF
        original_filename: Nom original du fichier
        params: Paramètres d'extraction
        components: Composants d'application
    """
    try:
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
        
        # Extraire les cases à cocher si demandé
        checkbox_results = None
        if params.get("extract_checkboxes", False):
            try:
                update_progress(10, "Extraction des cases à cocher")
                
                # Initialiser l'extracteur de cases à cocher si nécessaire
                if not hasattr(components, 'checkbox_extractor'):
                    from core.document_processing.table_extraction.checkbox_extractor import CheckboxExtractor
                    components._components["checkbox_extractor"] = CheckboxExtractor()
                
                checkbox_extractor = components.checkbox_extractor
                
                # Convertir la plage de pages
                pages = params.get("pages", "all")
                if pages == "all":
                    page_range = None
                else:
                    page_range = []
                    try:
                        for part in pages.split(','):
                            if '-' in part:
                                start, end = map(int, part.split('-'))
                                page_range.extend(range(start, end + 1))
                            else:
                                page_range.append(int(part))
                    except:
                        page_range = None
                
                checkbox_results = await checkbox_extractor.extract_checkboxes_from_pdf(file_path, page_range)
                logger.info(f"Extraction des cases à cocher terminée: {len(checkbox_results.get('checkboxes', []))} cases trouvées")
                update_progress(20, f"Extraction des cases à cocher terminée: {len(checkbox_results.get('checkboxes', []))} cases trouvées")
            except Exception as e:
                logger.error(f"Erreur extraction cases à cocher: {e}")
                checkbox_results = {"error": str(e)}
                update_progress(20, f"Erreur extraction cases à cocher: {str(e)}")
        
        # Vérifier si c'est un devis technique
        document_type = params.get("document_type", "generic").lower()
        
        # Initialisation de l'InvoiceProcessor si nécessaire
        if not hasattr(components, 'invoice_processor'):
            components._components["invoice_processor"] = InvoiceProcessor()
        
        # Détection automatique du type de document si demandé
        auto_detect = document_type.lower() == "auto"
        detected_type = "generic"
        
        if auto_detect:
            # Utiliser la méthode de détection intégrée à l'InvoiceProcessor
            update_progress(30, "Détection du type de document")
            invoice_processor = components.invoice_processor
            doc_types = invoice_processor.detect_document_type(file_path)
            dominant_type = max(doc_types.items(), key=lambda x: x[1])
            detected_type = dominant_type[0]
            detected_confidence = dominant_type[1]
            
            logger.info(f"Type de document auto-détecté: {detected_type} (confiance: {detected_confidence:.2f})")
            document_type = detected_type
            
        # S'assurer que le pipeline d'extraction est disponible
        if not hasattr(components, 'table_extraction_pipeline'):
            table_detector = components.table_detector if hasattr(components, 'table_detector') else None
            cache_manager = components.cache_manager if hasattr(components, 'cache_manager') else None
            components._components["table_extraction_pipeline"] = TableExtractionPipeline(
                cache_manager=cache_manager,
                table_detector=table_detector
            )
        
        # Code existant pour l'extraction...
        update_progress(40, "Extraction des tableaux en cours")
        pipeline = components.table_extraction_pipeline
        result = await pipeline.extract_tables(
            file_obj=file_path,
            pages=params.get("pages", "all"),
            strategy=params.get("strategy", "auto"),
            output_format=params.get("output_format", "json"),
            ocr_config=params.get("ocr_config"),
            use_cache=params.get("use_cache", True)
        )
        
        update_progress(70, f"Extraction des tableaux terminée: {result.tables_count} tableaux trouvés")
        
        # Post-traitement spécifique selon le type de document
        structured_data = None
        
        if document_type in ["devis", "technical_invoice", "devis_technique"]:
            update_progress(80, "Post-traitement de devis technique")
            try:
                # Appliquer le post-traitement pour les devis techniques
                invoice_processor = components.invoice_processor
                structured_data = invoice_processor.process_technical_invoice(result.tables)
                
                # Ajouter des métadonnées d'extraction
                structured_data["extraction_id"] = task_id
                structured_data["extraction_method"] = result.extraction_method_used
                structured_data["pdf_type"] = result.pdf_type
                structured_data["ocr_used"] = result.ocr_used
                
                # Si auto-détection, ajouter l'information
                if auto_detect:
                    structured_data["auto_detected"] = True
                    structured_data["detection_confidence"] = detected_confidence
                
                # Ajouter les cases à cocher si disponibles
                if checkbox_results and params.get("extract_checkboxes", False):
                    structured_data["checkboxes"] = checkbox_results
                
                # Écrire le résultat
                with open(os.path.join("temp", "pdf_tasks", f"{task_id}_result.json"), "w") as f:
                    json.dump(structured_data, f, default=str)
                    
                update_progress(100, "Traitement terminé")
                return
                
            except Exception as e:
                logger.error(f"Erreur lors du post-traitement de devis technique: {e}")
                structured_data = {"error": str(e)}
                
        elif document_type == "invoice":
            update_progress(80, "Post-traitement de facture")
            try:
                # Appliquer le post-traitement pour les factures
                invoice_processor = components.invoice_processor
                structured_data = invoice_processor.process(result.tables)
                
                # Ajouter les cases à cocher si disponibles
                if checkbox_results and params.get("extract_checkboxes", False):
                    structured_data["checkboxes"] = checkbox_results
            except Exception as e:
                logger.error(f"Erreur lors du post-traitement de facture: {e}")
                structured_data = {"error": str(e)}
        
        # Préparer le résultat final
        update_progress(90, "Préparation des résultats")
        result_dict = result.to_dict()
        
        if structured_data:
            result_dict["structured_data"] = structured_data
        
        # Ajouter les cases à cocher si disponibles
        if checkbox_results and params.get("extract_checkboxes", False):
            result_dict["checkboxes"] = checkbox_results
        
        # Sauvegarder le résultat
        with open(os.path.join("temp", "pdf_tasks", f"{task_id}_result.json"), "w") as f:
            json.dump(result_dict, f, default=str)
        
        # Mise à jour finale
        update_progress(100, "Traitement terminé")
        
        # Nettoyer le fichier PDF temporaire
        os.remove(file_path)
        
    except Exception as e:
        logger.error(f"Erreur traitement en arrière-plan: {e}", exc_info=True)
        # Sauvegarder l'erreur
        with open(os.path.join("temp", "pdf_tasks", f"{task_id}_result.json"), "w") as f:
            json.dump({
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "message": f"Erreur lors du traitement: {str(e)}"
            }, f)
        
        # Nettoyer le fichier PDF temporaire
        if os.path.exists(file_path):
            os.remove(file_path)

async def search_in_tables(tables: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """
    Recherche un texte dans les tableaux.
    
    Args:
        tables: Liste de tableaux
        query: Texte à rechercher
        
    Returns:
        Résultats de recherche
    """
    try:
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
                    "page": table.get("page", i + 1),
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
        
    except Exception as e:
        logger.error(f"Erreur recherche dans tableaux: {e}")
        return {
            "query": query,
            "total_matches": 0,
            "matching_tables": 0,
            "results": [],
            "error": str(e)
        }

async def analyze_tables(tables: List[Dict[str, Any]], output_format: str) -> Dict[str, Any]:
    """
    Analyse une liste de tableaux pour produire des statistiques.
    
    Args:
        tables: Liste de tableaux
        output_format: Format des tableaux
        
    Returns:
        Dictionnaire de statistiques
    """
    try:
        import pandas as pd
        import numpy as np
        
        if not tables:
            return {"error": "Aucun tableau à analyser"}
        
        # Pour les formats autres que pandas, convertir d'abord
        table_dfs = []
        for table in tables:
            if "data" not in table:
                continue
                
            df = None
            if output_format == "pandas":
                df = table["data"]
            elif isinstance(table["data"], list):
                df = pd.DataFrame(table["data"])
            elif isinstance(table["data"], str):
                # Tenter de parser les formats texte
                if output_format == "csv":
                    df = pd.read_csv(io.StringIO(table["data"]))
                elif output_format == "json":
                    df = pd.read_json(io.StringIO(table["data"]))
                elif output_format == "html":
                    df = pd.read_html(table["data"])[0]
            
            if df is not None:
                table_dfs.append(df)
        
        if not table_dfs:
            return {"error": "Impossible de convertir les tableaux pour analyse"}
        
        # Analyse globale
        result = {
            "table_count": len(table_dfs),
            "tables": []
        }
        
        total_rows = 0
        total_cols = 0
        total_cells = 0
        non_null_cells = 0
        
        for i, df in enumerate(table_dfs):
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
        result["average_row_count"] = round(total_rows / len(table_dfs), 2) if len(table_dfs) > 0 else 0
        result["total_cells"] = total_cells
        result["null_percentage"] = round(((total_cells - non_null_cells) / total_cells * 100), 2) if total_cells > 0 else 0
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur analyse tableaux: {e}")
        return {"error": str(e)}

async def export_tables(tables: List[Dict[str, Any]], format: str, filename: str, components) -> Dict[str, Any]:
    """
    Exporte les tableaux dans un format spécifique.
    
    Args:
        tables: Liste de tableaux
        format: Format d'export ('excel', 'csv', 'json', 'html', 'pdf', 'markdown')
        filename: Nom du fichier original
        components: Composants de l'application
        
    Returns:
        Informations sur le fichier exporté
    """
    try:
        # Vérifier si l'exportateur est déjà initialisé dans les composants
        if not hasattr(components, 'table_exporter'):
            from core.document_processing.table_extraction.exporter import TableExporter
            components._components["table_exporter"] = TableExporter()
            logger.info("Table Exporter initialisé")
        
        # Utiliser l'exportateur pour générer le fichier
        exporter = components.table_exporter
        return await exporter.export_tables(tables, format, filename)
        
    except Exception as e:
        logger.error(f"Erreur export tableaux: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'export des tableaux: {str(e)}"
        )