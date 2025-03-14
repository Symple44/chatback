# À ajouter dans le fichier api/routes/pdf_tables_routes.py

# 1. Importer la nouvelle classe en haut du fichier
from core.document_processing.table_extraction.technical_form_extractor import TechnicalFormExtractor


# 2. Modifier votre route principale pour utiliser l'extracteur spécialisé quand nécessaire

@router.post("/extract", response_model=Union[TableExtractionResponse, Dict[str, Any]])
async def extract_tables(
    file: UploadFile = File(...),
    # [Les autres paramètres restent identiques]
    # Ajout du paramètre pour forcer l'utilisation de l'extracteur spécialisé
    use_technical_extractor: bool = Form(False, description="Utiliser l'extracteur spécialisé pour fiches techniques"),
    components = Depends(get_components)
):
    """
    Extrait les tableaux et cases à cocher d'un fichier PDF.
    [Le reste de la documentation reste identique]
    """
    try:
        # [Le début de la fonction reste identique]
        
        # Ajouter la détection du type de document technique
        is_technical_form = False
        if use_technical_extractor:
            is_technical_form = True
        else:
            # Détection automatique des fiches CANAMETAL
            with fitz.open(stream=file_content) as doc:
                if doc.page_count > 0:
                    first_page_text = doc[0].get_text()
                    # Motifs spécifiques aux fiches CANAMETAL ou documents techniques
                    if ("CANAMETAL" in first_page_text or 
                        "Fiche Affaire" in first_page_text or 
                        "OSSATURE" in first_page_text or
                        "LOT" in first_page_text and "ECLISSAGE" in first_page_text):
                        is_technical_form = True
                        logger.info("Document technique CANAMETAL détecté, utilisation de l'extracteur spécialisé")
        
        # Si c'est un document technique et que l'extraction spécialisée est activée
        if is_technical_form:
            try:
                # Initialiser l'extracteur spécialisé s'il n'existe pas
                if not hasattr(components, 'technical_form_extractor'):
                    components._components["technical_form_extractor"] = TechnicalFormExtractor()
                
                tech_extractor = components.technical_form_extractor
                
                # Extraire les données avec l'extracteur spécialisé
                file_obj.seek(0)
                technical_data = await tech_extractor.extract_form_data(file_obj)
                
                # Ajouter des métadonnées 
                technical_data["extraction_id"] = extraction_id
                technical_data["filename"] = file.filename
                technical_data["file_size"] = file_size
                technical_data["processing_time"] = time.time() - start_time
                technical_data["status"] = "completed"
                
                # Retourner directement les données techniques
                return technical_data
                
            except Exception as e:
                logger.error(f"Erreur extraction technique: {e}")
                # Continuer avec l'extraction standard en cas d'échec
        
        # [Le reste de la route reste identique]
        
    except Exception as e:
        # [Gestion d'erreur inchangée]