# core/document_processing/pdf_processor.py
from typing import List, Dict, Tuple, Generator, Optional
from PyPDF2 import PdfReader
import fitz
import os
from pathlib import Path
import asyncio
from datetime import datetime

from core.utils.logger import get_logger
from core.vectorstore import ElasticsearchClient
from .image_processing import PDFImageProcessor

logger = get_logger("pdf_processor")

class PDFProcessor:
    def __init__(self, es_client: ElasticsearchClient):
        """
        Initialise le processeur PDF.
        
        Args:
            es_client: Client Elasticsearch pour l'indexation
        """
        self.es_client = es_client
        self.image_processor = PDFImageProcessor()
        self.temp_dir = Path("temp/pdf")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def process_pdf(
        self,
        file_path: str,
        extract_images: bool = True
    ) -> Dict[str, any]:
        """
        Traite un fichier PDF complet.
        
        Args:
            file_path: Chemin du fichier PDF
            extract_images: Si True, extrait aussi les images
            
        Returns:
            Contenu et métadonnées du PDF
        """
        try:
            logger.info(f"Traitement du PDF: {file_path}")
            
            # Extraction du texte
            text_content = await self.extract_text(file_path)
            
            # Extraction des images si demandé
            images = []
            if extract_images:
                images = await self.image_processor.extract_images_from_pdf(file_path)
            
            # Lecture des métadonnées
            metadata = await self._extract_metadata(file_path)
            
            return {
                "content": text_content,
                "images": images,
                "metadata": metadata,
                "processed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur traitement PDF {file_path}: {e}")
            raise

    async def extract_text(self, file_path: str) -> str:
        """
        Extrait le texte d'un PDF.
        
        Args:
            file_path: Chemin du fichier PDF
            
        Returns:
            Texte extrait
        """
        logger.info(f"Extraction du texte de {file_path}")
        text_content = ""
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Ajout du numéro de page et du texte
                text_content += f"\n=== Page {page_num + 1} ===\n{text}"
                
                logger.debug(f"Page {page_num + 1} extraite")
            
            return text_content
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction: {e}")
            raise
        finally:
            if 'doc' in locals():
                doc.close()

    async def _extract_metadata(self, file_path: str) -> Dict[str, any]:
        """
        Extrait les métadonnées d'un PDF.
        
        Args:
            file_path: Chemin du fichier PDF
            
        Returns:
            Métadonnées du PDF
        """
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                info = reader.metadata
                
                return {
                    "title": info.get("/Title", ""),
                    "author": info.get("/Author", ""),
                    "subject": info.get("/Subject", ""),
                    "creator": info.get("/Creator", ""),
                    "producer": info.get("/Producer", ""),
                    "creation_date": info.get("/CreationDate", ""),
                    "modification_date": info.get("/ModDate", ""),
                    "pages": len(reader.pages)
                }
                
        except Exception as e:
            logger.error(f"Erreur extraction métadonnées: {e}")
            return {}

    async def index_pdf(self, file_path: str, metadata: Optional[Dict] = None) -> bool:
        try:
            logger.info(f"Indexation du PDF: {file_path}")
            path = Path(file_path)
            application = path.parent.name
            filename = path.name
    
            # Extraction du contenu
            content = await self.extract_text(file_path)
            if not content.strip():
                logger.warning(f"Aucun contenu extrait de {file_path}")
                return False
    
            # Extraction des métadonnées PDF
            doc_metadata = await self._extract_metadata(file_path)
            
            # Construction du document
            document = {
                "title": filename,
                "content": content,
                "application": application,  # Stocké au niveau racine
                "file_path": str(file_path),
                "metadata": {
                    "page_count": doc_metadata.get("pages", 0),
                    "author": doc_metadata.get("author", ""),
                    "creation_date": doc_metadata.get("creation_date", ""),
                    "indexed_at": datetime.utcnow().isoformat(),
                    "sync_date": metadata.get("sync_date") if metadata else None,
                    **doc_metadata  # Autres métadonnées
                }
            }
    
            # Génération du vecteur si le modèle est disponible
            if self.embedding_model:
                try:
                    document["embedding"] = await self.embedding_model.generate_embedding(content)
                except Exception as e:
                    logger.error(f"Erreur génération embedding pour {filename}: {e}")
    
            # Indexation
            success = await self.es_client.index_document(**document)
            
            if success:
                logger.info(f"PDF {filename} indexé avec succès pour l'application {application}")
            
            return success
    
        except Exception as e:
            logger.error(f"Erreur indexation PDF {file_path}: {e}", exc_info=True)
            return False

    async def index_directory(
        self,
        directory: str,
        recursive: bool = True
    ) -> Tuple[int, int]:
        """
        Indexe récursivement les PDFs d'un répertoire.
        
        Args:
            directory: Répertoire à indexer
            recursive: Si True, indexe aussi les sous-répertoires
            
        Returns:
            Tuple (nombre de succès, nombre d'échecs)
        """
        success = 0
        failed = 0
        
        try:
            for root, _, files in os.walk(directory):
                if not recursive and root != directory:
                    continue
                
                for filename in files:
                    if not filename.lower().endswith('.pdf'):
                        continue
                        
                    file_path = os.path.join(root, filename)
                    try:
                        if await self.index_pdf(
                            file_path,
                            metadata={"directory": root}
                        ):
                            success += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"Échec indexation {filename}: {e}")
                        failed += 1
            
            logger.info(f"Indexation terminée: {success} succès, {failed} échecs")
            return success, failed
            
        except Exception as e:
            logger.error(f"Erreur indexation répertoire: {e}")
            return success, failed

    async def cleanup(self):
        """Nettoie les ressources temporaires."""
        try:
            # Nettoyage du répertoire temporaire
            for file in self.temp_dir.glob("*"):
                file.unlink()
            
            # Nettoyage du processeur d'images
            self.image_processor.cleanup()
            
            logger.info("Ressources PDF nettoyées")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage ressources PDF: {e}")
