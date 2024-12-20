# core/storage/sync_manager.py
import os
import hashlib
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List, Set
import logging
from datetime import datetime
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.document_processing.pdf_processor import PDFProcessor

logger = get_logger(__name__)

class DocumentSyncManager:
    def __init__(self, es_client, docs_dir: str = "documents", sync_file: str = "sync_status.json"):
        """
        Initialize the document synchronization manager.
        
        Args:
            es_client: ElasticsearchClient instance
            docs_dir: Directory for documents
            sync_file: File to store sync status
        """
        self.es_client = es_client
        self.docs_dir = docs_dir
        self.sync_file = sync_file
        self.pdf_processor = PDFProcessor(es_client)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def index_document(self, file_path: str, metadata: Optional[Dict] = None) -> bool:
        """
        Index a single document with its metadata.
        
        Args:
            file_path: Path to the document file
            metadata: Additional metadata for the document
            
        Returns:
            bool: Success status
        """
        try:
            # Création des répertoires si nécessaire
            os.makedirs(self.docs_dir, exist_ok=True)

            # Copie du fichier dans le répertoire de documents si nécessaire
            if not file_path.startswith(self.docs_dir):
                new_path = os.path.join(self.docs_dir, os.path.basename(file_path))
                import shutil
                shutil.copy2(file_path, new_path)
                file_path = new_path

            # Indexation selon le type de fichier
            if file_path.lower().endswith('.pdf'):
                success = await self.pdf_processor.index_pdf(file_path, metadata)
            else:
                # Traitement des autres types de fichiers
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Indexation du document
                await self.es_client.index_document(
                    title=os.path.basename(file_path),
                    content=content,
                    metadata={
                        "source": file_path,
                        "indexed_at": datetime.utcnow().isoformat(),
                        **(metadata or {})
                    }
                )
                success = True

            if success:
                logger.info(f"Document indexé avec succès: {file_path}")
                metrics.increment_counter("documents_indexed")
            return success

        except Exception as e:
            logger.error(f"Erreur lors de l'indexation de {file_path}: {e}")
            metrics.increment_counter("document_indexing_errors")
            return False

    async def sync_documents(self, folder_id: Optional[str] = None) -> Set[str]:
        """
        Synchronise les documents depuis un dossier.
        
        Args:
            folder_id: ID du dossier Google Drive optionnel
            
        Returns:
            Set des fichiers synchronisés
        """
        try:
            with metrics.timer("document_sync"):
                previous_status = self._load_sync_status()
                current_status = {}
                synced_files = set()
                
                # Parcours des fichiers locaux
                for subdir, _, files in os.walk(self.docs_dir):
                    for filename in files:
                        if not filename.lower().endswith(('.pdf', '.docx', '.txt')):
                            continue
                            
                        file_path = os.path.join(subdir, filename)
                        try:
                            # Calcul du hash pour vérifier les changements
                            file_hash = self._calculate_file_hash(file_path)
                            current_status[file_path] = file_hash
                            
                            # Vérifier si le fichier a été modifié
                            if previous_status.get(file_path) != file_hash:
                                # Traitement selon le type de fichier
                                if filename.lower().endswith('.pdf'):
                                    success = await self.pdf_processor.index_pdf(
                                        file_path,
                                        metadata={
                                            "application": os.path.basename(subdir),
                                            "last_synced": datetime.utcnow().isoformat(),
                                            "directory": subdir
                                        }
                                    )
                                    if success:
                                        synced_files.add(file_path)
                                        logger.info(f"PDF synchronisé: {filename}")
                                else:
                                    # Traitement pour les autres types de fichiers
                                    metadata = {
                                        "source": file_path,
                                        "filename": filename,
                                        "directory": subdir,
                                        "application": os.path.basename(subdir),
                                        "last_synced": datetime.utcnow().isoformat()
                                    }
                                    
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        content = f.read()
                                    
                                    # Indexation dans Elasticsearch
                                    await self.es_client.index_document(
                                        title=filename,
                                        content=content,
                                        metadata=metadata
                                    )
                                    synced_files.add(file_path)
                                    logger.info(f"Document synchronisé: {filename}")
                                    
                        except Exception as e:
                            logger.error(f"Erreur lors du traitement de {filename}: {e}")
                    
                    # Sauvegarde du statut de synchronisation
                    self._save_sync_status(current_status)
                    
                metrics.increment_counter("sync_operations")
                logger.info(f"Synchronisation terminée: {len(synced_files)} fichiers traités")
                
                return synced_files
                
        except Exception as e:
            logger.error(f"Erreur lors de la synchronisation: {e}", exc_info=True)
            metrics.increment_counter("sync_errors")
            return set()

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Erreur lors du calcul du hash pour {file_path}: {e}")
            return ""

    def _load_sync_status(self) -> Dict[str, str]:
        """Load synchronization status from file."""
        try:
            if os.path.exists(self.sync_file):
                with open(self.sync_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Erreur lors du chargement du statut de sync: {e}")
        return {}

    def _save_sync_status(self, status: Dict[str, str]) -> None:
        """Save synchronization status to file."""
        try:
            # Création du répertoire parent si nécessaire
            os.makedirs(os.path.dirname(self.sync_file), exist_ok=True)
            
            with open(self.sync_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du statut de sync: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources and temporary files."""
        try:
            self.executor.shutdown(wait=True)
            logger.info("Ressources nettoyées avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des ressources: {e}")

    def __del__(self):
        """Cleanup resources on deletion."""
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des ressources: {e}")
