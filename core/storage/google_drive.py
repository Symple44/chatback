# core/storage/google_drive.py
import os
import asyncio
from typing import List, Dict, Optional, Set
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import logging
import ssl
import json
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google.oauth2.credentials import Credentials
from googleapiclient.discovery_cache.base import Cache
from google.auth.transport.requests import Request

logger = logging.getLogger(__name__)

class MemoryCache(Cache):
    _CACHE = {}

    def get(self, url):
        return MemoryCache._CACHE.get(url)

    def set(self, url, content):
        MemoryCache._CACHE[url] = content

class GoogleDriveManager:
    def __init__(self, credentials_path: str):
        """Initialize Google Drive manager with credentials."""
        self.credentials_path = credentials_path
        self.credentials = None
        self.service = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.semaphore = asyncio.Semaphore(4)
        self.max_retries = 3
        self.retry_delay = 1.0
        
        logger.info(f"GoogleDriveManager initialisé avec {credentials_path}")

    async def initialize(self) -> bool:
        """
        Initialize credentials and service with proper error handling.
        """
        try:
            if not os.path.exists(self.credentials_path):
                logger.error(f"Credentials file not found: {self.credentials_path}")
                return False

            # Validate JSON structure
            try:
                with open(self.credentials_path, 'r') as f:
                    creds_content = json.load(f)
                    required_fields = ['client_email', 'private_key', 'project_id']
                    if not all(field in creds_content for field in required_fields):
                        logger.error("Invalid credentials file format")
                        return False
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in credentials file: {e}")
                return False

            # Create credentials
            self.credentials = ServiceAccountCredentials.from_service_account_file(
                self.credentials_path,
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )

            # Test credentials validity
            if not self.credentials.valid:
                self.credentials.refresh(Request())

            # Create service with retry
            for attempt in range(self.max_retries):
                try:
                    self.service = build(
                        'drive', 
                        'v3', 
                        credentials=self.credentials,
                        cache=MemoryCache()
                    )
                    # Test service
                    self.service.files().list(pageSize=1).execute()
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))

            logger.info("GoogleDriveManager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            return False

    async def sync_drive_folder(self, folder_id: str, save_path: str = "documents") -> Set[str]:
        """
        Synchronise un dossier Google Drive avec le système de fichiers local.
        Maintient la structure des dossiers d'applications.
        """
        logger.info(f"Démarrage de la synchronisation du dossier {folder_id}")
        logger.info(f"Chemin de sauvegarde: {save_path}")
        
        downloaded_files = set()
        
        try:
            # 1. Récupérer tous les sous-dossiers (applications)
            app_folders = await self.list_folders(folder_id)
            logger.info(f"Nombre de dossiers d'applications trouvés: {len(app_folders)}")
            
            for app_folder in app_folders:
                app_name = app_folder['name']
                app_id = app_folder['id']
                
                logger.info(f"Traitement de l'application: {app_name}")
                
                # Créer le dossier de l'application en local
                app_path = os.path.join(save_path, app_name)
                os.makedirs(app_path, exist_ok=True)
                
                # 2. Récupérer les fichiers PDF dans le dossier de l'application
                query = f"'{app_id}' in parents and mimeType='application/pdf'"
                results = await self._execute_with_retry(
                    lambda: self.service.files().list(
                        q=query,
                        fields="files(id, name, md5Checksum, modifiedTime, size)",
                        pageSize=1000
                    ).execute()
                )
                
                files = results.get('files', [])
                logger.info(f"Nombre de fichiers PDF trouvés dans {app_name}: {len(files)}")
                
                # 3. Télécharger les fichiers
                for file in files:
                    file_path = os.path.join(app_path, file['name'])
                    should_download = False
                    
                    # Vérifier si le fichier existe localement
                    if not os.path.exists(file_path):
                        logger.info(f"Nouveau fichier à télécharger: {file['name']}")
                        should_download = True
                    else:
                        # Vérifier le MD5 si disponible
                        if 'md5Checksum' in file:
                            local_md5 = await self._calculate_local_md5(file_path)
                            if local_md5 != file['md5Checksum']:
                                logger.info(f"Le fichier {file['name']} a été modifié (MD5 différent)")
                                should_download = True
                        else:
                            # Si pas de MD5, vérifier la date de modification
                            remote_time = datetime.fromisoformat(file['modifiedTime'].replace('Z', '+00:00'))
                            local_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                            if remote_time > local_time:
                                logger.info(f"Le fichier {file['name']} a été modifié (date plus récente)")
                                should_download = True
                    
                    if should_download:
                        success = await self._download_file_with_retries(file['id'], file_path)
                        if success:
                            downloaded_files.add(file_path)
                    else:
                        logger.info(f"Le fichier {file['name']} est à jour")

            logger.info(f"Nombre de fichiers PDF trouvés: {len(downloaded_files)}")
            if not downloaded_files:
                logger.info("Aucun fichier PDF trouvé dans le dossier")
            
            return downloaded_files

        except Exception as e:
            logger.error(f"Erreur lors de la synchronisation: {e}")
            return downloaded_files

    async def _execute_with_retry(self, func, max_retries: int = 3):
        """Execute a function with retry logic."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    func
                )
            except Exception as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
        raise last_error

    async def _download_file_with_retries(self, file_id: str, file_path: str) -> bool:
        """
        Télécharge un fichier depuis Google Drive avec mécanisme de retry.
        """
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    request = self.service.files().get_media(fileId=file_id)
                    
                    # Assurer que le répertoire existe
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    with open(file_path, 'wb') as file_handle:
                        downloader = MediaIoBaseDownload(file_handle, request, chunksize=1024*1024)
                        done = False
                        while not done:
                            status, done = downloader.next_chunk()
                            if status:
                                logger.debug(f"Téléchargement {file_path}: {int(status.progress() * 100)}%")
                    
                    logger.info(f"Fichier téléchargé: {file_path}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Erreur téléchargement {file_path}: {e}")
                    if attempt == self.max_retries - 1:
                        return False
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        return False

    async def _calculate_local_md5(self, file_path: str) -> str:
        """Calculate MD5 hash of a local file."""
        import hashlib
        try:
            md5_hash = hashlib.md5()
            with open(file_path, "rb") as f:
                # Lecture par blocs pour les gros fichiers
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
            return md5_hash.hexdigest()
        except Exception as e:
            logger.error(f"Erreur calcul MD5 pour {file_path}: {e}")
            return ""

    async def list_folders(self, parent_folder_id: Optional[str] = None) -> List[Dict]:
        """
        Liste tous les dossiers dans un dossier parent.
        """
        try:
            logger.info(f"Listage des dossiers{'dans '+parent_folder_id if parent_folder_id else ' racine'}")
            query = []
            
            if parent_folder_id:
                query.append(f"'{parent_folder_id}' in parents")
            
            query.append("mimeType='application/vnd.google-apps.folder'")
            final_query = " and ".join(query)
            
            logger.debug(f"Requête de recherche des dossiers: {final_query}")
            
            results = await self._execute_with_retry(
                lambda: self.service.files().list(
                    q=final_query,
                    fields="files(id, name, parents)",
                    pageSize=100
                ).execute()
            )
            
            folders = results.get('files', [])
            logger.info(f"Nombre de dossiers trouvés: {len(folders)}")
            return folders

        except Exception as e:
            logger.error(f"Erreur lors du listage des dossiers: {e}")
            return []

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
