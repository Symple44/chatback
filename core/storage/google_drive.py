# core/google_drive.py
import os
import asyncio
from typing import List, Dict, Optional, Set
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import logging
import ssl
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.discovery_cache.base import Cache

logger = logging.getLogger(__name__)

class MemoryCache(Cache):
    _CACHE = {}

    def get(self, url):
        return MemoryCache._CACHE.get(url)

    def set(self, url, content):
        MemoryCache._CACHE[url] = content

class GoogleDriveManager:
    def __init__(self, credentials_path: str, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialise le gestionnaire Google Drive avec les credentials fournis.
        """
        try:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"Fichier de credentials non trouvé: {credentials_path}")

            self.credentials = ServiceAccountCredentials.from_service_account_file(credentials_path)
            self.service = build('drive', 'v3', credentials=self.credentials, cache=MemoryCache())
            self.executor = ThreadPoolExecutor(max_workers=4)
            self.semaphore = asyncio.Semaphore(1)
            self.last_successful_sync = None
            self.max_retries = max_retries
            self.retry_delay = retry_delay
            logger.info("Gestionnaire Google Drive initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur d'initialisation Google Drive: {e}", exc_info=True)
            raise

    async def _get_file_metadata(self, file_id: str) -> Optional[Dict]:
        """
        Récupère les métadonnées d'un fichier de Google Drive.
        """
        try:
            logger.debug(f"Récupération des métadonnées pour le fichier {file_id}")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.service.files().get(
                    fileId=file_id,
                    fields="id, name, mimeType, modifiedTime, md5Checksum, size, parents"
                ).execute()
            )
            logger.debug(f"Métadonnées récupérées: {result}")
            return result
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métadonnées pour {file_id}: {e}", exc_info=True)
            return None

    async def _download_file_with_retries(self, file_id: str, file_path: str) -> bool:
        """
        Télécharge un fichier depuis Google Drive avec un mécanisme de réessai en cas d'échec.
        """
        async with self.semaphore:
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    logger.info(f"Début du téléchargement de {file_path}")
                    request = self.service.files().get_media(fileId=file_id)
                    file_handle = open(file_path, 'wb')
                    downloader = MediaIoBaseDownload(file_handle, request, chunksize=5 * 1024 * 1024)

                    loop = asyncio.get_event_loop()
                    done = False
                    while not done:
                        status, done = await loop.run_in_executor(
                            self.executor,
                            downloader.next_chunk
                        )
                        logger.info(f"Téléchargement {file_path}: {int(status.progress() * 100)}%")

                    logger.info(f"Téléchargement terminé: {file_path}")
                    return True
                except ssl.SSLError as e:
                    logger.error(f"Erreur SSL lors du téléchargement de {file_path}: {e}", exc_info=True)
                    retry_count += 1
                    if retry_count < self.max_retries:
                        delay = self.retry_delay * 2 ** retry_count
                        logger.info(f"Nouvelle tentative dans {delay:.2f} secondes...")
                        await asyncio.sleep(delay)
                except Exception as e:
                    logger.error(f"Erreur lors du téléchargement de {file_path}: {e}", exc_info=True)
                    retry_count += 1
                    if retry_count < self.max_retries:
                        delay = self.retry_delay * 2 ** retry_count
                        logger.info(f"Nouvelle tentative dans {delay:.2f} secondes...")
                        await asyncio.sleep(delay)
                finally:
                    if 'file_handle' in locals():
                        file_handle.close()

            logger.error(f"Échec du téléchargement de {file_path} après {self.max_retries} tentatives")
            return False

    async def _calculate_local_md5(self, file_path: str) -> str:
        """
        Calcule le MD5 d'un fichier local de manière asynchrone.
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self._calculate_md5_sync,
                file_path
            )
        except Exception as e:
            logger.error(f"Erreur lors du calcul du MD5 pour {file_path}: {e}")
            return ""

    def _calculate_md5_sync(self, file_path: str) -> str:
        """
        Calcule le MD5 d'un fichier de manière synchrone.
        """
        import hashlib
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            # Lecture par blocs pour gérer les gros fichiers
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    async def sync_drive_folder(self, folder_id: str, save_path: str = "downloads") -> Set[str]:
        """
        Synchronise un dossier Google Drive avec le système de fichiers local.
        Ne télécharge que les fichiers nouveaux ou modifiés.
        """
        logger.info(f"Démarrage de la synchronisation du dossier {folder_id}")
        logger.info(f"Chemin de sauvegarde: {save_path}")
        
        downloaded_files = set()
        os.makedirs(save_path, exist_ok=True)

        try:
            # Récupérer la liste des fichiers dans Drive
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.service.files().list(
                    q=f"'{folder_id}' in parents and mimeType='application/pdf'",
                    fields="files(id, name, md5Checksum, modifiedTime, size)",
                    pageSize=1000
                ).execute()
            )

            files = results.get('files', [])
            logger.info(f"Nombre de fichiers PDF trouvés: {len(files)}")

            if not files:
                logger.info("Aucun fichier PDF trouvé dans le dossier")
                return downloaded_files

            # Vérification des fichiers existants
            for file in files:
                file_path = os.path.join(save_path, file['name'])
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
                    logger.info(f"Le fichier {file['name']} est à jour, pas de téléchargement nécessaire")

            logger.info(f"{len(downloaded_files)} fichiers téléchargés sur {len(files)} fichiers trouvés")

        except Exception as e:
            logger.error(f"Erreur lors de la synchronisation du dossier {folder_id}: {e}", exc_info=True)
        
        return downloaded_files

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
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
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
            logger.error(f"Erreur lors du listage des dossiers: {e}", exc_info=True)
            return []

    def __del__(self):
        """
        Nettoyage des ressources lors de la destruction de l'objet.
        """
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)