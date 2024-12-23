import asyncio
import logging
from core.storage.google_drive import GoogleDriveManager
from core.config import settings
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_drive")

async def test_drive_access():
    folder_id = settings.GOOGLE_DRIVE_FOLDER_ID
    logger.info(f"Testing access to folder: {folder_id}")
    logger.info(f"Using credentials from: {settings.GOOGLE_DRIVE_CREDENTIALS_PATH}")

    try:
        # Création manuelle des credentials pour plus de détails
        credentials = service_account.Credentials.from_service_account_file(
            settings.GOOGLE_DRIVE_CREDENTIALS_PATH,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        logger.info("Credentials created successfully")

        # Création manuelle du service
        service = build('drive', 'v3', credentials=credentials)
        logger.info("Drive service created successfully")

        # Test simple de listage
        logger.info("Testing file listing...")
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            pageSize=10,
            fields="files(id, name, mimeType)"
        ).execute()
        
        files = results.get('files', [])
        logger.info(f"Found {len(files)} files/folders")
        
        # Afficher les détails de chaque fichier
        for file in files:
            logger.info(f"- {file.get('name')} ({file.get('mimeType')})")

        # Vérifier les permissions du dossier
        logger.info(f"\nChecking folder permissions...")
        folder = service.files().get(
            fileId=folder_id,
            fields="name, permissions"
        ).execute()
        logger.info(f"Folder name: {folder.get('name')}")
        logger.info("Permissions:")
        for permission in folder.get('permissions', []):
            logger.info(f"- {permission.get('emailAddress', 'N/A')}: {permission.get('role')}")

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_drive_access())
