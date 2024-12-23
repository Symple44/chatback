#!/usr/bin/env python3
import asyncio
import os
from core.storage.google_drive import GoogleDriveManager
from core.config import settings

async def main():
    drive_manager = GoogleDriveManager(settings.GOOGLE_DRIVE_CREDENTIALS_PATH)
    if await drive_manager.initialize():
        await drive_manager.list_folder(settings.GOOGLE_DRIVE_FOLDER_ID)
    else:
        print("Erreur d'initialisation du drive manager")

if __name__ == "__main__":
    asyncio.run(main())
