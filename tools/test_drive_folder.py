async def list_folder_content(self, folder_id: str) -> bool:
    """Liste le contenu d'un dossier Drive."""
    try:
        # Requête pour tous les fichiers dans le dossier
        query = f"'{folder_id}' in parents"
        results = await self._execute_with_retry(
            lambda: self.service.files().list(
                q=query,
                fields="files(id, name, mimeType, size, modifiedTime)",
                pageSize=100
            ).execute()
        )
        
        files = results.get('files', [])
        
        logger.info(f"Contenu du dossier {folder_id}:")
        logger.info("=" * 50)
        
        if not files:
            logger.info("Le dossier est vide")
            return True
            
        # Grouper par type MIME
        files_by_type = {}
        for file in files:
            mime_type = file.get('mimeType', 'unknown')
            if mime_type not in files_by_type:
                files_by_type[mime_type] = []
            files_by_type[mime_type].append(file)
            
        # Afficher les statistiques
        for mime_type, type_files in files_by_type.items():
            logger.info(f"\nType: {mime_type}")
            logger.info(f"Nombre de fichiers: {len(type_files)}")
            logger.info("Fichiers:")
            for file in type_files:
                modified = datetime.fromisoformat(file['modifiedTime'].replace('Z', '+00:00'))
                size = int(file.get('size', 0)) / (1024 * 1024)  # Conversion en MB
                logger.info(f"- {file['name']} (Modifié: {modified}, Taille: {size:.2f}MB)")
                
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du listage des fichiers: {e}")
        return False
