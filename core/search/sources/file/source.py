# core/search/sources/file/source.py
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from pathlib import Path

from ...interfaces import (
    SearchContext,
    SearchResultBase,
    SearchResultMetadata,
    SearchSourceCapabilities,
    SourceType,
    ContentType
)
from ..base import BaseDataSource
from ...exceptions import (
    SourceConnectionError,
    SearchConfigurationError
)
from core.utils.logger import get_logger

logger = get_logger(__name__)

class FileSource(BaseDataSource):
    """Source de données pour les fichiers."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_path: Optional[Path] = None
        self.supported_extensions = config.get("supported_extensions", [".txt", ".pdf", ".docx"])
        self.indexed_files: Dict[str, Dict[str, Any]] = {}
        
        self._capabilities = [
            SearchSourceCapabilities.FULL_TEXT_SEARCH,
            SearchSourceCapabilities.METADATA_SEARCH,
            SearchSourceCapabilities.FILTERS
        ]

    def get_source_type(self) -> SourceType:
        return SourceType.FILE

    def validate_config(self) -> None:
        """Valide la configuration de la source fichier."""
        if "base_path" not in self.config:
            raise SearchConfigurationError("base_path est requis dans la configuration")
        
        path = Path(self.config["base_path"])
        if not path.exists() or not path.is_dir():
            raise SearchConfigurationError(f"Le chemin {path} n'existe pas ou n'est pas un dossier")

    async def initialize(self) -> None:
        """Initialise la source de données fichier."""
        try:
            if not self._initialized:
                self.validate_config()
                self.base_path = Path(self.config["base_path"])
                await self._index_files()
                self._initialized = True
                logger.info(f"Source fichier initialisée avec {len(self.indexed_files)} fichiers")
        except Exception as e:
            raise SourceConnectionError(f"Erreur d'initialisation de la source fichier: {str(e)}")

    async def _index_files(self) -> None:
        """Indexe les fichiers du répertoire."""
        try:
            for root, _, files in os.walk(self.base_path):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.suffix.lower() in self.supported_extensions:
                        relative_path = str(file_path.relative_to(self.base_path))
                        # Stocker les métadonnées de base
                        self.indexed_files[relative_path] = {
                            "path": str(file_path),
                            "size": file_path.stat().st_size,
                            "modified": datetime.fromtimestamp(file_path.stat().st_mtime),
                            "extension": file_path.suffix.lower(),
                            "content_type": self._get_content_type(file_path.suffix.lower())
                        }

            logger.debug(f"Indexation terminée: {len(self.indexed_files)} fichiers trouvés")

        except Exception as e:
            raise SourceConnectionError(f"Erreur lors de l'indexation des fichiers: {str(e)}")

    def _get_content_type(self, extension: str) -> ContentType:
        """Détermine le type de contenu basé sur l'extension."""
        extension_mapping = {
            ".txt": ContentType.TEXT,
            ".md": ContentType.MARKDOWN,
            ".pdf": ContentType.BINARY,
            ".docx": ContentType.BINARY,
            ".json": ContentType.JSON
        }
        return extension_mapping.get(extension, ContentType.BINARY)

    async def search(self, context: SearchContext) -> List[SearchResultBase]:
        """Effectue une recherche dans les fichiers indexés."""
        if not self._initialized:
            await self.initialize()

        try:
            results = []
            query_lower = context.query.lower()

            for file_path, file_info in self.indexed_files.items():
                # Logique de filtrage basique sur le nom de fichier
                if query_lower in file_path.lower():
                    results.append(
                        SearchResultBase(
                            content=file_path,  # Le chemin comme contenu initial
                            score=1.0 if query_lower == file_path.lower() else 0.5,
                            metadata=SearchResultMetadata(
                                title=Path(file_path).name,
                                source="file",
                                source_type=SourceType.FILE,
                                content_type=file_info["content_type"],
                                timestamp=file_info["modified"],
                                extra={
                                    "size": file_info["size"],
                                    "extension": file_info["extension"],
                                    "full_path": file_info["path"]
                                }
                            ),
                            source_id=file_path
                        )
                    )

            # Application des filtres de métadonnées si présents
            if context.filters:
                results = self._apply_filters(results, context.filters)

            # Tri et limitation des résultats
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:context.max_results]

        except Exception as e:
            raise SourceConnectionError(f"Erreur de recherche fichier: {str(e)}")

    async def get_by_id(self, id: str) -> Optional[SearchResultBase]:
        """Récupère un fichier par son chemin relatif."""
        if not self._initialized:
            await self.initialize()

        try:
            if id not in self.indexed_files:
                return None

            file_info = self.indexed_files[id]
            return SearchResultBase(
                content=id,
                score=1.0,
                metadata=SearchResultMetadata(
                    title=Path(id).name,
                    source="file",
                    source_type=SourceType.FILE,
                    content_type=file_info["content_type"],
                    timestamp=file_info["modified"],
                    extra=file_info
                ),
                source_id=id
            )

        except Exception as e:
            logger.error(f"Erreur lors de la récupération du fichier {id}: {e}")
            return None

    def _apply_filters(
        self,
        results: List[SearchResultBase],
        filters: Dict[str, Any]
    ) -> List[SearchResultBase]:
        """Applique les filtres aux résultats."""
        filtered_results = []
        
        for result in results:
            match = True
            for key, value in filters.items():
                if key == "extension":
                    if result.metadata.extra.get("extension") != value:
                        match = False
                        break
                elif key == "max_size":
                    if result.metadata.extra.get("size", 0) > value:
                        match = False
                        break
                elif key == "min_date":
                    if result.metadata.timestamp < datetime.fromisoformat(value):
                        match = False
                        break
            
            if match:
                filtered_results.append(result)
                
        return filtered_results

    async def health_check(self) -> Dict[str, Any]:
        """Vérifie l'état de santé de la source fichier."""
        try:
            if not self._initialized:
                return {"status": "not_initialized"}

            # Vérification de l'accès au répertoire
            if not self.base_path.exists() or not self.base_path.is_dir():
                return {
                    "status": "unhealthy",
                    "error": "Répertoire de base inaccessible"
                }

            total_size = sum(info["size"] for info in self.indexed_files.values())
            latest_modified = max(
                (info["modified"] for info in self.indexed_files.values()),
                default=datetime.min
            )

            return {
                "status": "healthy",
                "files_count": len(self.indexed_files),
                "total_size_bytes": total_size,
                "latest_modified": latest_modified.isoformat(),
                "base_path": str(self.base_path),
                "supported_extensions": self.supported_extensions,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def cleanup(self) -> None:
        """Nettoie les ressources de la source fichier."""
        self.indexed_files.clear()
        await super().cleanup()