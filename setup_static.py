#!/usr/bin/env python3
from pathlib import Path
import httpx
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SWAGGER_FILES = {
    'css': {
        'swagger-ui.css': 'https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css',
    },
    'js': {
        'swagger-ui-bundle.js': 'https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js',
    }
}

async def download_file(url: str, dest_path: Path):
    """Télécharge un fichier depuis une URL."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            dest_path.write_bytes(response.content)
            logger.info(f"Téléchargé: {dest_path}")
    except Exception as e:
        logger.error(f"Erreur téléchargement {url}: {e}")
        raise

async def setup_static_files():
    """Configure les fichiers statiques nécessaires."""
    static_dir = Path("static")
    swagger_dir = static_dir / "swagger"
    
    # Création des répertoires
    for subdir in ["css", "js"]:
        (swagger_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Téléchargement des fichiers
    tasks = []
    for subdir, files in SWAGGER_FILES.items():
        for filename, url in files.items():
            dest_path = swagger_dir / subdir / filename
            if not dest_path.exists():
                tasks.append(download_file(url, dest_path))
    
    if tasks:
        await asyncio.gather(*tasks)
        logger.info("Configuration des fichiers statiques terminée")
    else:
        logger.info("Les fichiers statiques sont déjà présents")

if __name__ == "__main__":
    asyncio.run(setup_static_files())
