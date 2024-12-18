import asyncio
import logging
from typing import Dict, Any
import sys
import os

# Ajout du répertoire parent au PYTHONPATH pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vectorstore.search import ElasticsearchClient
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("setup_indexes")

class IndexSetup:
    def __init__(self):
        """Initialize index setup."""
        self.es_client = ElasticsearchClient()

    async def setup_main_index(self) -> bool:
        """Configure l'index principal pour les documents."""
        try:
            index_settings = {
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "content_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase",
                                    "stop",
                                    "snowball"
                                ]
                            }
                        }
                    },
                    "number_of_shards": 2,
                    "number_of_replicas": 1,
                    "refresh_interval": "1s"
                },
                "mappings": {
                    "properties": {
                        "title": {
                            "type": "text",
                            "analyzer": "content_analyzer",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "content": {
                            "type": "text",
                            "analyzer": "content_analyzer"
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "application": {"type": "keyword"},
                                "page": {"type": "integer"},
                                "source": {"type": "keyword"}
                            }
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": settings.ELASTICSEARCH_EMBEDDING_DIM,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "timestamp": {"type": "date"}
                    }
                }
            }

            if await self.es_client.create_index("documents", index_settings):
                logger.info("Index principal créé avec succès")
                return True
            return False
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'index principal: {e}")
            return False

    async def setup_vector_index(self) -> bool:
        """Configure l'index pour les vecteurs."""
        try:
            vector_settings = {
                "settings": {
                    "number_of_shards": 2,
                    "number_of_replicas": 1
                },
                "mappings": {
                    "properties": {
                        "vector": {
                            "type": "dense_vector",
                            "dims": settings.ELASTICSEARCH_EMBEDDING_DIM,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {
                            "type": "object"
                        },
                        "text": {
                            "type": "text"
                        }
                    }
                }
            }

            if await self.es_client.create_index("vectors", vector_settings):
                logger.info("Index des vecteurs créé avec succès")
                return True
            return False
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'index des vecteurs: {e}")
            return False

    async def verify_indexes(self) -> Dict[str, bool]:
        """Vérifie l'existence et la configuration des index."""
        try:
            results = {
                "documents": await self.es_client.check_index("documents"),
                "vectors": await self.es_client.check_index("vectors")
            }
            return results
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des index: {e}")
            return {"documents": False, "vectors": False}

async def main():
    """Point d'entrée principal."""
    setup = IndexSetup()
    
    logger.info("Début de la configuration des index...")
    
    # Vérification de la connexion
    if not await setup.es_client.check_connection():
        logger.error("Impossible de se connecter à Elasticsearch")
        return
    
    # Configuration des index
    main_index = await setup.setup_main_index()
    vector_index = await setup.setup_vector_index()
    
    # Vérification finale
    verification = await setup.verify_indexes()
    
    if all(verification.values()):
        logger.info("Configuration des index terminée avec succès")
    else:
        logger.warning(f"Certains index n'ont pas été créés correctement: {verification}")

if __name__ == "__main__":
    asyncio.run(main())