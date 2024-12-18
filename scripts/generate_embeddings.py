import asyncio
import logging
from typing import List, Dict, Optional
import sys
import os
from datetime import datetime
import argparse

# Ajout du répertoire parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm.embeddings import EmbeddingsManager
from core.vectorstore.search import ElasticsearchClient
from core.document_processing.text_splitter import DocumentSplitter
from core.utils.logger import get_logger
from core.config import settings

logger = get_logger("generate_embeddings")

class EmbeddingsGenerator:
    def __init__(self):
        """Initialize embeddings generator."""
        self.es_client = ElasticsearchClient()
        self.embeddings_manager = EmbeddingsManager()
        self.text_splitter = DocumentSplitter()
        
    async def process_documents(
        self,
        query: Optional[Dict] = None,
        batch_size: int = 100
    ) -> bool:
        """
        Traite les documents et génère leurs embeddings.
        
        Args:
            query: Filtre pour les documents à traiter
            batch_size: Taille des lots de traitement
            
        Returns:
            bool: Succès de l'opération
        """
        try:
            # Récupération des documents
            documents = await self.es_client.get_documents_without_embeddings(query)
            total_docs = len(documents)
            logger.info(f"Documents à traiter: {total_docs}")

            # Traitement par lots
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                await self._process_batch(batch)
                logger.info(f"Traité {i + len(batch)}/{total_docs} documents")

            return True
        except Exception as e:
            logger.error(f"Erreur lors du traitement des documents: {e}")
            return False

    async def _process_batch(self, documents: List[Dict]) -> None:
        """Traite un lot de documents."""
        try:
            for doc in documents:
                # Découpage du texte
                chunks = self.text_splitter.split_document(
                    doc.get("content", ""),
                    doc.get("metadata", {})
                )

                # Génération des embeddings pour chaque chunk
                for chunk in chunks:
                    embedding = self.embeddings_manager.generate_embeddings(
                        [chunk["content"]]
                    )[0]

                    # Mise à jour du document
                    await self.es_client.update_document(
                        doc["_id"],
                        {
                            "embedding": embedding,
                            "chunk_content": chunk["content"],
                            "processed_at": datetime.utcnow().isoformat()
                        }
                    )

        except Exception as e:
            logger.error(f"Erreur lors du traitement du lot: {e}")
            raise

    async def regenerate_embeddings(
        self,
        age_days: Optional[int] = None
    ) -> bool:
        """
        Régénère les embeddings des documents.
        
        Args:
            age_days: Limite d'âge des documents à retraiter
            
        Returns:
            bool: Succès de l'opération
        """
        try:
            query = None
            if age_days:
                query = {
                    "range": {
                        "processed_at": {
                            "lte": f"now-{age_days}d/d"
                        }
                    }
                }

            return await self.process_documents(query)
        except Exception as e:
            logger.error(f"Erreur lors de la régénération des embeddings: {e}")
            return False

async def main():
    parser = argparse.ArgumentParser(description='Génère les embeddings pour les documents.')
    parser.add_argument(
        '--regenerate',
        action='store_true',
        help='Régénère les embeddings existants'
    )
    parser.add_argument(
        '--age',
        type=int,
        help='Âge maximum des documents à retraiter (en jours)'
    )
    
    args = parser.parse_args()
    
    generator = EmbeddingsGenerator()
    
    if args.regenerate:
        logger.info("Début de la régénération des embeddings...")
        success = await generator.regenerate_embeddings(args.age)
    else:
        logger.info("Début de la génération des embeddings...")
        success = await generator.process_documents()

    if success:
        logger.info("Opération terminée avec succès")
    else:
        logger.error("Des erreurs sont survenues pendant l'opération")

if __name__ == "__main__":
    asyncio.run(main())