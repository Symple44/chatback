from typing import List, Dict, Optional, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class DocumentSplitter:
    def __init__(self):
        """
        Initialise le splitter de documents avec les configurations par défaut.
        """
        try:
            # Configuration du splitter principal
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Patterns à nettoyer ou ignorer
            self.patterns_to_clean = [
                r"Page \d+.*?(?=\n|$)",  # En-têtes de page
                r"^\s*\d+\s*$",          # Numéros de page isolés
                r"_{3,}",                # Lignes de séparation
                r"\[TOC\]",              # Tables des matières
                r"©.*?(?=\n|$)"          # Mentions de copyright
            ]
            
            logger.info("DocumentSplitter initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du DocumentSplitter: {e}")
            raise

    def split_document(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        min_chunk_size: int = 100
    ) -> List[Dict[str, Union[str, Dict]]]:
        """
        Découpe un document en chunks avec préservation du contexte.
        
        Args:
            text: Le texte à découper
            metadata: Métadonnées à associer aux chunks
            min_chunk_size: Taille minimale d'un chunk valide
            
        Returns:
            Liste de dictionnaires contenant les chunks et leurs métadonnées
        """
        try:
            # Nettoyage préliminaire du texte
            cleaned_text = self._clean_text(text)
            
            # Découpage du texte
            chunks = self.text_splitter.split_text(cleaned_text)
            
            # Post-traitement et filtrage des chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                # Ignorer les chunks trop petits
                if len(chunk.strip()) < min_chunk_size:
                    continue
                    
                # Enrichir le chunk avec son contexte
                enriched_chunk = self._enrich_chunk(chunk, i, len(chunks))
                
                # Préparer les métadonnées du chunk
                chunk_metadata = {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **(metadata or {})
                }
                
                processed_chunks.append({
                    "content": enriched_chunk,
                    "metadata": chunk_metadata
                })
            
            logger.info(f"Document découpé en {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Erreur lors du découpage du document: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        """
        Nettoie le texte en supprimant les éléments indésirables.
        """
        cleaned_text = text
        
        try:
            # Suppression des patterns indésirables
            for pattern in self.patterns_to_clean:
                cleaned_text = re.sub(pattern, "", cleaned_text)
            
            # Normalisation des espaces et sauts de ligne
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
            
            # Suppression des espaces en début et fin
            cleaned_text = cleaned_text.strip()
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du texte: {e}")
            return text

    def _enrich_chunk(self, chunk: str, index: int, total_chunks: int) -> str:
        """
        Enrichit un chunk avec du contexte si nécessaire.
        """
        try:
            enriched_chunk = chunk.strip()
            
            # Ajout d'indicateurs de position si nécessaire
            if index == 0:
                enriched_chunk = "Début du document:\n" + enriched_chunk
            elif index == total_chunks - 1:
                enriched_chunk = enriched_chunk + "\nFin du document."
            
            # Assurer que le chunk se termine par une ponctuation
            if not enriched_chunk.endswith((".", "!", "?")):
                enriched_chunk += "..."
            
            return enriched_chunk
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enrichissement du chunk: {e}")
            return chunk

    def split_documents_batch(
        self,
        documents: List[Dict[str, Union[str, Dict]]],
        min_chunk_size: int = 100
    ) -> List[Dict[str, Union[str, Dict]]]:
        """
        Traite un lot de documents.
        
        Args:
            documents: Liste de documents à traiter
            min_chunk_size: Taille minimale d'un chunk
            
        Returns:
            Liste de tous les chunks générés
        """
        try:
            all_chunks = []
            
            for doc in documents:
                text = doc.get("content", "")
                metadata = {
                    "title": doc.get("title", ""),
                    **(doc.get("metadata", {}))
                }
                
                chunks = self.split_document(
                    text=text,
                    metadata=metadata,
                    min_chunk_size=min_chunk_size
                )
                
                all_chunks.extend(chunks)
            
            logger.info(f"Traitement par lot terminé: {len(all_chunks)} chunks générés")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement par lot: {e}")
            return []

    def estimate_chunk_count(self, text: str) -> int:
        """
        Estime le nombre de chunks qui seront générés pour un texte.
        """
        try:
            return len(self.text_splitter.split_text(text))
        except Exception as e:
            logger.error(f"Erreur lors de l'estimation des chunks: {e}")
            return 0