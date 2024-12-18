# core/llm/embeddings.py
from typing import List
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from langchain.embeddings.base import Embeddings
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingsManager(Embeddings):
    def __init__(self):
        """Initialise le gestionnaire d'embeddings."""
        try:
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self.model.to("cpu")  # Force CPU pour la stabilité
            self.embedding_dim = settings.ELASTICSEARCH_EMBEDDING_DIM
            logger.info(f"Modèle d'embeddings initialisé: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des embeddings: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Génère des embeddings pour une liste de textes.
        
        Args:
            texts: Liste de textes à encoder
            
        Returns:
            Liste d'embeddings
        """
        try:
            if not texts:
                logger.warning("Liste de textes vide reçue pour embed_documents")
                return []
                
            # S'assurer que tous les textes sont des strings
            texts = [str(text) if not isinstance(text, str) else text for text in texts]
            
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=settings.BATCH_SIZE,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                return embeddings.tolist()
        except Exception as e:
            logger.error(f"Erreur lors de la génération des embeddings: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Génère l'embedding pour un texte unique.
        
        Args:
            text: Texte à encoder
            
        Returns:
            Embedding du texte
        """
        try:
            # S'assurer que l'entrée est une string
            text = str(text) if not isinstance(text, str) else text
            logger.debug(f"Génération d'embedding pour la requête: {text[:100]}...")
            
            with torch.no_grad():
                embedding = self.model.encode(
                    text,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                
                # S'assurer que l'embedding est un tableau 1D
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.flatten()
                    
                return embedding.tolist()
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'embedding pour la requête: {e}")
            raise

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calcule la similarité cosinus entre deux embeddings.
        """
        try:
            v1 = np.array(embedding1)
            v2 = np.array(embedding2)
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        except Exception as e:
            logger.error(f"Erreur lors du calcul de similarité: {e}")
            return 0.0

    def batch_compute_similarity(
        self,
        query_embedding: List[float],
        document_embeddings: List[List[float]]
    ) -> List[float]:
        """
        Calcule les similarités entre une requête et plusieurs documents.
        """
        try:
            query_vec = np.array(query_embedding)
            doc_vecs = np.array(document_embeddings)
            
            # Normalisation
            query_vec = query_vec / np.linalg.norm(query_vec)
            doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1)[:, np.newaxis]
            
            # Calcul des similarités
            similarities = np.dot(doc_vecs, query_vec)
            return similarities.tolist()
        except Exception as e:
            logger.error(f"Erreur lors du calcul des similarités par lot: {e}")
            return [0.0] * len(document_embeddings)

    def __del__(self):
        """Nettoyage des ressources."""
        try:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des ressources: {e}")