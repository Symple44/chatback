# core/llm/embeddings.py
from typing import List, Dict, Optional, Any, Union
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger("embeddings")

class EmbeddingsManager:
    def __init__(self):
        """Initialise le gestionnaire d'embeddings."""
        try:
            self.model = self._load_model()
            self.embedding_dim = settings.ELASTICSEARCH_EMBEDDING_DIM
            self.batch_size = settings.EMBEDDING_BATCH_SIZE
            self.executor = ThreadPoolExecutor(max_workers=4)
            self.cache = {}  # Cache simple pour les embeddings fréquents
            
            logger.info(f"Modèle d'embeddings initialisé: {settings.EMBEDDING_MODEL}")
            
        except Exception as e:
            logger.error(f"Erreur initialisation embeddings: {e}")
            raise

    def _load_model(self) -> SentenceTransformer:
        """Charge le modèle d'embeddings."""
        try:
            # Configuration du modèle
            model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device="cpu",  # Force CPU pour la stabilité
                cache_folder=str(settings.CACHE_DIR)
            )
            
            # Optimisations
            model.max_seq_length = 512
            model.eval()  # Mode évaluation
            
            return model
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise

    async def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Génère les embeddings pour une liste de textes.
        
        Args:
            texts: Liste de textes à encoder
            batch_size: Taille des lots optionnelle
            
        Returns:
            Liste d'embeddings
        """
        try:
            with metrics.timer("embedding_generation"):
                # Nettoyage et validation des textes
                texts = [self._preprocess_text(t) for t in texts]
                if not texts:
                    return []

                # Utilisation du cache pour les textes fréquents
                cached_embeddings = []
                texts_to_embed = []
                
                for text in texts:
                    if text in self.cache:
                        cached_embeddings.append(self.cache[text])
                    else:
                        texts_to_embed.append(text)

                if not texts_to_embed:
                    return cached_embeddings

                # Génération des nouveaux embeddings
                batch_size = batch_size or self.batch_size
                
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    self.executor,
                    self._batch_encode,
                    texts_to_embed,
                    batch_size
                )

                # Mise en cache des nouveaux embeddings
                for text, emb in zip(texts_to_embed, embeddings):
                    self.cache[text] = emb

                # Combinaison des embeddings cachés et nouveaux
                final_embeddings = []
                cache_idx = 0
                embed_idx = 0
                
                for text in texts:
                    if text in self.cache:
                        final_embeddings.append(cached_embeddings[cache_idx])
                        cache_idx += 1
                    else:
                        final_embeddings.append(embeddings[embed_idx])
                        embed_idx += 1

                metrics.increment_counter("embeddings_generated", len(texts))
                return final_embeddings

        except Exception as e:
            logger.error(f"Erreur génération embeddings: {e}")
            metrics.increment_counter("embedding_errors")
            raise

    def _batch_encode(
        self,
        texts: List[str],
        batch_size: int
    ) -> List[List[float]]:
        """Encode les textes par lots."""
        try:
            with torch.no_grad():
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    embeddings = self.model.encode(
                        batch,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True
                    )
                    all_embeddings.extend(embeddings)
                
                return [emb.tolist() for emb in all_embeddings]
                
        except Exception as e:
            logger.error(f"Erreur encodage par lots: {e}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """Prétraite un texte avant l'embedding."""
        if not isinstance(text, str):
            text = str(text)
        return text.strip()[:512]  # Limitation de la longueur

    @lru_cache(maxsize=1000)
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Génère l'embedding pour un texte unique.
        
        Args:
            text: Texte à encoder
            
        Returns:
            Embedding du texte
        """
        try:
            text = self._preprocess_text(text)
            # Exécuter dans un thread séparé car c'est une opération lourde
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                lambda: self.model.encode(
                    text, 
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
            )
                
            return embedding.tolist()
                
        except Exception as e:
            logger.error(f"Erreur génération embedding: {e}")
            raise

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calcule la similarité cosinus entre deux embeddings.
        
        Args:
            embedding1: Premier embedding
            embedding2: Deuxième embedding
            
        Returns:
            Score de similarité
        """
        try:
            v1 = np.array(embedding1)
            v2 = np.array(embedding2)
            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Erreur calcul similarité: {e}")
            return 0.0

    async def batch_similarity(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]]
    ) -> List[float]:
        """
        Calcule la similarité entre un embedding et une liste d'embeddings.
        
        Args:
            query_embedding: Embedding de requête
            embeddings: Liste d'embeddings à comparer
            
        Returns:
            Liste des scores de similarité
        """
        try:
            query_vec = np.array(query_embedding)
            doc_vecs = np.array(embeddings)
            
            # Normalisation
            query_vec = query_vec / np.linalg.norm(query_vec)
            doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1)[:, np.newaxis]
            
            # Calcul des similarités
            similarities = np.dot(doc_vecs, query_vec)
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Erreur calcul similarités par lot: {e}")
            return [0.0] * len(embeddings)

    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            self.executor.shutdown(wait=True)
            self.cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Ressources embeddings nettoyées")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage embeddings: {e}")

    def __del__(self):
        """Destructeur de la classe."""
        try:
            self.cleanup()
        except:
            pass
