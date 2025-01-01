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
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
                device=self.device,
                cache_folder=str(settings.CACHE_DIR)
            )
            
            # Optimisations
            model.max_seq_length = 512
            if self.device == "cuda":
                model.half()
            
            return model
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise

    async def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = True
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
            texts = [self._preprocess_text(t) for t in texts if t]
            if not texts:
                return []

            # Vérification du cache
            cached_embeddings = []
            texts_to_embed = []
            
            for text in texts:
                cache_key = hash(text)
                if cache_key in self._cache:
                    cached_embeddings.append(self._cache[cache_key])
                else:
                    texts_to_embed.append(text)

            if not texts_to_embed:
                return cached_embeddings

            # Génération par lots
            batch_size = batch_size or self.batch_size
            loop = asyncio.get_event_loop()
            
            embeddings = await loop.run_in_executor(
                self.executor,
                self._batch_encode,
                texts_to_embed,
                batch_size,
                normalize
            )

            # Mise en cache
            for text, emb in zip(texts_to_embed, embeddings):
                self._cache[hash(text)] = emb

            # Fusion des résultats
            final_embeddings = []
            cache_idx = 0
            embed_idx = 0
            
            for text in texts:
                cache_key = hash(text)
                if cache_key in self._cache:
                    final_embeddings.append(cached_embeddings[cache_idx])
                    cache_idx += 1
                else:
                    final_embeddings.append(embeddings[embed_idx])
                    embed_idx += 1

            return final_embeddings

        except Exception as e:
            logger.error(f"Erreur génération embeddings: {e}")
            return []

    def _batch_encode(
        self,
        texts: List[str],
        batch_size: int,
        normalize: bool = True
    ) -> List[List[float]]:
        """Encode les textes par lots avec optimisations."""
        try:
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        embeddings = self.model.encode(
                            texts,
                            batch_size=batch_size,
                            show_progress_bar=False,
                            normalize_embeddings=normalize,
                            convert_to_tensor=True
                        )
                else:
                    embeddings = self.model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=normalize,
                        convert_to_tensor=True
                    )

                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                return embeddings.cpu().numpy().tolist()

        except Exception as e:
            logger.error(f"Erreur encodage par lots: {e}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """Prétraitement amélioré des textes."""
        if not isinstance(text, str):
            text = str(text)
            
        # Nettoyage basique
        text = text.strip().lower()
        
        # Normalisation des caractères spéciaux
        text = text.replace('\n', ' ').replace('\t', ' ')
        
        # Limitation de longueur
        return text[:512]  # Limite à 512 tokens

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

    async def compute_similarity(
        self,
        embeddings1: Union[List[float], List[List[float]]],
        embeddings2: Union[List[float], List[List[float]]],
        metric: str = "cosine"
    ) -> Union[float, List[float]]:
        """Calcul de similarité avec plusieurs métriques."""
        try:
            e1 = np.array(embeddings1)
            e2 = np.array(embeddings2)
            
            if e1.ndim == 1:
                e1 = e1.reshape(1, -1)
            if e2.ndim == 1:
                e2 = e2.reshape(1, -1)

            if metric == "cosine":
                # Similarité cosinus optimisée
                norm1 = np.linalg.norm(e1, axis=1, keepdims=True)
                norm2 = np.linalg.norm(e2, axis=1, keepdims=True)
                dot_product = np.dot(e1, e2.T)
                similarities = dot_product / (norm1 * norm2.T)
                
            elif metric == "euclidean":
                # Distance euclidienne
                similarities = -np.linalg.norm(e1[:, None] - e2, axis=2)
                
            else:
                raise ValueError(f"Métrique non supportée: {metric}")

            return similarities.squeeze().tolist()

        except Exception as e:
            logger.error(f"Erreur calcul similarité: {e}")
            return 0.0 if len(embeddings1) == 1 and len(embeddings2) == 1 else []

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
        """Nettoyage des ressources."""
        try:
            self.executor.shutdown(wait=True)
            self._cache.clear()
            if hasattr(self.model, "cpu"):
                self.model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            logger.info("Ressources embeddings nettoyées")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage embeddings: {e}")

    def __del__(self):
        """Destructeur de la classe."""
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self.cleanup())
        except:
            pass
