# core/llm/embedding_manager.py
from typing import List, Dict, Optional, Union, Any
import torch
import numpy as np
from datetime import datetime
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .model_loader import ModelType, LoadedModel
from core.config.models import EMBEDDING_MODELS
from core.config.config import settings

logger = get_logger("embedding_manager")

class EmbeddingManager:
    """Gestionnaire des modèles d'embedding."""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.model: Optional[LoadedModel] = None
        self._initialized = False
        self.embedding_cache = {}
        self.max_cache_size = 10000
        
        # Vérification des configurations
        if self.model_name not in EMBEDDING_MODELS:
            available_models = ", ".join(EMBEDDING_MODELS.keys())
            raise ValueError(f"Modèle {self.model_name} non disponible. Modèles disponibles : {available_models}")
            
        # Récupération des configurations
        self.model_config = EMBEDDING_MODELS[self.model_name]
        self.perf_config = self.model_config.get("generation_config", {})

    async def initialize(self, model_manager):
        """Initialise le modèle d'embedding."""
        if self._initialized:
            return

        try:
            logger.info(f"Initialisation du modèle d'embedding: {self.model_name}")
            
            # Vérifier si le modèle est déjà chargé
            existing_model = model_manager.current_models.get(ModelType.EMBEDDING)
            if existing_model and existing_model.model_name == self.model_name:
                logger.info(f"Utilisation du modèle d'embedding déjà chargé: {self.model_name}")
                self.model = existing_model
            else:
                # Chargement avec les nouvelles configurations
                self.model = await model_manager.change_model(
                    model_name=self.model_name,
                    model_type=ModelType.EMBEDDING
                )

            self._initialized = True
            logger.info("Modèle d'embedding initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation embedding: {e}")
            raise

    async def change_model(self, model_name: str, model_loader) -> bool:
        """Change le modèle d'embedding."""
        try:
            # Vérification avec les nouvelles configurations
            if model_name not in EMBEDDING_MODELS:
                available_models = ", ".join(EMBEDDING_MODELS.keys())
                raise ValueError(f"Modèle {model_name} non disponible. Modèles disponibles : {available_models}")

            # Chargement avec les nouvelles configurations
            new_model = await model_loader.load_model(
                model_name=model_name,
                model_type=ModelType.EMBEDDING
            )

            # Mise à jour des configurations
            self.model_name = model_name
            self.model = new_model
            self.model_config = EMBEDDING_MODELS[model_name]
            self.perf_config = self.model_config.get("generation_config", {})
            
            # Vider le cache car les dimensions peuvent changer
            self.embedding_cache.clear()

            logger.info(f"Modèle d'embedding changé avec succès pour {model_name}")
            return True

        except Exception as e:
            logger.error(f"Erreur changement modèle embedding {model_name}: {e}")
            return False

    async def get_embeddings(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        use_cache: bool = True
    ) -> List[List[float]]:
        """Génère des embeddings pour une liste de textes."""
        if not self._initialized or not self.model:
            raise RuntimeError("EmbeddingManager non initialisé")

        try:
            # Normalisation des entrées
            if isinstance(texts, str):
                texts = [texts]

            # Utilisation de la configuration de performance
            batch_size = batch_size or self.perf_config["batch_size"]
            embeddings = []

            # Configuration d'embedding
            embedding_params = self.model_config["generation_config"]["embedding_params"]
            normalize = embedding_params.get("normalize_embeddings", True)

            # Traitement par lots avec les paramètres configurés
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = []

                for text in batch:
                    # Vérification du cache
                    if use_cache and text in self.embedding_cache:
                        cache_entry = self.embedding_cache[text]
                        if cache_entry["model"] == self.model_name:
                            batch_embeddings.append(cache_entry["embedding"])
                            continue

                    # Génération des nouveaux embeddings avec la configuration
                    with torch.no_grad():
                        embedding = self.model.model.encode(
                            text,
                            convert_to_tensor=True,
                            normalize_embeddings=normalize,
                            batch_size=batch_size
                        )
                        embedding = embedding.cpu().tolist()

                        # Mise en cache avec la nouvelle structure
                        if use_cache:
                            self._update_cache(text, embedding)
                        
                        batch_embeddings.append(embedding)

                embeddings.extend(batch_embeddings)

            metrics.increment_counter("embeddings_generated")
            return embeddings

        except Exception as e:
            logger.error(f"Erreur génération embeddings: {e}")
            metrics.increment_counter("embedding_errors")
            return []

    async def get_similarity(
        self,
        text1: str,
        text2: str,
        use_cache: bool = True
    ) -> float:
        """Calcule la similarité entre deux textes."""
        try:
            # Génération des embeddings
            embeddings = await self.get_embeddings(
                [text1, text2],
                use_cache=use_cache
            )

            if len(embeddings) != 2:
                return 0.0

            # Calcul de la similarité cosinus
            vec1 = torch.tensor(embeddings[0])
            vec2 = torch.tensor(embeddings[1])
            
            similarity = torch.nn.functional.cosine_similarity(
                vec1.unsqueeze(0),
                vec2.unsqueeze(0)
            ).item()

            return float(similarity)

        except Exception as e:
            logger.error(f"Erreur calcul similarité: {e}")
            return 0.0

    def _update_cache(self, text: str, embedding: List[float]):
        """Met à jour le cache des embeddings."""
        try:
            # Nettoyage du cache si nécessaire
            if len(self.embedding_cache) >= self.max_cache_size:
                # Supprime 10% des entrées les plus anciennes
                num_to_remove = self.max_cache_size // 10
                sorted_cache = sorted(
                    self.embedding_cache.items(),
                    key=lambda x: x[1].get("timestamp", 0)
                )
                for key, _ in sorted_cache[:num_to_remove]:
                    del self.embedding_cache[key]

            # Ajout du nouvel embedding
            self.embedding_cache[text] = {
                "embedding": embedding,
                "timestamp": datetime.utcnow().timestamp(),
                "model": self.model_name,
                "dimension": len(embedding)
            }

        except Exception as e:
            logger.warning(f"Erreur mise à jour cache: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle actuel."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.model_config["embedding_dimension"],
            "supported_languages": self.model_config["languages"],
            "performance_config": self.perf_config,
            "generation_config": self.model_config.get("generation_config", {}),
            "cache_size": len(self.embedding_cache),
            "device": str(self.model.device) if self.model else "non initialisé",
            "max_sequence_length": self.model_config["max_sequence_length"]
        }

    async def cleanup(self):
        """Nettoie les ressources."""
        self._initialized = False
        self.embedding_cache.clear()
        if self.model and hasattr(self.model, 'cleanup'):
            await self.model.cleanup()
        logger.info("Ressources embedding nettoyées")