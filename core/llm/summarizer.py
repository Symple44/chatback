# core/llm/summarizer.py
import torch
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.llm.model_loader import ModelType, LoadedModel
from core.config.models import SUMMARIZER_MODELS, SUMMARIZER_PERFORMANCE_CONFIGS

logger = get_logger("summarizer")

class DocumentSummarizer:
    def __init__(self):
        """Initialise le résumeur de contexte avec analyse thématique."""
        self.model_name = "mt5-base-multi-sum"  # Modèle par défaut
        self.model: Optional[LoadedModel] = None
        self._initialized = False
        self.current_config = SUMMARIZER_MODELS[self.model_name]
        self.perf_config = SUMMARIZER_PERFORMANCE_CONFIGS[self.model_name]
        
    async def initialize(self, model_manager):
        """Initialise le modèle de résumé."""
        if self._initialized:
            return

        try:
            logger.info(f"Initialisation du résumeur: {self.model_name}")
            
            # Utilisation de change_model au lieu de load_model
            self.model = await model_manager.change_model(
                model_name=self.model_name,
                model_type=ModelType.SUMMARIZER
            )

            self._initialized = True
            logger.info("Résumeur initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation summarizer: {e}")
            raise

    async def change_model(self, model_name: str, model_loader) -> bool:
        """Change le modèle de résumé."""
        try:
            if model_name not in SUMMARIZER_MODELS:
                raise ValueError(f"Modèle {model_name} non disponible")

            # Chargement du nouveau modèle
            new_model = await model_loader.load_model(
                model_name=model_name,
                model_type=ModelType.SUMMARIZER
            )

            # Mise à jour des configurations
            self.model_name = model_name
            self.model = new_model
            self.current_config = SUMMARIZER_MODELS[model_name]
            self.perf_config = SUMMARIZER_PERFORMANCE_CONFIGS[model_name]

            return True

        except Exception as e:
            logger.error(f"Erreur changement modèle summarizer: {e}")
            return False

    async def summarize_documents(self, documents: List[Dict]) -> Dict[str, str]:
        """Crée un résumé structuré avec analyse thématique."""
        if not self._initialized or not self.model:
            raise RuntimeError("Summarizer non initialisé")

        try:
            with metrics.timer("context_analysis"):
                # Préparation du contenu
                content = self._prepare_content(documents)
                
                # Configuration de génération selon le modèle actuel
                gen_config = {
                    "max_length": self.perf_config["max_length"],
                    "min_length": self.perf_config["min_length"],
                    "num_beams": self.perf_config["num_beams"],
                    "length_penalty": self.perf_config["length_penalty"],
                    "early_stopping": self.perf_config["early_stopping"],
                    "no_repeat_ngram_size": self.perf_config["no_repeat_ngram_size"]
                }

                # Tokenisation avec paramètres optimisés
                inputs = self.model.tokenizer(
                    content,
                    max_length=self.perf_config["preprocessing"]["max_length"],
                    padding=self.perf_config["preprocessing"]["padding"],
                    truncation=True,
                    return_tensors="pt"
                )

                # Envoi sur le bon device
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                # Génération avec autocast pour optimisation mémoire
                with torch.amp.autocast('cuda', enabled=True):
                    output_ids = self.model.model.generate(
                        **inputs,
                        **gen_config
                    )

                # Décodage
                summary = self.model.tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                # Structure de la réponse
                return {
                    "summary": summary,
                    "metadata": {
                        "model_used": self.model_name,
                        "timestamp": datetime.utcnow().isoformat(),
                        "config_used": gen_config
                    }
                }

        except Exception as e:
            logger.error(f"Erreur génération résumé: {e}")
            metrics.increment_counter("summarization_errors")
            return self._get_default_summary()

    def _prepare_content(self, documents: List[Dict]) -> str:
        """Prépare le contenu pour la génération du résumé."""
        processed_docs = []
        
        for doc in documents:
            if not isinstance(doc, dict):
                continue
                
            content = doc.get("content", "").strip()
            if not content:
                continue

            # Ajout du contexte structuré
            metadata = doc.get("metadata", {})
            app_name = metadata.get("application", "unknown")
            processed_docs.append(f"[Source: {app_name}]\n{content}")

        return "\n---\n".join(processed_docs)

    def _get_default_summary(self) -> Dict[str, str]:
        """Retourne un résumé par défaut en cas d'erreur."""
        return {
            "summary": "Une erreur est survenue lors de la génération du résumé.",
            "metadata": {
                "error": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    async def cleanup(self):
        """Nettoie les ressources."""
        self._initialized = False
        if hasattr(self.model, 'cleanup'):
            await self.model.cleanup()