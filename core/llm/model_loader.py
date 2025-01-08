# core/llm/model_loader.py
from typing import Dict, Tuple, Optional, Any, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer
import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from core.config.models import (
    AVAILABLE_MODELS,
    EMBEDDING_MODELS,
    SUMMARIZER_MODELS,
    MODEL_PERFORMANCE_CONFIGS,
    EMBEDDING_PERFORMANCE_CONFIGS,
    SUMMARIZER_PERFORMANCE_CONFIGS
)
from core.utils.logger import get_logger
from core.llm.cuda_manager import ModelPriority

logger = get_logger("model_loader")

class ModelType(Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    SUMMARIZER = "summarizer"

@dataclass
class LoadedModel:
    """Représente un modèle chargé avec ses métadonnées."""
    model: PreTrainedModel
    tokenizer: Optional[AutoTokenizer]
    model_type: ModelType
    model_name: str
    config: Dict
    loaded_at: float
    device: torch.device

class ModelLoader:
    """Gestionnaire de chargement des modèles avec support multi-type."""
    
    def __init__(self, cuda_manager, tokenizer_manager):
        self.cuda_manager = cuda_manager
        self.tokenizer_manager = tokenizer_manager
        self.loaded_models = {}
        self.max_loaded_models = {
            ModelType.CHAT: 1,  # Un seul modèle principal
            ModelType.EMBEDDING: 2,  # Max 2 modèles d'embedding
            ModelType.SUMMARIZER: 1  # Un seul summarizer
        }
        
    async def load_model(
        self,
        model_name: str,
        model_type: ModelType,
        force_reload: bool = False
    ) -> LoadedModel:
        """Charge ou récupère un modèle selon son type."""
        try:
            # Vérifier si le modèle est déjà chargé
            model_key = f"{model_type.value}_{model_name}"
            if model_key in self.loaded_models and not force_reload:
                logger.info(f"Utilisation du modèle déjà chargé: {model_name}")
                return self.loaded_models[model_key]

            # Récupérer la configuration selon le type
            config = self._get_model_config(model_name, model_type)
            if not config:
                raise ValueError(f"Configuration non trouvée pour {model_name}")

            # Libérer de la mémoire si nécessaire
            await self._cleanup_for_model_type(model_type)

            # Charger le modèle selon son type
            if model_type == ModelType.CHAT:
                loaded_model = await self._load_chat_model(model_name, config)
            elif model_type == ModelType.EMBEDDING:
                loaded_model = await self._load_embedding_model(model_name, config)
            elif model_type == ModelType.SUMMARIZER:
                loaded_model = await self._load_summarizer_model(model_name, config)
            else:
                raise ValueError(f"Type de modèle non supporté: {model_type}")

            # Stocker le modèle chargé
            self.loaded_models[model_key] = loaded_model
            logger.info(f"Modèle {model_name} ({model_type.value}) chargé avec succès")

            return loaded_model

        except Exception as e:
            logger.error(f"Erreur chargement modèle {model_name}: {e}")
            raise

    async def _load_chat_model(self, model_name: str, config: Dict) -> LoadedModel:
        """Charge un modèle de chat."""
        try:
            logger.info(f"Chargement du modèle {model_name}")
            
            # Récupération des paramètres de base du modèle
            load_params = config["load_params"].copy()
            
            # Configuration de la quantization avec BitsAndBytesConfig
            if "quantization_config" in load_params:
                quant_config = load_params.pop("quantization_config")
                load_params["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=quant_config.get("load_in_4bit", True),
                    bnb_4bit_compute_dtype=quant_config.get("bnb_4bit_compute_dtype", torch.float16),
                    bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
                    bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
                    llm_int8_enable_fp32_cpu_offload=quant_config.get("llm_int8_enable_fp32_cpu_offload", True)
                )
            
            # Récupération des paramètres CUDA optimisés
            cuda_params = self.cuda_manager.get_model_load_parameters(model_name, ModelPriority.HIGH)
            
            # Fusionner les paramètres en gardant la priorité pour load_params
            cuda_params.update(load_params)
            load_params = cuda_params
            
            logger.debug(f"Paramètres de chargement: {load_params}")
            
            # Chargement avec gestion de la précision moderne
            with torch.amp.autocast(device_type='cuda', dtype=load_params.get('torch_dtype', torch.float16)):
                model = AutoModelForCausalLM.from_pretrained(
                    config["path"],
                    **load_params
                )

            # Configuration post-chargement
            model.eval()
            model.requires_grad_(False)

            return LoadedModel(
                model=model,
                tokenizer=self.tokenizer_manager.get_tokenizer(model_name, ModelType.CHAT),
                model_type=ModelType.CHAT,
                model_name=model_name,
                config=config,
                loaded_at=torch.cuda.current_stream().record_event(),
                device=next(model.parameters()).device
            )

        except Exception as e:
            logger.error(f"Erreur chargement modèle {model_name}: {e}")
            raise

    async def _load_summarizer_model(self, model_name: str, config: Dict) -> LoadedModel:
        """Charge un modèle de résumé."""
        try:
            logger.info(f"Chargement du summarizer {model_name}")
            
            # Récupération des paramètres de base du modèle
            load_params = config["load_params"].copy()
            
            # Configuration de la quantization avec BitsAndBytesConfig
            if "quantization_config" in load_params:
                quant_config = load_params.pop("quantization_config")
                load_params["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=quant_config["bnb_4bit_compute_dtype"],
                    bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
                    bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"]
                )

            # Récupération des paramètres CUDA optimisés
            cuda_params = self.cuda_manager.get_model_load_parameters(
                model_name,
                ModelPriority.MEDIUM
            )
            
            # Fusionner les paramètres
            cuda_params.update(load_params)
            load_params = cuda_params

            with torch.amp.autocast(device_type='cuda', dtype=load_params.get('torch_dtype', torch.float16)):
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    config["path"],
                    **load_params
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    config["path"],
                    use_fast=True
                )

            model.eval()
            model.requires_grad_(False)

            return LoadedModel(
                model=model,
                tokenizer=tokenizer,
                model_type=ModelType.SUMMARIZER,
                model_name=model_name,
                config=config,
                loaded_at=torch.cuda.current_stream().record_event(),
                device=next(model.parameters()).device
            )

        except Exception as e:
            logger.error(f"Erreur chargement summarizer {model_name}: {e}")
            raise

    async def _load_embedding_model(self, model_name: str, config: Dict) -> LoadedModel:
        """Charge un modèle d'embedding."""
        try:
            logger.info(f"Chargement du modèle d'embedding {model_name}")
            
            # Récupération des paramètres de base du modèle
            load_params = config["load_params"].copy()
            
            # Récupération des paramètres CUDA optimisés
            cuda_params = self.cuda_manager.get_model_load_parameters(
                model_name,
                ModelPriority.LOW
            )
            
            # Fusionner les paramètres
            cuda_params.update(load_params)
            device = self.cuda_manager.device

            # Les modèles d'embedding utilisent SentenceTransformer
            model = SentenceTransformer(
                config["path"],
                device=str(device)
            )

            # Optimisations pour l'inférence
            model.eval()

            # Note: Pas de quantization pour les modèles d'embedding car 
            # ils sont déjà optimisés par SentenceTransformer

            return LoadedModel(
                model=model,
                tokenizer=None,  # SentenceTransformer gère son propre tokenizer
                model_type=ModelType.EMBEDDING,
                model_name=model_name,
                config=config,
                loaded_at=torch.cuda.current_stream().record_event(),
                device=device
            )

        except Exception as e:
            logger.error(f"Erreur chargement embedding {model_name}: {e}")
            raise

    def _get_model_config(self, model_name: str, model_type: ModelType) -> Optional[Dict]:
        """Récupère la configuration d'un modèle selon son type."""
        if model_type == ModelType.CHAT:
            return AVAILABLE_MODELS.get(model_name)
        elif model_type == ModelType.EMBEDDING:
            return EMBEDDING_MODELS.get(model_name)
        elif model_type == ModelType.SUMMARIZER:
            return SUMMARIZER_MODELS.get(model_name)
        return None

    def _get_performance_config(self, model_name: str, model_type: ModelType) -> Dict:
        """Récupère la configuration de performance d'un modèle."""
        if model_type == ModelType.CHAT:
            return MODEL_PERFORMANCE_CONFIGS.get(model_name, {})
        elif model_type == ModelType.EMBEDDING:
            return EMBEDDING_PERFORMANCE_CONFIGS.get(model_name, {})
        elif model_type == ModelType.SUMMARIZER:
            return SUMMARIZER_PERFORMANCE_CONFIGS.get(model_name, {})
        return {}

    async def _cleanup_for_model_type(self, model_type: ModelType):
        """Nettoie la mémoire pour un type de modèle si nécessaire."""
        type_models = [k for k, v in self.loaded_models.items() 
                      if k.startswith(model_type.value)]
        
        if len(type_models) >= self.max_loaded_models[model_type]:
            oldest_model = min(
                type_models,
                key=lambda k: self.loaded_models[k].loaded_at
            )
            await self._unload_model(oldest_model)

    async def _unload_model(self, model_key: str):
        """Décharge un modèle de la mémoire."""
        try:
            if model_key not in self.loaded_models:
                logger.debug(f"Le modèle {model_key} n'est pas chargé, ignoré")
                return

            model = self.loaded_models.pop(model_key)
            
            # Nettoyage du modèle
            if model and hasattr(model, 'model'):
                try:
                    if hasattr(model.model, 'cpu'):
                        model.model.cpu()
                    del model.model
                    logger.debug(f"Modèle {model_key} déchargé et supprimé")
                except Exception as e:
                    logger.warning(f"Erreur lors du nettoyage du modèle {model_key}: {e}")

            # Nettoyage du cache CUDA
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.debug("Cache CUDA nettoyé")
                except Exception as e:
                    logger.warning(f"Erreur lors du nettoyage du cache CUDA: {e}")

            logger.info(f"Modèle {model_key} déchargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du déchargement du modèle {model_key}: {e}")

    def _apply_quantization(self, model: PreTrainedModel, quantization: str) -> PreTrainedModel:
        """Applique la quantization à un modèle."""
        try:
            if quantization == "bitsandbytes-4bit":
                model = model.half()  # Conversion en FP16
            return model
        except Exception as e:
            logger.warning(f"Échec quantization {quantization}: {e}")
            return model

    async def cleanup(self):
        """Nettoie toutes les ressources."""
        try:
            for model_key in list(self.loaded_models.keys()):
                await self._unload_model(model_key)
            torch.cuda.empty_cache()
            logger.info("Tous les modèles ont été nettoyés")
        except Exception as e:
            logger.error(f"Erreur nettoyage modèles: {e}")