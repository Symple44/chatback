# core/llm/model_loader.py
from typing import Dict, Tuple, Optional, Any, List
import asyncio
import gc
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
    MODEL_PERFORMANCE_CONFIGS
)
from core.utils.logger import get_logger
from core.llm.cuda_manager import ModelPriority
from core.llm.tokenizer_manager import TokenizerType

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
            ModelType.CHAT: 1,        # Un seul modèle principal
            ModelType.EMBEDDING: 2,    # Max 2 modèles d'embedding
            ModelType.SUMMARIZER: 1    # Un seul summarizer
        }
        self._initialization_lock = asyncio.Lock()
        self._model_locks = {}
        
    async def load_model(
        self,
        model_name: str,
        model_type: ModelType,
        force_reload: bool = False
    ) -> LoadedModel:
        """Charge ou récupère un modèle selon son type."""
        try:
            # Vérification de la configuration
            model_config = self._get_model_config(model_name, model_type)
            if not model_config:
                raise ValueError(f"Configuration non trouvée pour {model_name}")

            # Clé unique pour le modèle
            model_key = f"{model_type.value}_{model_name}"
            
            # Verrou spécifique au modèle
            if model_key not in self._model_locks:
                self._model_locks[model_key] = asyncio.Lock()
            
            async with self._model_locks[model_key]:
                # Vérifier si le modèle est déjà chargé
                if model_key in self.loaded_models and not force_reload:
                    logger.info(f"Utilisation du modèle déjà chargé: {model_name}")
                    return self.loaded_models[model_key]

                # Déterminer la priorité du modèle
                model_priority = {
                    ModelType.CHAT: ModelPriority.HIGH,
                    ModelType.SUMMARIZER: ModelPriority.MEDIUM,
                    ModelType.EMBEDDING: ModelPriority.LOW
                }[model_type]

                # Vérifier la mémoire disponible
                if not self.cuda_manager._check_memory_availability(model_priority):
                    raise RuntimeError(f"Mémoire insuffisante pour charger {model_name}")

                # Libérer de la mémoire si nécessaire
                await self._cleanup_for_model_type(model_type)

                # Charger le modèle selon son type
                if model_type == ModelType.CHAT:
                    loaded_model = await self._load_chat_model(model_name, model_config)
                elif model_type == ModelType.EMBEDDING:
                    loaded_model = await self._load_embedding_model(model_name, model_config)
                elif model_type == ModelType.SUMMARIZER:
                    loaded_model = await self._load_summarizer_model(model_name, model_config)
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
            
            # 1. Récupération de toutes les configurations pertinentes
            model_config = AVAILABLE_MODELS[model_name]
            performance_config = MODEL_PERFORMANCE_CONFIGS[model_name]
            
            # 2. Configuration du tokenizer
            tokenizer = self.tokenizer_manager.get_tokenizer(model_name, ModelType.CHAT)
            if not tokenizer:
                raise ValueError(f"Tokenizer non trouvé pour {model_name}")
            
            # Configuration explicite pad_token comme eos_token
            if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("pad_token_id configuré sur eos_token_id")
            
            # 3. Construction des paramètres de chargement
            load_params = model_config["load_params"].copy()
            
            # 4. Configuration de la quantization
            if "quantization_config" in load_params:
                quant_config = load_params.pop("quantization_config")
                load_params["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=quant_config.get("load_in_4bit", True),
                    bnb_4bit_compute_dtype=quant_config.get("bnb_4bit_compute_dtype", torch.float16),
                    bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
                    bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
                    llm_int8_enable_fp32_cpu_offload=quant_config.get("llm_int8_enable_fp32_cpu_offload", True),
                    quant_method="llm_int8"  # Modifié ici
                )
            
            # 5. Récupération et fusion des paramètres CUDA optimisés
            cuda_params = self._get_model_load_parameters(
                model_name=model_name,
                model_type=ModelType.CHAT
            )
            
            # 6. Ajout des paramètres de performance
            performance_params = {
                "torch_dtype": performance_config.get("execution", {}).get("torch_dtype", torch.float16),
                "low_cpu_mem_usage": True,
                "device_map": cuda_params.get("device_map", "auto")
            }
            
            # 7. Fusion de tous les paramètres dans le bon ordre
            final_params = {
                **load_params,          # Paramètres de base du modèle
                **cuda_params,          # Paramètres CUDA
                **performance_params    # Paramètres de performance
            }
            
            logger.debug(f"Paramètres de chargement finaux: {final_params}")
            
            # 8. Chargement avec gestion de la précision
            with torch.amp.autocast(device_type='cuda', dtype=final_params.get('torch_dtype', torch.float16)):
                model = AutoModelForCausalLM.from_pretrained(
                    model_config["path"],
                    **final_params
                )

                # Configuration du pad_token_id dans le modèle
                if model.config.pad_token_id is None:
                    model.config.pad_token_id = model.config.eos_token_id

            # 9. Configuration post-chargement
            model.eval()
            model.requires_grad_(False)
            
            # 10. Optimisations spécifiques selon la configuration de performance
            if performance_config.get("optimization", {}).get("enable_jit", False):
                model = torch.jit.optimize_for_inference(model)
                
            if performance_config.get("optimization", {}).get("memory_efficient_attention", True):
                model.config.use_cache = False

            # 11. Création du LoadedModel avec toutes les métadonnées
            loaded_model = LoadedModel(
                model=model,
                tokenizer=tokenizer,
                model_type=ModelType.CHAT,
                model_name=model_name,
                config={
                    **model_config,
                    "performance_config": performance_config,
                    "cuda_config": cuda_params
                },
                loaded_at=datetime.utcnow(),
                device=next(model.parameters()).device
            )

            logger.info(f"Modèle {model_name} chargé avec succès sur {loaded_model.device}")
            return loaded_model

        except Exception as e:
            logger.error(f"Erreur chargement modèle {model_name}: {e}")
            raise

    async def _load_summarizer_model(self, model_name: str, config: Dict) -> LoadedModel:
        """Charge un modèle de résumé."""
        try:
            logger.info(f"Chargement du summarizer {model_name}")
            
            # 1. Récupération des configurations
            model_config = SUMMARIZER_MODELS[model_name]
            performance_config = SUMMARIZER_PERFORMANCE_CONFIGS[model_name]
            
            # 2. Configuration du tokenizer
            tokenizer = self.tokenizer_manager.get_tokenizer(
                model_name=model_name,
                tokenizer_type=TokenizerType.SUMMARIZER
            )
            if not tokenizer:
                raise ValueError(f"Tokenizer non trouvé pour {model_name}")

            # 3. Construction des paramètres de base
            load_params = model_config["load_params"].copy()
            
            # 4. Configuration de la quantization
            if "quantization_config" in load_params:
                quant_config = load_params.pop("quantization_config")
                load_params["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=quant_config["bnb_4bit_compute_dtype"],
                    bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
                    bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"]
                )

            # 5. Paramètres CUDA
            cuda_params = self._get_model_load_parameters(
                model_name=model_name,
                model_type=ModelType.SUMMARIZER
            )
            
            # 6. Paramètres de performance
            performance_params = {
                "torch_dtype": performance_config.get("execution", {}).get("torch_dtype", torch.float16),
                "low_cpu_mem_usage": True,
                "device_map": cuda_params.get("device_map", "auto")
            }

            # 7. Fusion des paramètres
            final_params = {
                **load_params,
                **cuda_params,
                **performance_params
            }

            # 8. Paramètres spécifiques pour T5/MT5
            if any(prefix in model_name.lower() for prefix in ['t5', 'mt5']):
                final_params.update({
                    "return_dict": True,
                    "output_hidden_states": False,
                    "output_attentions": False,
                })

            # 9. Chargement du modèle
            with torch.amp.autocast(device_type='cuda', dtype=final_params.get('torch_dtype', torch.float16)):
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_config["path"],
                    **final_params
                )

            # 10. Configuration post-chargement
            model.eval()
            model.requires_grad_(False)

            # 11. Optimisations spécifiques
            if performance_config.get("optimization", {}).get("memory_efficient_attention", True):
                model.config.use_cache = False

            # 12. Création du LoadedModel
            loaded_model = LoadedModel(
                model=model,
                tokenizer=tokenizer,
                model_type=ModelType.SUMMARIZER,
                model_name=model_name,
                config={
                    **model_config,
                    "performance_config": performance_config,
                    "cuda_config": cuda_params
                },
                loaded_at=datetime.utcnow(),
                device=next(model.parameters()).device
            )

            logger.info(f"Summarizer {model_name} chargé avec succès sur {loaded_model.device}")
            return loaded_model

        except Exception as e:
            logger.error(f"Erreur chargement summarizer {model_name}: {e}")
            raise

    async def _load_embedding_model(self, model_name: str, config: Dict) -> LoadedModel:
        """Charge un modèle d'embedding."""
        try:
            logger.info(f"Chargement du modèle d'embedding {model_name}")
            
            # 1. Récupération des configurations
            model_config = EMBEDDING_MODELS[model_name]
            performance_config = EMBEDDING_PERFORMANCE_CONFIGS[model_name]
            
            # 2. Configuration CUDA et device
            cuda_params = self._get_model_load_parameters(
                model_name=model_name,
                model_type=ModelType.EMBEDDING
            )
            device = self.cuda_manager.device

            # 3. Construction des paramètres de chargement
            load_params = {
                **model_config["load_params"],
                "device_map": cuda_params.get("device_map", "auto"),
                "torch_dtype": performance_config.get("execution", {}).get("torch_dtype", torch.float16)
            }

            # 4. Configuration des paramètres d'embedding
            embedding_params = model_config.get("generation_config", {}).get("embedding_params", {})
            
            # 5. Chargement du modèle avec SentenceTransformer
            with torch.amp.autocast(device_type='cuda', dtype=load_params.get('torch_dtype', torch.float16)):
                model = SentenceTransformer(
                    model_config["path"],
                    device=str(device)
                )

                # Application des paramètres d'embedding
                if hasattr(model, 'normalize_embeddings'):
                    model.normalize_embeddings = embedding_params.get("normalize_embeddings", True)
                
                if hasattr(model, 'pooling_strategy'):
                    model.pooling_strategy = embedding_params.get("pooling_strategy", "mean")

            # 6. Optimisations post-chargement
            model.eval()
            
            if performance_config.get("optimization", {}).get("enable_jit", False):
                model = torch.jit.optimize_for_inference(model)

            # 7. Création du LoadedModel
            loaded_model = LoadedModel(
                model=model,
                tokenizer=None,  # SentenceTransformer gère son propre tokenizer
                model_type=ModelType.EMBEDDING,
                model_name=model_name,
                config={
                    **model_config,
                    "performance_config": performance_config,
                    "cuda_config": cuda_params
                },
                loaded_at=datetime.utcnow(),
                device=device
            )

            logger.info(f"Modèle d'embedding {model_name} chargé avec succès sur {device}")
            return loaded_model

        except Exception as e:
            logger.error(f"Erreur chargement embedding {model_name}: {e}")
            raise
    
    def _get_model_load_parameters(
        self,
        model_name: str,
        model_type: ModelType
    ) -> Dict[str, Any]:
        """Retourne les paramètres optimaux pour le chargement du modèle."""
        model_config = self._get_model_config(model_name, model_type)
        load_params = model_config.get("load_params", {}).copy()

        # Vérification de la mémoire via CUDAManager
        priority = {
            ModelType.CHAT: ModelPriority.HIGH,
            ModelType.SUMMARIZER: ModelPriority.MEDIUM,
            ModelType.EMBEDDING: ModelPriority.LOW
        }[model_type]

        if not self.cuda_manager._check_memory_availability(priority):
            # Si pas assez de VRAM, forcer le chargement sur CPU
            load_params["device_map"] = {"": "cpu"}
            
        return load_params

    def _get_model_config(self, model_name: str, model_type: ModelType) -> Optional[Dict]:
        """Récupère la configuration complète d'un modèle."""
        try:
            if model_type == ModelType.CHAT:
                return AVAILABLE_MODELS.get(model_name)
            elif model_type == ModelType.EMBEDDING:
                return EMBEDDING_MODELS.get(model_name)
            elif model_type == ModelType.SUMMARIZER:
                return SUMMARIZER_MODELS.get(model_name)
            return None
        except Exception as e:
            logger.error(f"Erreur récupération config modèle {model_name}: {e}")
            return None

    def _get_performance_config(self, model_name: str, model_type: ModelType) -> Dict:
        """Récupère la configuration de performance d'un modèle."""
        try:
            if model_type == ModelType.CHAT:
                return MODEL_PERFORMANCE_CONFIGS.get(model_name, {})
            elif model_type == ModelType.EMBEDDING:
                return EMBEDDING_PERFORMANCE_CONFIGS.get(model_name, {})
            elif model_type == ModelType.SUMMARIZER:
                return SUMMARIZER_PERFORMANCE_CONFIGS.get(model_name, {})
            return {}
        except Exception as e:
            logger.error(f"Erreur récupération config performance {model_name}: {e}")
            return {}

    async def _cleanup_for_model_type(self, model_type: ModelType) -> None:
        """Nettoie la mémoire pour un type de modèle si nécessaire."""
        try:
            # Récupération des modèles du type spécifié
            type_models = [k for k in self.loaded_models.keys() 
                         if k.startswith(f"{model_type.value}_")]
            
            max_models = self.max_loaded_models[model_type]
            
            # Si on dépasse la limite, on décharge les plus anciens
            if len(type_models) >= max_models:
                # Trier par date de chargement
                sorted_models = sorted(
                    type_models,
                    key=lambda k: self.loaded_models[k].loaded_at
                )
                
                # Décharger les modèles en trop
                for model_key in sorted_models[:(len(type_models) - max_models + 1)]:
                    await self._unload_model(model_key)
                    
                # Forcer le garbage collector
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.error(f"Erreur nettoyage type {model_type}: {e}")

    async def _unload_model(self, model_key: str) -> None:
        """Décharge un modèle de la mémoire."""
        try:
            if model_key not in self.loaded_models:
                return

            model = self.loaded_models.pop(model_key)
            
            if model and hasattr(model, 'model'):
                try:
                    # Forcer le transfert sur CPU avant de supprimer
                    if hasattr(model.model, 'cpu'):
                        model.model = model.model.cpu()
                    # Forcer le garbage collection
                    model.model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.warning(f"Erreur nettoyage modèle {model_key}: {e}")

            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Erreur nettoyage cache CUDA: {e}")

            # Libérer le verrou associé
            if model_key in self._model_locks:
                self._model_locks.pop(model_key)

            logger.info(f"Modèle {model_key} déchargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur déchargement modèle {model_key}: {e}")

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
            logger.info("Début du nettoyage des modèles...")
            
            # Décharger tous les modèles
            for model_key in list(self.loaded_models.keys()):
                await self._unload_model(model_key)
                
            # Nettoyage final
            self.loaded_models.clear()
            self._model_locks.clear()
            
            # Forcer le garbage collector
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Nettoyage des modèles terminé")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage ModelLoader: {e}")
            raise