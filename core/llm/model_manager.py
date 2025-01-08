# core/llm/model_manager.py
from typing import Dict, Optional, Tuple, List, Any
import torch
from datetime import datetime
import psutil
import json
import numpy as np
from pathlib import Path
import asyncio

from core.config.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .model_loader import ModelLoader, ModelType, LoadedModel
from .cuda_manager import ModelPriority
from core.config.models import (
    AVAILABLE_MODELS,
    EMBEDDING_MODELS,
    SUMMARIZER_MODELS,
    MODEL_PERFORMANCE_CONFIGS,
    EMBEDDING_PERFORMANCE_CONFIGS,
    SUMMARIZER_PERFORMANCE_CONFIGS
)

logger = get_logger("model_manager")

class ModelManager:
    """
    Gestionnaire centralisé des modèles.
    Gère le cycle de vie des modèles, leur chargement, et leur monitoring.
    """
    
    def __init__(self, cuda_manager, tokenizer_manager):
        """Initialise le gestionnaire de modèles."""
        self.cuda_manager = cuda_manager
        self.tokenizer_manager = tokenizer_manager
        self.model_loader = None
        self._initialized = False
        self.model_info = {}
        
        # États actuels des modèles
        self.current_models = {
            ModelType.CHAT: None,
            ModelType.EMBEDDING: None,
            ModelType.SUMMARIZER: None
        }
        
        # Configuration
        self.models_dir = Path(settings.MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_states_file = self.models_dir / "model_states.json"
        
    async def initialize(self):
        """Initialise le gestionnaire de modèles."""
        if self._initialized:
            return

        try:
            logger.info("Initialisation du ModelManager")
            
            # Création du ModelLoader
            self.model_loader = ModelLoader(
                self.cuda_manager,
                self.tokenizer_manager
            )
            
            # Chargement des états des modèles
            await self._load_model_states()
            
            # Chargement des modèles par défaut
            await self._load_default_models()
            
            self._initialized = True
            logger.info("ModelManager initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation ModelManager: {e}")
            raise

    async def _load_default_models(self):
        """Charge les modèles par défaut pour chaque type."""
        try:
            # Modèle de chat principal
            self.current_models[ModelType.CHAT] = await self.model_loader.load_model(
                settings.MODEL_NAME,
                ModelType.CHAT
            )

            # Modèle d'embedding
            self.current_models[ModelType.EMBEDDING] = await self.model_loader.load_model(
                settings.EMBEDDING_MODEL,
                ModelType.EMBEDDING
            )

            # Modèle de summarization
            if settings.MODEL_NAME_SUMMARIZER:
                self.current_models[ModelType.SUMMARIZER] = await self.model_loader.load_model(
                    settings.MODEL_NAME_SUMMARIZER,
                    ModelType.SUMMARIZER
                )
                
        except Exception as e:
            logger.error(f"Erreur chargement modèles par défaut: {e}")
            raise

    async def change_model(
        self,
        model_name: str,
        model_type: ModelType,
        keep_old: bool = False
    ) -> LoadedModel:
        """
        Change le modèle actif pour un type donné.
        
        Args:
            model_name: Nom du nouveau modèle
            model_type: Type de modèle
            keep_old: Conserve l'ancien modèle en mémoire si possible
        """
        try:
            # Vérification de la disponibilité du modèle
            if not self.is_model_available(model_name, model_type):
                raise ValueError(f"Modèle {model_name} non disponible pour le type {model_type}")

            # Vérification des ressources
            if not await self._check_system_resources():
                raise RuntimeError("Ressources système insuffisantes")
                
            # Chargement du nouveau modèle
            model = await self.model_loader.load_model(
                model_name=model_name,
                model_type=model_type
            )

            # Nettoyage de l'ancien modèle si nécessaire
            if not keep_old:
                old_model = self.current_models[model_type]
                if old_model:
                    await self.model_loader._unload_model(
                        f"{model_type.value}_{old_model.model_name}"
                    )

            # Mise à jour du modèle courant
            self.current_models[model_type] = model
            
            # Mise à jour des informations
            self._update_model_info(
                model_name=model_name,
                model_type=model_type,
                model=model
            )
            
            # Sauvegarde des états
            await self._save_model_states()
            
            return model
            
        except Exception as e:
            logger.error(f"Erreur changement modèle: {e}")
            raise

    def is_model_available(self, model_name: str, model_type: ModelType) -> bool:
        """Vérifie si un modèle est disponible."""
        if model_type == ModelType.CHAT:
            return model_name in AVAILABLE_MODELS
        elif model_type == ModelType.EMBEDDING:
            return model_name in EMBEDDING_MODELS
        elif model_type == ModelType.SUMMARIZER:
            return model_name in SUMMARIZER_MODELS
        return False

    async def get_available_models(self) -> Dict[str, List[str]]:
        """Retourne la liste des modèles disponibles par type."""
        return {
            "chat_models": list(AVAILABLE_MODELS.keys()),
            "embedding_models": list(EMBEDDING_MODELS.keys()),
            "summarizer_models": list(SUMMARIZER_MODELS.keys())
        }

    def get_current_model(self, model_type: ModelType = ModelType.CHAT) -> Optional[str]:
        """Retourne le nom du modèle actif pour un type donné."""
        model = self.current_models.get(model_type)
        return model.model_name if model else None

    def get_model_info(self, model_type: Optional[ModelType] = None) -> Dict:
        """
        Récupère les informations sur un ou tous les modèles.
        """
        if model_type:
            return self.model_info.get(f"{model_type.value}", {})
        return self.model_info

    def _update_model_info(
        self,
        model_name: str,
        model_type: ModelType,
        model: LoadedModel
    ):
        """Met à jour les informations sur un modèle."""
        try:
            # Vérification que le modèle est valide avant de l'utiliser
            if not model or not hasattr(model, 'model'):
                logger.warning(f"Modèle invalide pour {model_name}")
                return

            self.model_info[f"{model_type.value}_{model_name}"] = {
                "last_loaded": datetime.utcnow().isoformat(),
                "device": str(model.device),
                "model_type": model_type.value,
                "config": self._get_model_config(model_name, model_type),
                "performance_config": self._get_performance_config(model_name, model_type),
                "metadata": {
                    "is_quantized": hasattr(model.model, "is_quantized"),
                    "requires_grad": next(model.model.parameters()).requires_grad
                }
            }
        except Exception as e:
            logger.warning(f"Erreur mise à jour info modèle: {e}")

    def _get_model_config(self, model_name: str, model_type: ModelType) -> Dict:
        """Récupère la configuration d'un modèle."""
        if model_type == ModelType.CHAT:
            return AVAILABLE_MODELS.get(model_name, {})
        elif model_type == ModelType.EMBEDDING:
            return EMBEDDING_MODELS.get(model_name, {})
        elif model_type == ModelType.SUMMARIZER:
            return SUMMARIZER_MODELS.get(model_name, {})
        return {}

    def _get_performance_config(self, model_name: str, model_type: ModelType) -> Dict:
        """Récupère la configuration de performance d'un modèle."""
        if model_type == ModelType.CHAT:
            return MODEL_PERFORMANCE_CONFIGS.get(model_name, {})
        elif model_type == ModelType.EMBEDDING:
            return EMBEDDING_PERFORMANCE_CONFIGS.get(model_name, {})
        elif model_type == ModelType.SUMMARIZER:
            return SUMMARIZER_PERFORMANCE_CONFIGS.get(model_name, {})
        return {}

    async def _check_system_resources(self) -> bool:
        """Vérifie si les ressources système sont suffisantes."""
        try:
            # Vérification RAM
            memory = psutil.virtual_memory()
            if memory.percent > 95:  # Plus de 95% utilisé
                logger.warning("RAM presque saturée")
                return False
                
            # Vérification GPU
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    if torch.cuda.memory_reserved(i) / torch.cuda.get_device_properties(i).total_memory > 0.95:
                        logger.warning(f"VRAM GPU {i} presque saturée")
                        return False
                        
            # Vérification espace disque
            disk = psutil.disk_usage(str(self.models_dir))
            if disk.percent > 95:  # Plus de 95% utilisé
                logger.warning("Espace disque presque saturé")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur vérification ressources: {e}")
            return False

    async def _save_model_states(self):
        """Sauvegarde l'état des modèles."""
        try:
            data = {
                "model_info": self.model_info,
                "last_updated": datetime.utcnow().isoformat(),
                "model_states": {
                    model_type.value: model.model_name if model else None
                    for model_type, model in self.current_models.items()
                }
            }
            
            json_str = json.dumps(
                data, 
                indent=2, 
                cls=ModelStateEncoder
            )
            self.model_states_file.write_text(json_str)
            logger.debug("États des modèles sauvegardés avec succès")
            
        except Exception as e:
            logger.warning(f"Erreur sauvegarde états modèles: {e}")

    async def _load_model_states(self):
        """Charge l'état des modèles depuis le fichier."""
        try:
            if self.model_states_file.exists():
                data = json.loads(self.model_states_file.read_text())
                self.model_info = data.get("model_info", {})
                
                # Conversion des types si nécessaire
                for model_key, info in self.model_info.items():
                    if "device" in info:
                        info["device"] = torch.device(info["device"])
                    if "metadata" in info and "requires_grad" in info["metadata"]:
                        info["metadata"]["requires_grad"] = bool(info["metadata"]["requires_grad"])
                        
                logger.info("États des modèles chargés")
        except Exception as e:
            logger.warning(f"Erreur chargement états modèles: {e}")
            self.model_info = {}
            
    async def test_model_health(
        self,
        model_type: ModelType,
        test_input: str
    ) -> bool:
        """Teste la santé d'un modèle."""
        try:
            model = self.current_models.get(model_type)
            if not model:
                return False

            with torch.no_grad():
                if model_type == ModelType.CHAT:
                    # Test simple de génération
                    tokenized = self.tokenizer_manager.tokenizer(
                        test_input,
                        return_tensors="pt",
                        truncation=True,
                        max_length=32
                    ).to(model.device)
                    
                    _ = model.model.generate(
                        **tokenized,
                        max_length=32,
                        num_return_sequences=1
                    )

                elif model_type == ModelType.EMBEDDING:
                    # Test de création d'embedding
                    _ = model.model.encode(test_input)

                else:  # SUMMARIZER
                    # Test de génération de résumé
                    tokenized = model.tokenizer(
                        test_input,
                        return_tensors="pt",
                        truncation=True,
                        max_length=32
                    ).to(model.device)
                    
                    _ = model.model.generate(
                        **tokenized,
                        max_length=32,
                        num_return_sequences=1
                    )

            return True

        except Exception as e:
            logger.error(f"Erreur test santé modèle {model_type}: {e}")
            return False

    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            if self.model_loader:
                await self.model_loader.cleanup()
            
            # Sauvegarde finale des états
            await self._save_model_states()
            
            self.current_models = {
                model_type: None for model_type in ModelType
            }
            self._initialized = False
            logger.info("ModelManager nettoyé")
        except Exception as e:
            logger.error(f"Erreur nettoyage ModelManager: {e}")
        
class ModelStateEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            if isinstance(obj, torch.dtype):
                return str(obj)
            elif isinstance(obj, ModelPriority):
                return obj.value
            elif isinstance(obj, (torch.Tensor, np.ndarray)):
                return obj.tolist()
            elif isinstance(obj, (datetime, Path)):
                return str(obj)
            elif hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, '__dict__'):
                return {
                    k: v for k, v in obj.__dict__.items() 
                    if not k.startswith('_')
                }
            return str(obj)
        except Exception:
            return str(obj)