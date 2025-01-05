# core/llm/model_loader.py
from typing import Dict, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("model_loader")

class ModelLoader:
    """Gestionnaire de chargement des modèles."""
    
    def __init__(self, cuda_manager, tokenizer_manager):
        self.cuda_manager = cuda_manager
        self.tokenizer_manager = tokenizer_manager
        self.current_model_name = None
        self.loaded_models = {}
        self.max_loaded_models = 2  # Nombre maximum de modèles chargés simultanément
        
    async def load_model(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Charge ou récupère un modèle et son tokenizer.
        
        Args:
            model_name: Nom du modèle à charger
            
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: Le modèle et son tokenizer
        """
        try:
            # Vérifier si le modèle est déjà chargé
            if model_name in self.loaded_models:
                logger.info(f"Utilisation du modèle déjà chargé: {model_name}")
                return self.loaded_models[model_name]
            
            # Libération de mémoire si nécessaire
            if len(self.loaded_models) >= self.max_loaded_models:
                await self._cleanup_oldest_model()
            
            logger.info(f"Chargement du nouveau modèle: {model_name}")
            
            # Obtention des paramètres optimisés pour le modèle
            load_params = self._get_model_config(model_name)
            
            # Chargement du modèle avec gestion de la mémoire
            is_mistral = "mistral" in model_name.lower()
            with torch.amp.autocast('cuda', enabled=settings.USE_FP16):
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    revision=settings.MODEL_REVISION,
                    **load_params
                )

            # Configuration post-chargement
            if settings.USE_FP16:
                model = model.half()
            model.eval()  # Mode évaluation
            
            # Stockage du modèle
            self.loaded_models[model_name] = (model, self.tokenizer_manager.tokenizer)
            self.current_model_name = model_name
            
            # Log des informations de device et mémoire
            device_info = {
                "model_device": next(model.parameters()).device,
                "dtype": next(model.parameters()).dtype,
                "memory_stats": self.cuda_manager.memory_stats
            }
            logger.info("État du modèle après chargement:")
            for key, value in device_info.items():
                logger.info(f"  {key}: {value}")
            
            return model, self.tokenizer_manager.tokenizer
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle {model_name}: {e}")
            raise

    def _get_model_config(self, model_name: str) -> Dict:
        """
        Obtient la configuration optimale pour le chargement du modèle.
        """
        is_mistral = "mistral" in model_name.lower()
        
        # Configuration de base commune
        config = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16 if is_mistral else torch.float16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True
        }

        # Configuration de la mémoire
        if self.cuda_manager.config.max_memory:
            config["max_memory"] = self.cuda_manager.config.max_memory

        # Configuration de la quantization
        if settings.USE_8BIT:
            config.update({
                "load_in_8bit": True,
                "llm_int8_enable_fp32_cpu_offload": True
            })
        elif settings.USE_4BIT:
            config.update({
                "load_in_4bit": True,
                "bnb_4bit_quant_type": settings.BNB_4BIT_QUANT_TYPE,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True
            })

        # Configuration du flash attention pour Mistral
        if is_mistral and settings.USE_FLASH_ATTENTION:
            config["attn_implementation"] = "flash_attention_2"

        # Ajout des tokens spéciaux depuis le tokenizer manager
        config.update({
            "pad_token_id": self.tokenizer_manager.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer_manager.tokenizer.eos_token_id
        })

        return config
            
    async def _cleanup_oldest_model(self):
        """
        Libère la mémoire en supprimant le plus ancien modèle chargé.
        """
        if not self.loaded_models:
            return
            
        # Trouve le modèle le plus ancien (différent du modèle actuel)
        models_to_remove = [name for name in self.loaded_models.keys() 
                          if name != self.current_model_name]
        
        if models_to_remove:
            oldest_model = models_to_remove[0]
            model, _ = self.loaded_models.pop(oldest_model)
            
            # Nettoyage explicite
            model.cpu()  # Déplacer sur CPU avant suppression
            del model
            torch.cuda.empty_cache()
            logger.info(f"Modèle libéré: {oldest_model}")
            
    async def cleanup(self):
        """
        Nettoie tous les modèles chargés.
        """
        try:
            for model_name in list(self.loaded_models.keys()):
                model, _ = self.loaded_models.pop(model_name)
                model.cpu()
                del model
                
            torch.cuda.empty_cache()
            self.current_model_name = None
            logger.info("Tous les modèles ont été nettoyés")
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des modèles: {e}")
            raise