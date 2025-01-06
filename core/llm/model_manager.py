# core/llm/model_manager.py
from typing import Dict, Optional, Tuple, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import psutil
import json
from pathlib import Path

from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .model_loader import ModelLoader

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
        
        # Configuration
        self.models_dir = Path(settings.MODEL_DIR)
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
            
            # Chargement des états des modèles si existe
            await self._load_model_states()
            
            self._initialized = True
            logger.info("ModelManager initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation ModelManager: {e}")
            raise

    async def load_model(
        self,
        model_name: str,
        force_reload: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Charge un modèle et son tokenizer.
        
        Args:
            model_name: Nom du modèle à charger
            force_reload: Force le rechargement même si déjà chargé
            
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]
        """
        if not self._initialized:
            raise RuntimeError("ModelManager non initialisé")
            
        try:
            start_time = datetime.utcnow()
            
            # Vérification de l'espace disponible
            if not self._check_system_resources():
                raise RuntimeError("Ressources système insuffisantes")
                
            # Chargement du modèle via le loader
            model, tokenizer = await self.model_loader.load_model(model_name)
            
            # Configuration optimisée du modèle après chargement
            model.eval()  # Mode évaluation
            if settings.USE_FP16:
                model = model.half()
                
            # Désactivation du gradient de manière permanente
            model.requires_grad_(False)
            
            # Configuration CUDA si disponible
            device = self.cuda_manager.current_device
            if device.type == "cuda":
                if settings.USE_FP16:
                    torch.cuda.amp.autocast('cuda', enabled=True).__enter__()
            
            # Mise à jour des informations
            self._update_model_info(
                model_name,
                model,
                datetime.utcnow() - start_time
            )
            
            # Sauvegarde des états
            await self._save_model_states()
            
            logger.info(f"Modèle {model_name} chargé et configuré avec succès")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle {model_name}: {e}")
            metrics.increment_counter("model_load_errors")
            raise

    async def change_model(
        self,
        new_model_name: str,
        keep_old: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Change le modèle actif.
        
        Args:
            new_model_name: Nom du nouveau modèle
            keep_old: Conserve l'ancien modèle en mémoire si possible
            
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]
        """
        try:
            if not keep_old:
                # Si on ne garde pas l'ancien, on force le nettoyage
                await self.model_loader._cleanup_oldest_model()
                
            # Chargement du nouveau modèle
            model, tokenizer = await self.load_model(new_model_name)
            
            logger.info(f"Changement de modèle effectué vers {new_model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Erreur changement modèle: {e}")
            raise

    def get_model_info(self, model_name: Optional[str] = None) -> Dict:
        """
        Récupère les informations sur un ou tous les modèles.
        
        Args:
            model_name: Nom du modèle spécifique ou None pour tous
            
        Returns:
            Dict: Informations sur le(s) modèle(s)
        """
        if model_name:
            return self.model_info.get(model_name, {})
        return self.model_info

    def get_current_model(self) -> Optional[str]:
        """Retourne le nom du modèle actif."""
        return self.model_loader.current_model_name if self.model_loader else None

    async def _load_model_states(self):
        """Charge l'état des modèles depuis le fichier."""
        try:
            if self.model_states_file.exists():
                data = json.loads(self.model_states_file.read_text())
                self.model_info = data.get("model_info", {})
                logger.info("États des modèles chargés")
        except Exception as e:
            logger.warning(f"Erreur chargement états modèles: {e}")
            self.model_info = {}

    async def _save_model_states(self):
        """Sauvegarde l'état des modèles."""
        try:
            data = {
                "model_info": self.model_info,
                "last_updated": datetime.utcnow().isoformat()
            }
            self.model_states_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Erreur sauvegarde états modèles: {e}")

    def _update_model_info(
        self,
        model_name: str,
        model: AutoModelForCausalLM,
        load_time: datetime
    ):
        """Met à jour les informations sur un modèle."""
        try:
            device = next(model.parameters()).device
            memory_stats = torch.cuda.get_device_properties(device).total_memory \
                if str(device) != "cpu" else None
            
            self.model_info[model_name] = {
                "last_loaded": datetime.utcnow().isoformat(),
                "load_time_seconds": load_time.total_seconds(),
                "device": str(device),
                "dtype": str(next(model.parameters()).dtype),
                "memory_used": str(memory_stats) if memory_stats else "N/A",
                "is_quantized": bool(settings.USE_8BIT or settings.USE_4BIT),
                "metadata": {
                    "architecture": model.config.architectures[0] if hasattr(model.config, 'architectures') else "unknown",
                    "revision": settings.MODEL_REVISION
                }
            }
        except Exception as e:
            logger.warning(f"Erreur mise à jour info modèle: {e}")

    def _check_system_resources(self) -> bool:
        """
        Vérifie si les ressources système sont suffisantes.
        Returns:
            bool: True si ok, False sinon
        """
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

    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            if self.model_loader:
                await self.model_loader.cleanup()
            
            # Sauvegarde finale des états
            await self._save_model_states()
            
            self._initialized = False
            logger.info("ModelManager nettoyé")
        except Exception as e:
            logger.error(f"Erreur nettoyage ModelManager: {e}")