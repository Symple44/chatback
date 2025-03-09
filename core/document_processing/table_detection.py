# core/document_processing/table_detection.py
from typing import List, Dict, Any, Union, Optional, Tuple
import torch
import os
import asyncio
from pathlib import Path
import numpy as np
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import logging

from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings
from core.llm.cuda_manager import ModelPriority

# Réduire le niveau de log pour masquer les avertissements non critiques
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = get_logger("table_detection")

class TableDetectionModel:
    """Détecteur de tableaux utilisant un modèle DETR pré-entraîné."""
    
    def __init__(self, cuda_manager=None):
        """
        Initialise le détecteur de tableaux.
        
        Args:
            cuda_manager: Gestionnaire CUDA pour l'allocation mémoire (optionnel)
        """
        self.model_name = settings.document.TABLE_DETECTION_MODEL
        self.processor = None
        self.model = None
        self.cuda_manager = cuda_manager
        self._initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label = None
        
        # Configurations
        self.threshold = settings.document.TABLE_DETECTION_THRESHOLD
        self.max_tables = settings.document.TABLE_DETECTION_MAX_TABLES
        
        # Dossier pour les modèles
        self.models_dir = Path(settings.MODELS_DIR) / "table_detection"
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialise le modèle de détection de tableaux."""
        if self._initialized:
            return
            
        try:
            logger.info(f"Chargement du modèle de détection de tableaux: {self.model_name}")
            
            # Allouer la mémoire GPU si le cuda_manager est disponible
            if self.cuda_manager:
                allocated = self.cuda_manager.allocate_memory(ModelPriority.LOW)
                if not allocated:
                    logger.warning("Mémoire insuffisante pour le GPU, utilisation du CPU")
                    self.device = torch.device("cpu")
            
            # Chargement du processeur d'image avec configuration mise à jour
            # Pour éviter l'avertissement "max_size" déprécié
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                self.processor = DetrImageProcessor.from_pretrained(
                    self.model_name,
                    size={"longest_edge": 800},  # Remplace max_size
                    do_resize=True,
                    do_normalize=True
                )
                
                # Configuration de chargement adaptée
                load_params = {
                    "device_map": "auto" if self.device.type == "cuda" else "cpu",
                    "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                    "local_files_only": False,
                    "cache_dir": str(self.models_dir),
                    # Forcer le chargement en DETR pour éviter l'avertissement de type
                    "_from_model_config": True,
                    "low_cpu_mem_usage": True
                }
                
                self.model = DetrForObjectDetection.from_pretrained(
                    self.model_name,
                    **load_params
                )
                
                # Mise en mode évaluation et désactivation du calcul de gradient
                self.model.eval()
                self.model.requires_grad_(False)
                
                # Récupération du mapping d'étiquettes
                self.id2label = self.model.config.id2label
                
                logger.info(f"Modèle de détection de tableaux chargé sur {self.device}")
                self._initialized = True
                
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du modèle de détection: {e}")
            raise
    
    async def detect_tables(
        self, 
        image: Union[np.ndarray, str, Path], 
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Détecte les tableaux dans une image.
        
        Args:
            image: Image au format numpy array (OpenCV), chemin ou PIL Image
            threshold: Seuil de confiance pour la détection (0-1)
            
        Returns:
            Liste de coordonnées et scores des tableaux détectés
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            with metrics.timer("table_detection"):
                # Convertir l'image au bon format
                if isinstance(image, (str, Path)):
                    pil_image = Image.open(image)
                elif isinstance(image, np.ndarray):
                    # Conversion en PIL Image
                    if len(image.shape) == 2:  # Image en niveaux de gris
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif image.shape[2] == 4:  # Image RGBA
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    pil_image = Image.fromarray(image)
                else:
                    pil_image = image  # Supposons que c'est déjà une PIL Image
                
                # Préparation de l'image pour le modèle
                # Utilisation du size avec longest_edge au lieu de max_size
                inputs = self.processor(
                    images=pil_image, 
                    return_tensors="pt",
                    size={"longest_edge": 800},  # Correction de la dépréciation
                    do_resize=True,
                    do_normalize=True
                )
                
                if self.device.type == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Inférence avec torch.no_grad() pour économiser de la mémoire
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Post-traitement des résultats
                threshold = threshold or self.threshold
                target_sizes = torch.tensor([pil_image.size[::-1]])
                if self.device.type == "cuda":
                    target_sizes = target_sizes.to(self.device)
                
                results = self.processor.post_process_object_detection(
                    outputs, 
                    threshold=threshold, 
                    target_sizes=target_sizes
                )[0]
                
                # Formatage des résultats
                detections = []
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    # Convertir en coordonnées x, y, w, h
                    x, y, x2, y2 = box.tolist()
                    width = x2 - x
                    height = y2 - y
                    
                    # Vérifier si le label est 'table'
                    label_name = self.id2label.get(label.item(), "unknown")
                    
                    # Ne conserver que les détections de type table
                    if label_name.lower() == 'table':
                        detections.append({
                            "x": int(x),
                            "y": int(y),
                            "width": int(width),
                            "height": int(height),
                            "label": label_name,
                            "score": float(score)
                        })
                
                # Trier par score et limiter le nombre
                detections = sorted(detections, key=lambda x: x["score"], reverse=True)
                detections = detections[:self.max_tables]
                
                metrics.increment_counter("tables_detected", len(detections))
                logger.debug(f"Détection de {len(detections)} tableaux avec seuil {threshold}")
                
                return detections
        
        except Exception as e:
            logger.error(f"Erreur lors de la détection des tableaux: {e}")
            metrics.increment_counter("table_detection_errors")
            return []
    
    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            if self.model is not None:
                # Procédure de nettoyage explicite pour libérer la mémoire
                self.model = self.model.to("cpu") if torch.cuda.is_available() else self.model
                self.model = None
            
            if self.processor is not None:
                self.processor = None
            
            # Libération de la mémoire CUDA si nécessaire
            if self.cuda_manager and self.device.type == "cuda":
                self.cuda_manager.release_memory(ModelPriority.LOW)
                
            # Forcer la libération de la mémoire CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            self._initialized = False
            
            logger.info("Ressources du modèle de détection nettoyées")
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du modèle de détection: {e}")