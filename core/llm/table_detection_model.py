# core/llm/table_detection_model.py
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
import warnings

from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings
from core.llm.cuda_manager import ModelPriority  # Import inchangé mais plus cohérent dans ce dossier

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
        # Lire la configuration depuis settings
        self.model_name = settings.table_extraction.AI_DETECTION.MODEL
        self.processor = None
        self.model = None
        self.cuda_manager = cuda_manager
        self._initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label = None
        
        # Configurations depuis settings
        self.threshold = settings.table_extraction.AI_DETECTION.CONFIDENCE_THRESHOLD
        self.max_tables = settings.table_extraction.AI_DETECTION.MAX_TABLES
        
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
            
            # Supprimer les avertissements pour le chargement
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                
                # Chargement du processeur d'image
                self.processor = DetrImageProcessor.from_pretrained(self.model_name)
                
                # Détermine le dtype en fonction du device
                # Important: utiliser float32 sur CPU et float16 sur CUDA pour éviter les erreurs de type
                dtype = torch.float16 if self.device.type == "cuda" else torch.float32
                
                # Configuration de chargement adaptée
                load_params = {
                    "device_map": "auto" if self.device.type == "cuda" else "cpu",
                    "torch_dtype": dtype,
                    "local_files_only": False,
                    "cache_dir": str(self.models_dir),
                    "low_cpu_mem_usage": True
                }
                
                # Chargement du modèle 
                self.model = DetrForObjectDetection.from_pretrained(
                    self.model_name,
                    **load_params
                )
                
                # Mise en mode évaluation et désactivation du calcul de gradient
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                
                # S'assurer que tout le modèle est au bon dtype
                self.model = self.model.to(dtype)
                
                # Récupération du mapping d'étiquettes
                self.id2label = self.model.config.id2label
                
                logger.info(f"Modèle de détection de tableaux chargé sur {self.device} avec dtype {dtype}")
                self._initialized = True
                
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du modèle de détection: {e}")
            logger.warning("La détection de tableaux IA sera désactivée")
            self._initialized = False
    
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
        # Si non initialisé ou erreur d'initialisation, utiliser la détection par défaut
        if not self._initialized:
            try:
                return await self._detect_tables_simple(image)
            except Exception as e:
                logger.error(f"Erreur lors de la détection simple des tableaux: {e}")
                return []
        
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
                # Ne pas utiliser size ou max_size comme paramètres
                inputs = self.processor(
                    images=pil_image, 
                    return_tensors="pt"
                )
                
                # Détermine le dtype en fonction du device
                # Important: utiliser float32 sur CPU et float16 sur CUDA pour éviter les erreurs de type
                dtype = torch.float16 if self.device.type == "cuda" else torch.float32
                
                # S'assurer que les tenseurs sont au bon dtype
                inputs = {k: v.to(self.device, dtype=dtype) for k, v in inputs.items()}
                
                # S'assurer que le modèle est sur le même device et dtype
                self.model = self.model.to(self.device, dtype=dtype)
                
                # Inférence avec torch.no_grad() pour économiser de la mémoire
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Post-traitement des résultats
                threshold = threshold or self.threshold
                target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
                
                # Traitement des résultats en fonction du modèle
                try:
                    results = self.processor.post_process_object_detection(
                        outputs, 
                        threshold=threshold, 
                        target_sizes=target_sizes
                    )[0]
                except Exception as e:
                    logger.error(f"Erreur post-traitement de détection: {e}")
                    return await self._detect_tables_simple(image)
                
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
                    if 'table' in label_name.lower():
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
            # Fallback sur la détection simple
            try:
                return await self._detect_tables_simple(image)
            except Exception as e2:
                logger.error(f"Erreur lors de la détection simple des tableaux: {e2}")
                return []
    
    async def _detect_tables_simple(self, image: Union[np.ndarray, str, Path]) -> List[Dict[str, Any]]:
        """
        Détecte les tableaux dans une image en utilisant des techniques d'OpenCV classiques.
        Cette méthode est utilisée en fallback si le modèle IA échoue.
        
        Args:
            image: Image à analyser
            
        Returns:
            Liste des tableaux détectés
        """
        try:
            # Convertir l'image au format numpy array
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))
            elif isinstance(image, Image.Image):
                image = np.array(image)
            
            # Convertir en niveaux de gris
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Appliquer un filtre pour détecter les lignes horizontales et verticales
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            # Binarisation avec Otsu
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Détecter les lignes horizontales
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # Détecter les lignes verticales
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combiner les lignes
            table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
            
            # Dilater pour connecter les lignes proches
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            table_mask = cv2.dilate(table_mask, kernel, iterations=4)
            
            # Trouver les contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrer les petits contours
            min_area = 0.005 * image.shape[0] * image.shape[1]  # Au moins 0.5% de l'image
            detections = []
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > min_area:
                    # Vérifier les proportions
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.2 < aspect_ratio < 5:  # Pas trop allongé
                        detections.append({
                            "x": int(x),
                            "y": int(y),
                            "width": int(w),
                            "height": int(h),
                            "label": "table",
                            "score": 0.8  # Score fixe pour les détections simples
                        })
            
            # Si aucune table détectée, considérer toute l'image comme un tableau potentiel
            if not detections:
                detections.append({
                    "x": 0,
                    "y": 0,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "label": "table",
                    "score": 0.6  # Score plus faible
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"Erreur lors de la détection simple: {e}")
            # En cas d'échec complet, retourner toute l'image comme un tableau
            if isinstance(image, np.ndarray):
                height, width = image.shape[:2]
                return [{
                    "x": 0,
                    "y": 0,
                    "width": width,
                    "height": height,
                    "label": "table",
                    "score": 0.5
                }]
            return []
    
    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            if self.model is not None:
                # Procédure de nettoyage explicite pour libérer la mémoire
                if torch.cuda.is_available():
                    self.model = self.model.to("cpu")
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