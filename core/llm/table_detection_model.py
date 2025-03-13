# Mise à jour de core/llm/table_detection_model.py

from typing import List, Dict, Any, Union, Optional, Tuple
import torch
import os
import asyncio
from pathlib import Path
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForObjectDetection, DetrForObjectDetection
from PIL import Image
import logging
import warnings

from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings
from core.llm.cuda_manager import ModelPriority

# Réduire le niveau de log pour masquer les avertissements non critiques
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = get_logger("table_detection")

class TableDetectionModel:
    """Détecteur de tableaux utilisant des modèles de pointe pour l'extraction de tables."""
    
    # Modèles disponibles avec leurs caractéristiques
    AVAILABLE_MODELS = {
        "microsoft/table-transformer-detection": {
            "description": "Microsoft DETR model for table detection",
            "size_mb": 183,
            "precision": "high",
            "speed": "medium",
            "preferred_device": "cuda"
        },
        "microsoft/table-transformer-structure-recognition": {
            "description": "Microsoft DETR model for table structure recognition",
            "size_mb": 183,
            "precision": "high", 
            "speed": "medium",
            "preferred_device": "cuda"
        },
        "TahaDouaji/detr-doc-table-detection": {
            "description": "Fine-tuned DETR for document table detection",
            "size_mb": 166,
            "precision": "high",
            "speed": "fast",
            "preferred_device": "cuda"
        },
        "microsoft/dit-base-finetuned-pubtables-structure": {
            "description": "Microsoft Document Image Transformer - better for complex tables",
            "size_mb": 498,
            "precision": "very high",
            "speed": "slow",
            "preferred_device": "cuda"
        },
    }
    
    def __init__(self, cuda_manager=None):
        """
        Initialise le détecteur de tableaux.
        
        Args:
            cuda_manager: Gestionnaire CUDA pour l'allocation mémoire (optionnel)
        """
        # Lire la configuration depuis settings
        ai_detection_config = settings.table_extraction.AI_DETECTION
        
        # Permettre de spécifier un modèle alternatif dans la configuration
        self.model_name = getattr(ai_detection_config, "ALTERNATIVE_MODEL", None) or ai_detection_config.MODEL
        
        # Vérifier si le modèle est dans notre liste de modèles disponibles
        if self.model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Modèle {self.model_name} non reconnu. Utilisation du modèle par défaut.")
            self.model_name = "microsoft/table-transformer-detection"
            
        logger.info(f"Modèle sélectionné: {self.model_name} - {self.AVAILABLE_MODELS[self.model_name]['description']}")
        
        self.processor = None
        self.model = None
        self.cuda_manager = cuda_manager
        self._initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label = None
        
        # Configurations depuis settings
        self.threshold = ai_detection_config.CONFIDENCE_THRESHOLD
        self.max_tables = ai_detection_config.MAX_TABLES
        
        # Dossier pour les modèles
        self.models_dir = Path(settings.MODELS_DIR) / "table_detection"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimisations supplémentaires
        self.cache_enabled = True
        self.cache = {}  # Cache simple pour les résultats de détection
    
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
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                
                # Détermine le dtype en fonction du device
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
                self.model = AutoModelForObjectDetection.from_pretrained(
                    self.model_name,
                    **load_params
                )
                
                # Mise en mode évaluation et désactivation du calcul de gradient
                self.model.eval()
                
                # Optimisation pour l'inférence
                if self.device.type == "cuda":
                    # Optimisations spécifiques à CUDA
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                
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
        image: Union[np.ndarray, str, Path, Image.Image], 
        threshold: Optional[float] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Détecte les tableaux dans une image.
        
        Args:
            image: Image au format numpy array, PIL Image, chemin ou Path
            threshold: Seuil de confiance pour la détection (0-1)
            use_cache: Utiliser le cache pour les résultats de détection
            
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
                # Vérifier le cache si activé
                if use_cache and self.cache_enabled:
                    image_hash = self._get_image_hash(image)
                    if image_hash in self.cache:
                        logger.debug(f"Détection trouvée en cache pour {image_hash}")
                        return self.cache[image_hash]
                
                # Convertir l'image au bon format
                pil_image = self._convert_to_pil_image(image)
                
                # Adapter la taille de l'image si nécessaire
                max_size = 1600  # Maximum pour une bonne détection sans trop de mémoire
                if pil_image.width > max_size or pil_image.height > max_size:
                    ratio = max_size / max(pil_image.width, pil_image.height)
                    new_width = int(pil_image.width * ratio)
                    new_height = int(pil_image.height * ratio)
                    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Préparation de l'image pour le modèle
                inputs = self.processor(images=pil_image, return_tensors="pt")
                
                # Détermine le dtype en fonction du device
                dtype = torch.float16 if self.device.type == "cuda" else torch.float32
                
                # S'assurer que les tenseurs sont au bon dtype et device
                inputs = {k: v.to(self.device, dtype=dtype) for k, v in inputs.items()}
                
                # Inférence avec torch.no_grad() pour économiser de la mémoire
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Post-traitement des résultats
                threshold = threshold or self.threshold
                target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
                
                # Traitement des résultats
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
                
                # Post-traitement pour éliminer les détections qui se chevauchent trop
                detections = self._filter_overlapping_detections(detections, iou_threshold=0.5)
                
                # Stocker dans le cache
                if self.cache_enabled and use_cache:
                    image_hash = self._get_image_hash(image)
                    self.cache[image_hash] = detections
                
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
    
    def _convert_to_pil_image(self, image: Union[np.ndarray, str, Path, Image.Image]) -> Image.Image:
        """
        Convertit divers formats d'image en PIL Image.
        
        Args:
            image: Image à convertir
            
        Returns:
            Image PIL
        """
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Conversion OpenCV (BGR) à PIL (RGB)
            if len(image.shape) == 2:  # Image en niveaux de gris
                return Image.fromarray(image).convert("RGB")
            elif image.shape[2] == 3:  # Image BGR
                return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif image.shape[2] == 4:  # Image BGRA
                return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
            else:
                return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise ValueError(f"Format d'image non supporté: {type(image)}")
    
    def _get_image_hash(self, image: Union[np.ndarray, str, Path, Image.Image]) -> str:
        """
        Génère un hash unique pour une image.
        
        Args:
            image: Image à hacher
            
        Returns:
            Hash de l'image
        """
        import hashlib
        
        # Convertir en PIL Image puis en bytes
        if not isinstance(image, Image.Image):
            pil_image = self._convert_to_pil_image(image)
        else:
            pil_image = image
        
        # Redimensionner pour un hash plus rapide
        pil_image = pil_image.resize((100, 100), Image.Resampling.LANCZOS)
        
        # Convertir en bytes et générer un hash
        img_bytes = np.array(pil_image).tobytes()
        return hashlib.md5(img_bytes).hexdigest()
    
    def _filter_overlapping_detections(
        self, 
        detections: List[Dict[str, Any]], 
        iou_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Filtre les détections qui se chevauchent trop en utilisant NMS (Non-Maximum Suppression).
        
        Args:
            detections: Liste des détections
            iou_threshold: Seuil IoU pour la suppression
            
        Returns:
            Liste des détections filtrées
        """
        if not detections:
            return []
        
        # Extraire les boîtes et les scores
        boxes = np.array([[d["x"], d["y"], d["x"] + d["width"], d["y"] + d["height"]] for d in detections])
        scores = np.array([d["score"] for d in detections])
        
        # Appliquer la NMS
        indices = self._non_max_suppression(boxes, scores, iou_threshold)
        
        # Retourner les détections filtrées
        return [detections[i] for i in indices]
    
    def _non_max_suppression(
        self, 
        boxes: np.ndarray, 
        scores: np.ndarray, 
        iou_threshold: float
    ) -> List[int]:
        """
        Implémentation de l'algorithme Non-Maximum Suppression.
        
        Args:
            boxes: Array de boîtes [x1, y1, x2, y2]
            scores: Array de scores de confiance
            iou_threshold: Seuil IoU pour la suppression
            
        Returns:
            Indices des boîtes à conserver
        """
        # Trier par score
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculer l'IoU avec les autres boîtes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            # Calculer les aires
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            
            # IoU = intersection / union
            iou = inter / (area_i + area_others - inter)
            
            # Garder les boîtes avec IoU < threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    async def _detect_tables_simple(self, image: Union[np.ndarray, str, Path, Image.Image]) -> List[Dict[str, Any]]:
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
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Convertir en niveaux de gris
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Méthode améliorée avec multiples techniques
            
            # 1. Détection basée sur les lignes
            # Appliquer un filtre pour détecter les lignes horizontales et verticales
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            # Binarisation adaptative pour une meilleure détection des lignes
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Détecter les lignes horizontales
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)
            
            # Détecter les lignes verticales
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)
            
            # Combiner les lignes
            table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
            
            # 2. Détection basée sur les bordures
            # Opérations morphologiques pour fermer les bordures incomplètes
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            border_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            border_mask = cv2.dilate(border_mask, kernel, iterations=3)
            
            # 3. Détection basée sur le texte
            # Appliquer des méthodes de détection de texte pour améliorer la détection des tableaux sans bordures
            text_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 8)
            text_areas = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            rows_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            text_rows = cv2.morphologyEx(text_areas, cv2.MORPH_OPEN, rows_kernel, iterations=2)
            text_rows = cv2.dilate(text_rows, rows_kernel, iterations=3)
            
            # Combiner toutes les méthodes
            combined_mask = cv2.bitwise_or(border_mask, text_rows)
            
            # Trouver les contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrer les contours
            min_area = 0.005 * image.shape[0] * image.shape[1]  # Au moins 0.5% de l'image
            max_area = 0.95 * image.shape[0] * image.shape[1]   # Pas plus de 95% de l'image
            detections = []
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                if min_area < area < max_area:
                    # Vérifier les proportions
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.1 < aspect_ratio < 10:  # Proportions raisonnables pour un tableau
                        # Score basé sur la densité des lignes et du texte
                        region_mask = combined_mask[y:y+h, x:x+w]
                        content_density = np.sum(region_mask > 0) / (w * h)
                        
                        # Ajuster le score en fonction de la densité
                        score = min(0.95, 0.5 + content_density * 0.5)
                        
                        detections.append({
                            "x": int(x),
                            "y": int(y),
                            "width": int(w),
                            "height": int(h),
                            "label": "table",
                            "score": float(score)
                        })
            
            # Filtrer les détections qui se chevauchent
            if detections:
                detections = self._filter_overlapping_detections(detections, iou_threshold=0.3)
            
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
            
            # Vider le cache
            self.cache.clear()
            
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