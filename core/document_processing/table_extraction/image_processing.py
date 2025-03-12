# core/document_processing/table_extraction/image_processing.py
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance
import os
import tempfile
import base64
import io
import math

from core.utils.logger import get_logger

logger = get_logger("table_image_processing")

class PDFImageProcessor:
    """
    Processeur d'images spécialisé pour les tableaux extraits de PDFs.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialise le processeur d'images.
        
        Args:
            temp_dir: Répertoire temporaire pour les images
        """
        self.temp_dir = temp_dir or os.path.join("temp", "table_images")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.max_image_size = (1200, 1200)  # Taille max pour les traitements
    
    def preprocess_image(
        self, 
        image: Union[np.ndarray, Image.Image],
        deskew: bool = True,
        enhance_contrast: bool = True,
        denoise: bool = True,
        binarize: bool = True
    ) -> np.ndarray:
        """
        Prétraite une image pour améliorer la détection et l'OCR des tableaux.
        
        Args:
            image: Image à prétraiter
            deskew: Si True, redresse l'image
            enhance_contrast: Si True, améliore le contraste
            denoise: Si True, réduit le bruit
            binarize: Si True, binarise l'image
            
        Returns:
            Image prétraitée
        """
        try:
            # Conversion en PIL Image si nécessaire
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Redimensionnement si nécessaire
            if pil_image.width > self.max_image_size[0] or pil_image.height > self.max_image_size[1]:
                pil_image.thumbnail(self.max_image_size, Image.LANCZOS)
            
            # Redressement si demandé
            if deskew:
                pil_image = self.deskew_image(pil_image)
            
            # Amélioration du contraste
            if enhance_contrast:
                # Contraste
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.5)
                
                # Netteté
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.5)
            
            # Conversion en numpy array pour OpenCV
            image_cv = np.array(pil_image)
            
            # Conversion en niveaux de gris
            if len(image_cv.shape) > 2 and image_cv.shape[2] == 3:
                gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_cv
            
            # Débruitage
            if denoise:
                gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # Binarisation
            if binarize:
                # Binarisation adaptative (meilleure pour les textes)
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                return binary
            
            return gray
            
        except Exception as e:
            logger.error(f"Erreur prétraitement image: {e}")
            # En cas d'erreur, retourner l'image d'origine
            if isinstance(image, np.ndarray):
                return image
            else:
                return np.array(image)
    
    def deskew_image(self, image: Image.Image) -> Image.Image:
        """
        Redresse une image inclinée.
        
        Args:
            image: Image PIL à redresser
            
        Returns:
            Image redressée
        """
        try:
            # Conversion en OpenCV
            image_np = np.array(image)
            
            # Conversion en niveaux de gris
            if len(image_np.shape) > 2 and image_np.shape[2] == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
            
            # Détecter les bords
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Détecter les lignes
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10
            )
            
            if lines is None:
                return image
            
            # Calculer l'angle moyen
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                # Ne considérer que les angles proches de l'horizontale ou verticale
                if abs(angle) < 45 or abs(angle - 90) < 45 or abs(angle + 90) < 45:
                    angles.append(angle)
            
            if not angles:
                return image
            
            # Angle médian (plus robuste que la moyenne)
            median_angle = np.median(angles)
            
            # Calculer l'angle de correction
            if abs(median_angle) < 45:  # Proche de l'horizontale
                skew_angle = median_angle
            elif abs(median_angle - 90) < 45:  # Proche de la verticale (vers le haut)
                skew_angle = median_angle - 90
            elif abs(median_angle + 90) < 45:  # Proche de la verticale (vers le bas)
                skew_angle = median_angle + 90
            else:
                skew_angle = 0
            
            # Ne corriger que si l'angle est significatif mais pas trop grand
            if 0.5 < abs(skew_angle) < 30:
                # Rotation de l'image avec PIL
                return image.rotate(skew_angle, resample=Image.BICUBIC, expand=True)
            
            return image
            
        except Exception as e:
            logger.error(f"Erreur redressement image: {e}")
            return image
    
    def enhance_table_image(
        self, 
        image: np.ndarray,
        high_contrast: bool = True,
        sharpen: bool = True,
        adaptive_contrast: bool = True
    ) -> np.ndarray:
        """
        Améliore spécifiquement une image de tableau pour OCR.
        
        Args:
            image: Image du tableau
            high_contrast: Si True, applique un contraste élevé
            sharpen: Si True, améliore la netteté
            adaptive_contrast: Si True, applique une amélioration adaptative du contraste
            
        Returns:
            Image améliorée
        """
        try:
            # Conversion en PIL Image
            pil_image = Image.fromarray(image)
            
            if high_contrast:
                # Augmenter le contraste significativement
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(2.0)
            
            if sharpen:
                # Augmenter la netteté
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(2.0)
            
            # Reconversion en numpy array
            enhanced = np.array(pil_image)
            
            # Si l'image est en couleur, conversion en niveaux de gris
            if len(enhanced.shape) > 2 and enhanced.shape[2] == 3:
                gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            else:
                gray = enhanced
            
            # Appliquer une amélioration de contraste adaptative si demandé
            if adaptive_contrast:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                return enhanced
            
            return gray
            
        except Exception as e:
            logger.error(f"Erreur amélioration image tableau: {e}")
            return image
    
    def crop_table_region(
        self, 
        image: np.ndarray, 
        x: int, 
        y: int, 
        width: int, 
        height: int,
        padding: int = 5
    ) -> np.ndarray:
        """
        Découpe une région de tableau dans une image.
        
        Args:
            image: Image source
            x, y, width, height: Coordonnées de la région
            padding: Marge supplémentaire à ajouter
            
        Returns:
            Image découpée
        """
        try:
            # Ajouter le padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            width = min(image.shape[1] - x, width + 2*padding)
            height = min(image.shape[0] - y, height + 2*padding)
            
            # Découper la région
            return image[y:y+height, x:x+width]
            
        except Exception as e:
            logger.error(f"Erreur découpage région tableau: {e}")
            return image
    
    def detect_grid_lines(
        self, 
        image: np.ndarray,
        min_line_length: int = 20,
        threshold_ratio: float = 0.3
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Détecte les lignes de grille horizontales et verticales.
        
        Args:
            image: Image binarisée du tableau
            min_line_length: Longueur minimale des lignes
            threshold_ratio: Seuil de détection par rapport aux dimensions
            
        Returns:
            Tuple (lignes horizontales, lignes verticales)
        """
        try:
            # S'assurer que l'image est en niveaux de gris
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Binarisation
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Calcul des seuils de détection
            height, width = binary.shape
            h_threshold = int(width * threshold_ratio)
            v_threshold = int(height * threshold_ratio)
            
            # Détection des lignes horizontales
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_threshold, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # Détection des lignes verticales
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_threshold))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Amélioration des lignes
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, 1))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_length))
            
            horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_DILATE, h_kernel)
            vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_DILATE, v_kernel)
            
            # Trouver les composants connectés (lignes)
            h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            v_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            return h_contours, v_contours
            
        except Exception as e:
            logger.error(f"Erreur détection lignes grille: {e}")
            return [], []
    
    def encode_image_base64(
        self, 
        image: Union[np.ndarray, Image.Image],
        format: str = "PNG",
        quality: int = 85
    ) -> str:
        """
        Encode une image en base64.
        
        Args:
            image: Image à encoder
            format: Format d'image (PNG, JPEG)
            quality: Qualité pour JPEG
            
        Returns:
            Chaîne encodée en base64
        """
        try:
            # Conversion en PIL Image si nécessaire
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Conversion en RGB si nécessaire
            if pil_image.mode in ('RGBA', 'LA', 'P'):
                pil_image = pil_image.convert('RGB')
            
            # Encodage en mémoire
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format.upper(), quality=quality)
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return base64_str
            
        except Exception as e:
            logger.error(f"Erreur encodage image: {e}")
            return ""
    
    def cleanup(self):
        """Nettoie les fichiers temporaires."""
        try:
            # Vérifier si le répertoire existe avant d'essayer de le nettoyer
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                logger.debug(f"Nettoyage des fichiers temporaires réussi dans {self.temp_dir}")
            else:
                logger.debug(f"Répertoire temporaire {self.temp_dir} non trouvé, aucun nettoyage nécessaire")
        except Exception as e:
            logger.error(f"Erreur nettoyage fichiers temporaires: {e}")