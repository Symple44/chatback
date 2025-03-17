from typing import List, Dict, Any, Optional, Union, Tuple, BinaryIO
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import io
import numpy as np
import cv2
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import tempfile
import base64
import re
import time
import uuid
import shutil

from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings

# Importer le processeur d'image pour le prétraitement
from .image_processing import PDFImageProcessor
from .utils import convert_pdf_to_images, parse_page_range

logger = get_logger("checkbox_extractor")

class CheckboxExtractor:
    """
    Extracteur de cases à cocher dans les documents PDF.
    
    Cette classe permet de détecter les cases à cocher dans un document PDF,
    d'analyser leur état (coché ou non coché) et de les associer à du texte
    environnant pour en comprendre le contexte.
    """
    
    def __init__(self, table_detector=None):
        """
        Initialise l'extracteur de cases à cocher.
        
        Args:
            table_detector: Détecteur de tableaux par IA (optionnel)
        """
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        self.image_processor = PDFImageProcessor()
        self.table_detector = table_detector
        
        # Configurations depuis settings
        try:
            if hasattr(settings, 'table_extraction') and hasattr(settings.table_extraction, 'CHECKBOX'):
                checkbox_config = settings.table_extraction.CHECKBOX
                self.min_size = checkbox_config.MIN_SIZE if hasattr(checkbox_config, 'MIN_SIZE') else 10
                self.max_size = checkbox_config.MAX_SIZE if hasattr(checkbox_config, 'MAX_SIZE') else 50
                self.default_confidence = checkbox_config.DEFAULT_CONFIDENCE if hasattr(checkbox_config, 'DEFAULT_CONFIDENCE') else 0.6
            else:
                # Valeurs par défaut
                self.min_size = 10  # Taille minimale en pixels
                self.max_size = 50  # Taille maximale en pixels
                self.default_confidence = 0.6  # Seuil de confiance par défaut
        except Exception as e:
            logger.warning(f"Erreur lors du chargement des configurations: {e}, utilisation des valeurs par défaut")
            self.min_size = 10
            self.max_size = 50
            self.default_confidence = 0.6
        
        # Initialiser un modèle dédié à la détection des cases à cocher si disponible
        self.checkbox_model = None
        self.use_ai_detection = self.table_detector is not None
        
        # Patterns regex pour la détection de texte associé aux cases à cocher
        self.label_patterns = [
            r"([A-Za-z0-9\u00C0-\u017F][A-Za-z0-9\s\-_.,;:()/\u00C0-\u017F]{2,50})\s*[□☐☑✓✔✕✖✗✘]",  # Label suivi d'une case
            r"[□☐☑✓✔✕✖✗✘]\s*([A-Za-z0-9\u00C0-\u017F][A-Za-z0-9\s\-_.,;:()/\u00C0-\u017F]{2,50})",  # Case suivie d'un label
            r"([A-Za-z0-9\u00C0-\u017F][A-Za-z0-9\s\-_.,;:()/\u00C0-\u017F]{2,50})"                  # Tout texte à proximité
        ]
        
        # Configurer pytesseract
        if hasattr(settings, 'table_extraction') and hasattr(settings.table_extraction, 'OCR'):
            tesseract_cmd = settings.table_extraction.OCR.TESSERACT_CMD
            if os.path.exists(tesseract_cmd) and os.access(tesseract_cmd, os.X_OK):
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    async def extract_checkboxes_from_pdf(
        self,
        file_obj: Union[str, bytes, BinaryIO],
        page_range: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extrait les cases à cocher d'un document PDF.
        
        Args:
            file_obj: Fichier PDF (chemin, bytes ou objet fichier)
            page_range: Plage de pages à analyser (None = toutes les pages)
            config: Configuration supplémentaire
                - confidence_threshold: Seuil de confiance (0-1)
                - enhance_detection: Activer la détection améliorée
                - include_images: Inclure les images des cases à cocher
                
        Returns:
            Dictionnaire contenant les cases à cocher détectées et leur état
        """
        try:
            with metrics.timer("checkbox_extraction"):
                # Gestion du fichier d'entrée
                file_path = None
                is_temp = False
                
                if isinstance(file_obj, str):
                    file_path = file_obj
                else:
                    # Créer un fichier temporaire
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                        if isinstance(file_obj, bytes):
                            temp_file.write(file_obj)
                        else:
                            # Supposer que c'est un objet file-like
                            file_obj.seek(0)
                            shutil.copyfileobj(file_obj, temp_file)
                        file_path = temp_file.name
                        is_temp = True
                
                try:
                    # Configuration
                    if config is None:
                        config = {}
                    
                    confidence_threshold = config.get('confidence_threshold', self.default_confidence)
                    enhance_detection = config.get('enhance_detection', True)
                    include_images = config.get('include_images', False)
                    
                    # Analyser les pages du PDF
                    all_checkboxes = []
                    form_values = {}
                    form_sections = {}
                    checkbox_images = [] if include_images else None
                    
                    # Convertir le range de pages si fourni
                    if page_range is None:
                        page_indices = await parse_page_range(file_path, "all")
                    else:
                        page_indices = [p-1 for p in page_range]  # Convertir en 0-indexed
                    
                    # Obtenir les images des pages
                    page_images = await convert_pdf_to_images(file_path, page_indices, dpi=300)
                    
                    # Extraire le texte des pages pour analyse contextuelle
                    doc = fitz.open(file_path)

                    # Traiter chaque page
                    for idx, img in enumerate(page_images):
                        page_idx = page_indices[idx] if idx < len(page_indices) else idx
                        page_num = page_idx + 1  # 1-indexed pour l'affichage
                        
                        # Extraire le texte de la page pour le contexte
                        page_text = ""
                        if page_idx < len(doc):
                            page = doc[page_idx]
                            # Utiliser la méthode get_text() sans options spéciales
                            # Compatible avec PyMuPDF 1.25.3
                            page_text = page.get_text("text")
                            
                            # Extraire les cases à cocher de type formulaire si présentes
                            widgets = page.widgets()
                            if widgets:
                                for widget in widgets:
                                    # Vérifier si c'est une case à cocher (en utilisant une méthode compatible)
                                    try:
                                        field_type = getattr(widget, 'field_type', None)
                                        # Si nous ne pouvons pas accéder à field_type directement
                                        if field_type is None and hasattr(widget, 'field_type_string'):
                                            if widget.field_type_string == 'Button':
                                                field_type = 4  # Code pour PDF_WIDGET_TYPE_BUTTON
                                        
                                        # Si c'est un bouton (case à cocher)
                                        if field_type == 4:  # PDF_WIDGET_TYPE_BUTTON dans les versions plus récentes
                                            checkbox_value = widget.field_value
                                            checkbox_name = widget.field_name
                                            checkbox_label = getattr(widget, 'field_label', "") or ""
                                            checkbox_rect = widget.rect
                                            
                                            # Ajouter à la liste des cases à cocher détectées
                                            checkbox_data = {
                                                "id": str(len(all_checkboxes) + 1),
                                                "page": page_num,
                                                "x": checkbox_rect.x0,
                                                "y": checkbox_rect.y0,
                                                "width": checkbox_rect.width,
                                                "height": checkbox_rect.height,
                                                "label": checkbox_label,
                                                "field_name": checkbox_name,
                                                "is_checked": bool(checkbox_value),
                                                "confidence": 1.0,  # Confiance maximale pour les cases de formulaire
                                                "type": "form_widget",
                                                "text_context": ""
                                            }
                                            
                                            all_checkboxes.append(checkbox_data)
                                            
                                            # Ajouter aux valeurs de formulaire
                                            form_values[checkbox_name] = bool(checkbox_value)
                                    except Exception as widget_error:
                                        logger.debug(f"Erreur traitement widget: {widget_error}")
                        
                        # Post-validation pour éliminer les faux positifs
                        checkboxes = await self._post_validate_checkboxes(checkboxes)
                        
                        # Si très peu de cases à cocher validées, tenter une approche plus permissive
                        if len(checkboxes) < 2 and enhance_detection:
                            logger.info(f"Peu de cases à cocher détectées, essai d'une approche plus permissive")
                            # Approche plus permissive avec seuil de confiance plus bas
                            permissive_checkboxes = await self._detect_checkboxes_in_image(
                                img, 
                                page_num=page_num,
                                confidence_threshold=confidence_threshold * 0.8,
                                enhance_detection=True
                            )
                            # Post-valider même avec l'approche permissive
                            permissive_checkboxes = await self._post_validate_checkboxes(permissive_checkboxes)
                            checkboxes = permissive_checkboxes
                        
                        # Pour les pages sans case à cocher détectée, ne pas perdre de temps avec le contexte
                        if not checkboxes:
                            continue
                        
                        # Optimisation pour limiter les cases à cocher similaires sur la même page
                        if len(checkboxes) > 15:  # Si trop de cases détectées sur une page, filtrer davantage
                            logger.info(f"Trop de cases à cocher détectées sur la page {page_num} ({len(checkboxes)}), filtrage supplémentaire")
                            checkboxes = self._filter_similar_checkboxes(checkboxes)
                        
                        # Analyser et associer le contexte textuel pour chaque case à cocher
                        for checkbox in checkboxes:
                            # Extraire le contexte textuel autour de la case à cocher
                            context = await self._extract_checkbox_context(
                                img, 
                                checkbox, 
                                page_text
                            )
                            
                            # Vérifier si le contexte est significatif, sinon l'ignorer
                            if context and not self._is_context_meaningful(context):
                                context = ""
                            
                            checkbox["text_context"] = context
                            checkbox["id"] = str(len(all_checkboxes) + 1)
                            
                            # Identifier le type de section (formulaire, contrat, etc.)
                            section = await self._identify_form_section(context)
                            if section:
                                if section not in form_sections:
                                    form_sections[section] = []
                                form_sections[section].append(checkbox["id"])
                            
                            # Identifier le nom de champ potentiel
                            field_name = await self._extract_field_name(context)
                            if field_name and len(field_name) >= 3:
                                checkbox["field_name"] = field_name
                                # Ajouter aux valeurs de formulaire structurées
                                form_values[field_name] = checkbox["is_checked"]
                            
                            # Ajouter aux résultats
                            all_checkboxes.append(checkbox)
                    
                    # Fermer le document
                    doc.close()
                    
                    # Organiser les résultats
                    result = {
                        "extraction_id": str(uuid.uuid4()),
                        "checkbox_count": len(all_checkboxes),
                        "checkboxes": all_checkboxes,
                        "form_values": form_values,
                        "form_sections": form_sections
                    }
                    
                    if include_images:
                        result["checkbox_images"] = checkbox_images
                    
                    logger.info(f"Extraction de {len(all_checkboxes)} cases à cocher terminée")
                    return result
                    
                finally:
                    # Nettoyage du fichier temporaire
                    if is_temp and file_path and os.path.exists(file_path):
                        os.unlink(file_path)
                        
        except Exception as e:
            logger.error(f"Erreur extraction cases à cocher: {e}")
            return {
                "extraction_id": str(uuid.uuid4()),
                "checkbox_count": 0,
                "checkboxes": [],
                "form_values": {},
                "form_sections": {},
                "error": str(e)
            }
    
    def _filter_similar_checkboxes(self, checkboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filtre les cases à cocher similaires pour réduire les redondances.
        
        Args:
            checkboxes: Liste de cases à cocher détectées
            
        Returns:
            Liste filtrée sans doublons similaires
        """
        if len(checkboxes) <= 5:  # Pas besoin de filtrer s'il y a peu de cases
            return checkboxes
            
        # Trier par confiance décroissante
        sorted_checkboxes = sorted(checkboxes, key=lambda x: x.get("confidence", 0), reverse=True)
        
        filtered = []
        # Utiliser les positions pour détecter les doublons
        used_positions = set()
        
        for checkbox in sorted_checkboxes:
            x, y = checkbox.get("x", 0), checkbox.get("y", 0)
            
            # Vérifier si on a déjà une case similaire à proximité
            is_duplicate = False
            for used_x, used_y in used_positions:
                # Si les cases sont trop proches l'une de l'autre
                if abs(x - used_x) < 20 and abs(y - used_y) < 20:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                used_positions.add((x, y))
                filtered.append(checkbox)
                
                # Limiter le nombre total de cases à cocher par page
                if len(filtered) >= 10:  # Maximum 10 cases par page
                    break
        
        return filtered

    def _is_context_meaningful(self, context: str) -> bool:
        """
        Vérifie si un contexte textuel est significatif.
        
        Args:
            context: Texte de contexte
            
        Returns:
            True si le contexte est significatif
        """
        if not context:
            return False
            
        # Supprimer les caractères spéciaux et espaces
        clean_text = re.sub(r'[^\w]', '', context)
        
        # Vérifier la longueur utile
        if len(clean_text) < 3:
            return False
            
        # Vérifier s'il y a au moins un mot significatif
        words = re.findall(r'\b[a-zA-Z]{3,}\b', context)
        if not words:
            return False
            
        # Vérifier le ratio de caractères spéciaux vs alphabétiques
        alpha_count = sum(1 for c in context if c.isalpha())
        special_count = sum(1 for c in context if not c.isalnum() and not c.isspace())
        
        if alpha_count == 0 or special_count / (alpha_count + 1) > 0.7:
            return False
            
        return True

    async def _detect_checkboxes_in_image(
        self,
        image: np.ndarray,
        page_num: int,
        confidence_threshold: float = 0.6,
        enhance_detection: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Détecte les cases à cocher dans une image.
        
        Args:
            image: Image à analyser (numpy array)
            page_num: Numéro de page
            confidence_threshold: Seuil de confiance
            enhance_detection: Utiliser la détection améliorée
            
        Returns:
            Liste des cases à cocher détectées
        """
        try:
            # Si l'IA est disponible et activée, utiliser le modèle IA
            if self.use_ai_detection and self.table_detector:
                try:
                    # Tenter la détection avec IA
                    checkboxes = await self._detect_checkboxes_with_ai(image, confidence_threshold)
                    
                    if checkboxes:
                        for checkbox in checkboxes:
                            checkbox["page"] = page_num
                        return checkboxes
                except Exception as e:
                    logger.warning(f"Erreur détection IA des cases à cocher: {e}")
            
            # Utiliser les méthodes classiques de vision par ordinateur
            checkboxes = []
            
            # Traitement de l'image
            if enhance_detection:
                # Prétraitement avancé pour améliorer la détection
                processed_img = await self._preprocess_image_for_checkbox(image)
            else:
                # Prétraitement basique
                if len(image.shape) > 2:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                processed_img = gray
            
            # 1. Détection basée sur les contours pour les cases à cocher rectangulaires
            rectangular_checkboxes = await self._detect_rectangular_checkboxes(processed_img, confidence_threshold)
            checkboxes.extend(rectangular_checkboxes)
            
            # 2. Détection basée sur les cercles pour les cases à cocher circulaires (radio buttons)
            circular_checkboxes = await self._detect_circular_checkboxes(processed_img, confidence_threshold)
            checkboxes.extend(circular_checkboxes)
            
            # 3. Détection basée sur les symboles pour les cases à cocher par symboles (□, ☐, ☑, ✓)
            if enhance_detection:
                symbol_checkboxes = await self._detect_checkbox_symbols(image, processed_img, confidence_threshold)
                checkboxes.extend(symbol_checkboxes)
            
            # Regrouper les détections qui se chevauchent
            merged_checkboxes = await self._merge_overlapping_checkboxes(checkboxes)
            
            # Déterminer l'état de chaque case à cocher (cochée ou non)
            for checkbox in merged_checkboxes:
                # Extraire la région de la case à cocher
                x, y, w, h = checkbox["x"], checkbox["y"], checkbox["width"], checkbox["height"]
                
                # Vérifier les limites de l'image
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(image.shape[1], x + w)
                y2 = min(image.shape[0], y + h)
                
                # Extraire la région
                checkbox_region = image[y1:y2, x1:x2]
                
                # Déterminer si la case est cochée
                is_checked, check_confidence = await self._is_checkbox_checked(checkbox_region, checkbox["type"])
                
                checkbox["is_checked"] = is_checked
                checkbox["check_confidence"] = check_confidence
                checkbox["page"] = page_num
            
            return merged_checkboxes
            
        except Exception as e:
            logger.error(f"Erreur détection cases à cocher: {e}")
            return []
    
    async def _preprocess_image_for_checkbox(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraite une image pour améliorer la détection des cases à cocher.
        
        Args:
            image: Image à prétraiter
            
        Returns:
            Image prétraitée
        """
        try:
            # Conversion en niveaux de gris
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Réduction du bruit
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # Amélioration du contraste - CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Binarisation adaptative
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Opérations morphologiques pour réduire le bruit et renforcer les contours
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Erreur prétraitement image: {e}")
            return image
    
    async def _detect_rectangular_checkboxes(
        self,
        image: np.ndarray,
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Détecte les cases à cocher rectangulaires avec filtrage amélioré.
        
        Args:
            image: Image prétraitée
            confidence_threshold: Seuil de confiance
            
        Returns:
            Liste des cases à cocher rectangulaires détectées
        """
        try:
            checkboxes = []
            
            # Trouver les contours
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approcher le contour par un polygone
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Vérifier si c'est un rectangle (4 côtés) ou proche
                # Filtrage plus strict: uniquement 4 côtés
                if len(approx) == 4:
                    # Obtenir le rectangle englobant
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filtrage par taille - plage plus étroite
                    if self.min_size <= w <= self.max_size and self.min_size <= h <= self.max_size:
                        # Vérifier si le rapport largeur/hauteur est proche de 1 (carré) - plus strict
                        aspect_ratio = w / h
                        if 0.8 <= aspect_ratio <= 1.2:
                            # Vérifier l'aire du contour comparée à l'aire du rectangle
                            rect_area = w * h
                            contour_area = cv2.contourArea(contour)
                            area_ratio = contour_area / rect_area if rect_area > 0 else 0
                            
                            # Filtrage plus strict - le contour doit correspondre au rectangle
                            if area_ratio > 0.7:
                                # Une case à cocher idéale a un ratio proche de 1
                                confidence = area_ratio * (1 - abs(1 - aspect_ratio))
                                
                                # Vérification supplémentaire: au moins 3 points distants
                                if self._has_distant_corners(approx):
                                    if confidence >= confidence_threshold:
                                        checkboxes.append({
                                            "x": int(x),
                                            "y": int(y),
                                            "width": int(w),
                                            "height": int(h),
                                            "confidence": float(confidence),
                                            "type": "rectangular"
                                        })
            
            return checkboxes
            
        except Exception as e:
            logger.error(f"Erreur détection cases rectangulaires: {e}")
            return []
        
    def _has_distant_corners(self, approx) -> bool:
        """Vérifie si les points du contour forment un vrai rectangle."""
        if len(approx) != 4:
            return False
            
        # Extraire les points
        points = [p[0] for p in approx]
        
        # Calculer les distances entre tous les points
        min_dist = float('inf')
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                min_dist = min(min_dist, dist)
    
        # Pour être un vrai rectangle, les points doivent être suffisamment distants
        return min_dist >= self.min_size * 0.8

    async def _detect_circular_checkboxes(
        self,
        image: np.ndarray,
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Détecte les cases à cocher circulaires (radio buttons).
        
        Args:
            image: Image prétraitée
            confidence_threshold: Seuil de confiance
            
        Returns:
            Liste des cases à cocher circulaires détectées
        """
        try:
            checkboxes = []
            
            # Détecter les cercles avec Hough Circle Transform
            circles = cv2.HoughCircles(
                image, 
                cv2.HOUGH_GRADIENT, 
                dp=1.2, 
                minDist=20, 
                param1=50, 
                param2=30, 
                minRadius=int(self.min_size / 2), 
                maxRadius=int(self.max_size / 2)
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles[0, :]))
                
                for circle in circles:
                    # Extraire les coordonnées et le rayon
                    center_x, center_y, radius = circle
                    
                    # Calculer le rectangle englobant
                    x = int(center_x - radius)
                    y = int(center_y - radius)
                    w = int(radius * 2)
                    h = int(radius * 2)
                    
                    # Vérifier les limites de l'image
                    if x >= 0 and y >= 0 and x + w < image.shape[1] and y + h < image.shape[0]:
                        # Calculer la confiance basée sur la netteté du cercle
                        # Extraire la région du cercle
                        circle_region = image[y:y+h, x:x+w]
                        
                        # Créer un masque circulaire
                        mask = np.zeros_like(circle_region)
                        cv2.circle(mask, (radius, radius), radius, 255, -1)
                        
                        # Appliquer le masque
                        masked_circle = cv2.bitwise_and(circle_region, mask)
                        
                        # Calcul de la confiance basée sur la proportion de pixels de contour
                        # dans la zone circulaire par rapport à l'aire totale
                        total_pixels = np.pi * radius * radius
                        contour_pixels = np.sum(masked_circle > 0)
                        
                        # Rapport pixels de contour / aire théorique
                        if total_pixels > 0:
                            edge_ratio = contour_pixels / total_pixels
                            
                            # Un bon cercle a un ratio entre 0.2 et 0.5 (contour net, intérieur vide)
                            if 0.1 <= edge_ratio <= 0.7:
                                # Normaliser la confiance
                                confidence = 1.0 - abs(0.3 - edge_ratio) * 2
                                confidence = max(0.0, min(1.0, confidence))
                                
                                if confidence >= confidence_threshold:
                                    checkboxes.append({
                                        "x": int(x),
                                        "y": int(y),
                                        "width": int(w),
                                        "height": int(h),
                                        "confidence": float(confidence),
                                        "type": "circular"
                                    })
            
            return checkboxes
            
        except Exception as e:
            logger.error(f"Erreur détection cases circulaires: {e}")
            return []
    
    async def _detect_checkbox_symbols(
        self,
        original_image: np.ndarray,
        processed_image: np.ndarray,
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Détecte les symboles de cases à cocher (□, ☐, ☑, ✓, etc.).
        
        Args:
            original_image: Image originale
            processed_image: Image prétraitée
            confidence_threshold: Seuil de confiance
            
        Returns:
            Liste des cases à cocher symboliques détectées
        """
        try:
            checkboxes = []
            
            # Trouver tous les contours
            contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Les cases à cocher ont souvent une hiérarchie parent-enfant (le contour externe et interne)
            if hierarchy is not None:
                hierarchy = hierarchy[0]
                
                for i, contour in enumerate(contours):
                    # Ignorer les contours trop petits
                    if cv2.contourArea(contour) < self.min_size * self.min_size * 0.5:
                        continue
                    
                    # Approcher le contour
                    epsilon = 0.04 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Vérifier si c'est un rectangle (4 côtés)
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Filtrer par taille
                        if self.min_size <= w <= self.max_size and self.min_size <= h <= self.max_size:
                            # Vérifier le rapport largeur/hauteur
                            aspect_ratio = w / h
                            if 0.7 <= aspect_ratio <= 1.3:
                                # Vérifier s'il y a des contours enfants (intérieurs)
                                child_idx = hierarchy[i][2]
                                
                                # Calcul de la confiance
                                confidence = 0.6  # Confiance de base
                                
                                # Si contour enfant trouvé, il peut s'agir d'une case cochée
                                if child_idx > -1:
                                    child_contour = contours[child_idx]
                                    child_area = cv2.contourArea(child_contour)
                                    parent_area = cv2.contourArea(contour)
                                    
                                    # Ratio de surface enfant/parent
                                    area_ratio = child_area / parent_area if parent_area > 0 else 0
                                    
                                    # Si le contour intérieur occupe entre 10% et 80% de la case
                                    if 0.1 <= area_ratio <= 0.8:
                                        confidence += 0.2
                                
                                # Vérifier les sous-contours pour détecter les marques de cochage
                                is_checked = False
                                if child_idx > -1:
                                    is_checked = True
                                
                                if confidence >= confidence_threshold:
                                    checkboxes.append({
                                        "x": int(x),
                                        "y": int(y),
                                        "width": int(w),
                                        "height": int(h),
                                        "confidence": float(confidence),
                                        "type": "symbol",
                                        "is_checked": is_checked
                                    })
            
            return checkboxes
            
        except Exception as e:
            logger.error(f"Erreur détection symboles cases à cocher: {e}")
            return []
    
    async def _merge_overlapping_checkboxes(
        self,
        checkboxes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Fusionne les détections qui se chevauchent.
        
        Args:
            checkboxes: Liste des cases à cocher détectées
            
        Returns:
            Liste filtrée sans chevauchements
        """
        if not checkboxes:
            return []
            
        # Convertir en format de boîtes pour NMS: [x1, y1, x2, y2, confidence]
        boxes = []
        for cb in checkboxes:
            boxes.append([
                cb["x"], 
                cb["y"], 
                cb["x"] + cb["width"], 
                cb["y"] + cb["height"], 
                cb["confidence"]
            ])
        
        boxes = np.array(boxes)
        
        # Appliquer la Non-Maximum Suppression
        indices = await self._non_max_suppression(boxes[:, :4], boxes[:, 4], iou_threshold=0.5)
        
        # Sélectionner les boîtes et ajouter les métadonnées
        result = []
        for i in indices:
            result.append(checkboxes[i])
        
        return result
    
    async def _non_max_suppression(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float = 0.5
    ) -> List[int]:
        """
        Applique l'algorithme Non-Maximum Suppression pour filtrer les détections.
        
        Args:
            boxes: Coordonnées des boîtes [x1, y1, x2, y2]
            scores: Scores de confiance
            iou_threshold: Seuil de chevauchement
            
        Returns:
            Indices des boîtes à conserver
        """
        # Trier par score
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calcul de l'IoU avec les boîtes restantes
            if order.size == 1:
                break
                
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            # Calculer l'aire de chaque boîte
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_candidates = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            
            # Union = somme des aires - intersection
            union = area_i + area_candidates - intersection
            
            # IoU
            iou = intersection / union
            
            # Garder les boîtes avec IoU inférieur au seuil
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    async def _is_checkbox_checked(
        self,
        checkbox_region: np.ndarray,
        checkbox_type: str
    ) -> Tuple[bool, float]:
        """
        Détermine si une case à cocher est cochée avec une approche améliorée.
        
        Args:
            checkbox_region: Image de la case à cocher
            checkbox_type: Type de case à cocher
            
        Returns:
            Tuple (est_cochée, confiance)
        """
        try:
            if checkbox_region.size == 0:
                return False, 0.0
                
            # Convertir en niveaux de gris si ce n'est pas déjà le cas
            if len(checkbox_region.shape) > 2:
                gray = cv2.cvtColor(checkbox_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = checkbox_region
            
            # Appliquer un débruitage
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Binarisation avec seuil adaptatif pour une meilleure sensibilité aux marques
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
            
            # Calculer le pourcentage de pixels noirs (cochés) dans la région
            total_pixels = binary.size
            if total_pixels == 0:
                return False, 0.0
                
            # Trouver les contours dans la région binaire
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extraire uniquement les contours significatifs (pas de bruit)
            significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > total_pixels * 0.01]
            
            # Filtrer les contours basés sur la position (ignorer les contours de bordure)
            center_contours = []
            h, w = binary.shape
            for cnt in significant_contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                # Vérifier si le contour est principalement au centre (pas sur les bords)
                if x > w*0.1 and y > h*0.1 and x+cw < w*0.9 and y+ch < h*0.9:
                    center_contours.append(cnt)
            
            # Calcul du taux de remplissage intérieur
            filled_pixels = sum(cv2.contourArea(cnt) for cnt in center_contours)
            fill_ratio = filled_pixels / total_pixels if total_pixels > 0 else 0
            
            # Vérifier la disposition des pixels pour les marques caractéristiques (X, ✓)
            has_mark_pattern = self._detect_mark_pattern(binary)
            
            # Déterminer l'état selon le type de case
            is_checked = False
            confidence = 0.0
            
            if checkbox_type == "rectangular":
                # Pour les cases rectangulaires: combine le ratio de remplissage et la détection de marque
                if fill_ratio > 0.2 or has_mark_pattern:
                    is_checked = True
                    confidence = max(0.7, fill_ratio) if has_mark_pattern else min(1.0, fill_ratio * 1.5)
                else:
                    is_checked = False
                    confidence = max(0.7, 1.0 - fill_ratio * 2.0)
            elif checkbox_type == "circular":
                # Pour les boutons radio: le remplissage central est déterminant
                if fill_ratio > 0.25:
                    is_checked = True
                    confidence = min(1.0, fill_ratio * 1.2)
                else:
                    is_checked = False
                    confidence = min(1.0, (1.0 - fill_ratio * 2.0))
            elif checkbox_type == "symbol":
                # Pour les symboles: combiner plusieurs facteurs
                if fill_ratio > 0.25 or has_mark_pattern or len(center_contours) > 0:
                    is_checked = True
                    confidence = 0.7 if has_mark_pattern else min(1.0, fill_ratio * 1.3)
                else:
                    is_checked = False
                    confidence = min(1.0, (1.0 - fill_ratio * 2.5))
            
            return is_checked, confidence
            
        except Exception as e:
            logger.error(f"Erreur vérification état case à cocher: {e}")
            return False, 0.0
    
    def _detect_mark_pattern(self, binary_img: np.ndarray) -> bool:
        """
        Détecte les motifs caractéristiques des marques de case cochée (X, ✓).
        
        Args:
            binary_img: Image binaire de la case à cocher
            
        Returns:
            True si un motif de marque est détecté
        """
        try:
            # Réduire le bruit
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
            
            # Détection des lignes avec Hough Transform
            lines = cv2.HoughLinesP(cleaned, 1, np.pi/180, 
                                threshold=int(min(binary_img.shape) * 0.3), 
                                minLineLength=int(min(binary_img.shape) * 0.3), 
                                maxLineGap=int(min(binary_img.shape) * 0.1))
            
            if lines is None or len(lines) == 0:
                return False
                
            # Regrouper les lignes par angle
            diagonals = []
            verticals = []
            horizontals = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:  # Division par zéro
                    angle = 90
                else:
                    angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
                    
                # Classer la ligne selon son angle
                if 70 <= angle <= 110:  # Presque vertical
                    verticals.append(line)
                elif angle <= 20 or angle >= 160:  # Presque horizontal
                    horizontals.append(line)
                else:  # Diagonal
                    diagonals.append(line)
            
            # Motif X = au moins 2 diagonales dans des directions différentes
            if len(diagonals) >= 2:
                angles = []
                for line in diagonals:
                    x1, y1, x2, y2 = line[0]
                    if x2 != x1:  # Éviter division par zéro
                        angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                        angles.append(angle)
                
                # Vérifier s'il y a des angles de signes opposés (diagonales croisées)
                if angles and any(a > 0 for a in angles) and any(a < 0 for a in angles):
                    return True
            
            # Motif ✓ = une diagonale + possiblement une horizontale/verticale
            if len(diagonals) >= 1 and (len(horizontals) >= 1 or len(verticals) >= 1):
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Erreur détection pattern: {e}")
            return False
    
    async def _extract_checkbox_context(
        self,
        image: np.ndarray,
        checkbox: Dict[str, Any],
        page_text: str
    ) -> str:
        """
        Extrait le contexte textuel autour d'une case à cocher.
        
        Args:
            image: Image de la page
            checkbox: Informations sur la case à cocher
            page_text: Texte complet de la page
            
        Returns:
            Contexte textuel
        """
        try:
            # Récupérer les coordonnées de la case à cocher
            x, y, w, h = checkbox["x"], checkbox["y"], checkbox["width"], checkbox["height"]
            
            # 1. Approche par OCR - extraire le texte à proximité
            # Définir une zone de contexte plus large
            context_margin = max(h * 5, 100)  # marge adaptative
            
            # Vérifier les limites de l'image
            x1 = max(0, x - context_margin)
            y1 = max(0, y - context_margin // 2)
            x2 = min(image.shape[1], x + w + context_margin)
            y2 = min(image.shape[0], y + h + context_margin // 2)
            
            # Extraire la région de contexte
            context_region = image[y1:y2, x1:x2]
            
            if context_region.size == 0:
                return ""
                
            # OCR sur la région de contexte
            try:
                # Configuration OCR
                custom_config = '--oem 3 --psm 11 -l fra+eng'  # Mode complet, langues FR+EN
                
                # Effectuer l'OCR
                loop = asyncio.get_event_loop()
                context_text = await loop.run_in_executor(
                    self.executor,
                    lambda: pytesseract.image_to_string(context_region, config=custom_config)
                )
                
                # Nettoyer le texte OCR
                context_text = context_text.strip()
                
                # Si aucun texte détecté par OCR ou texte trop court, chercher dans le texte de la page
                if not context_text or len(context_text) < 5:
                    # 2. Approche par recherche dans le texte de la page
                    context_text = await self._find_nearby_text_in_page(page_text, x, y)
            except Exception as ocr_error:
                logger.debug(f"Erreur OCR pour contexte: {ocr_error}")
                # En cas d'erreur OCR, utiliser l'approche de texte de page
                context_text = await self._find_nearby_text_in_page(page_text, x, y)
            
            # Extraire le label potentiel
            label = await self._extract_label_from_context(context_text)
            
            # Ajouter le label détecté au contexte
            if label and label not in context_text:
                context_text = f"{label} - {context_text}"
            
            return context_text
            
        except Exception as e:
            logger.error(f"Erreur extraction contexte: {e}")
            return ""
    
    async def _find_nearby_text_in_page(self, page_text: str, x: int, y: int) -> str:
        """
        Trouve le texte à proximité d'une position donnée dans le texte de la page.
        
        Args:
            page_text: Texte complet de la page
            x, y: Coordonnées de la case à cocher
            
        Returns:
            Texte à proximité
        """
        try:
            # Si aucun texte de page disponible
            if not page_text:
                return ""
                
            # Diviser le texte en paragraphes
            paragraphs = page_text.split('\n\n')
            
            # Sélectionner un segment de texte (approximation sans information spatiale précise)
            # Utilisation d'une heuristique basée sur la position y (hauteur dans la page)
            # Pour estimer quel paragraphe est proche
            
            # Estimer la position relative dans la page (0-1)
            # Supposons que y=0 est en haut et y=1000 est en bas (valeurs approximatives)
            # Ces valeurs pourraient être ajustées en fonction des métadonnées du PDF
            relative_pos = y / 1000.0
            
            # Estimer l'index du paragraphe correspondant
            if paragraphs:
                paragraph_index = min(int(relative_pos * len(paragraphs)), len(paragraphs) - 1)
                
                # Prendre le paragraphe estimé et quelques autres autour
                start_idx = max(0, paragraph_index - 1)
                end_idx = min(len(paragraphs), paragraph_index + 2)
                
                # Joindre les paragraphes pertinents
                relevant_text = '\n'.join(paragraphs[start_idx:end_idx])
                
                # Limiter la longueur du texte
                if len(relevant_text) > 500:
                    relevant_text = relevant_text[:500] + "..."
                
                return relevant_text.strip()
            
            return ""
            
        except Exception as e:
            logger.error(f"Erreur recherche texte: {e}")
            return ""
    
    async def _extract_label_from_context(self, context: str) -> str:
        """
        Extrait un label potentiel d'un texte de contexte avec filtrage amélioré.
        
        Args:
            context: Texte de contexte
            
        Returns:
            Label extrait ou chaîne vide
        """
        try:
            if not context:
                return ""
                
            # Nettoyer le texte - supprimer les caractères spéciaux et les séquences improbables
            context = context.replace('\n', ' ').strip()
            
            # Filtrer les caractères spéciaux isolés et les séquences non significatives
            context = re.sub(r'(\s|^)[^\w\s]{1,2}(\s|$)', ' ', context)
            context = re.sub(r'\s+', ' ', context)
            
            # Supprimer les séquences aléatoires de lettres/chiffres qui semblent être du bruit OCR
            context = re.sub(r'(\s|^)[a-zA-Z0-9]{1,2}(\s|$)', ' ', context)
            
            # Rechercher des motifs plus stricts pour les libellés
            label_patterns = [
                # Option 1: Texte suivi d'une case à cocher
                r'([A-Za-z\u00C0-\u017F][A-Za-z\u00C0-\u017F\s\-_.,;:()/]{3,50})[\s\:]*[□☐☑✓✔✕✖✗✘]',
                
                # Option 2: Case à cocher suivie d'un texte
                r'[□☐☑✓✔✕✖✗✘][\s\:]*([A-Za-z\u00C0-\u017F][A-Za-z\u00C0-\u017F\s\-_.,;:()/]{3,50})',
                
                # Option 3: Texte entre guillemets
                r'[\"\']([A-Za-z\u00C0-\u017F][A-Za-z\u00C0-\u017F\s\-_.,;:()/]{3,50})[\"\']',
                
                # Option 4: Chaîne significative avec majuscule
                r'(\b[A-Z][a-zÀ-ÿ]{2,}\s*(?:[A-Za-zÀ-ÿ]+\s*){0,3})'
            ]
            
            # Rechercher les motifs dans le texte
            for pattern in label_patterns:
                matches = re.findall(pattern, context)
                for match in matches:
                    if match and len(match) >= 3:
                        # Nettoyer et retourner
                        label = match.strip()
                        # Filtrer les labels qui semblent être des codes ou des nombres
                        if not re.match(r'^[0-9]+$', label) and not re.match(r'^[A-Z0-9_]+$', label):
                            return label
            
            # Si aucun motif ne correspond, prendre le premier groupe de mots significatif
            words = context.split()
            words = [w for w in words if len(w) > 2 and not w.isdigit() and not all(c in '.,;:/_-+!?' for c in w)]
            
            if len(words) >= 2:
                # Prendre les premiers mots (probablement le label)
                label = ' '.join(words[:min(4, len(words))]).strip()
                return label
            elif words:
                return words[0].strip()
            
            return ""
            
        except Exception as e:
            logger.error(f"Erreur extraction label: {e}")
            return ""
  
    async def _identify_form_section(self, text: str) -> str:
        """
        Identifie la section du formulaire basée sur le contexte textuel.
        
        Args:
            text: Texte de contexte
            
        Returns:
            Nom de la section identifiée ou chaîne vide
        """
        try:
            if not text:
                return ""
                
            # Normaliser le texte
            text_lower = text.lower()
            
            # Dictionnaire de patterns pour différentes sections
            section_patterns = {
                "identification": ["identité", "identification", "identite", "coordonnées", "nom", "prénom", "personne", "patient"],
                "facturation": ["facturation", "facture", "paiement", "prix", "montant", "tarif", "financier"],
                "signature": ["signature", "signer", "approuvé", "validé", "atteste", "certifie"],
                "contact": ["contact", "adresse", "téléphone", "email", "courriel", "portable"],
                "formation": ["formation", "compétence", "diplôme", "certification", "qualification"],
                "experience": ["expérience", "travail", "emploi", "poste", "fonction"],
                "consentement": ["consentement", "accord", "accepte", "approuve", "autorise"],
                "medical": ["médical", "santé", "symptôme", "diagnostic", "traitement", "pathologie"],
                "transport": ["transport", "livraison", "expédition", "véhicule", "automobile", "logistique"],
                "options": ["option", "choix", "préférence", "sélection"]
            }
            
            # Rechercher les patterns dans le texte
            matches = {}
            for section, patterns in section_patterns.items():
                count = sum(1 for pattern in patterns if pattern in text_lower)
                if count > 0:
                    matches[section] = count
            
            # Retourner la section avec le plus grand nombre de matches
            if matches:
                return max(matches.items(), key=lambda x: x[1])[0]
            
            return ""
            
        except Exception as e:
            logger.error(f"Erreur identification section: {e}")
            return ""
    
    async def _extract_field_name(self, text: str) -> str:
        """
        Extrait un nom de champ potentiel du texte de contexte avec validation améliorée.
        
        Args:
            text: Texte de contexte
            
        Returns:
            Nom de champ ou chaîne vide
        """
        try:
            if not text or len(text) < 3:
                return ""
                
            # Nettoyer le texte
            text = text.replace('\n', ' ').strip()
            
            # Supprimer les caractères spéciaux isolés et les chiffres isolés
            text = re.sub(r'(\s|^)[^\w\s]{1,2}(\s|$)', ' ', text)
            text = re.sub(r'(\s|^)\d{1,2}(\s|$)', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Rechercher des motifs plus spécifiques de noms de champs
            field_patterns = [
                # Format: label: valeur
                r'(?:^|\s)([A-Za-z][A-Za-z0-9_]{3,30})(?:\s*[:=])',
                
                # Format: préfixe_mot significatif
                r'(?:champ|field|input|form|formulaire|question)(?:\s*[:=]\s*)([A-Za-z][A-Za-z0-9_]{3,30})',
                
                # Format: [nom_champ]
                r'\[([A-Za-z][A-Za-z0-9_]{3,30})\]',
                
                # Format: nom significatif (>3 lettres, commençant par une lettre)
                r'\b([A-Za-z][A-Za-z]{3,}[A-Za-z0-9_]*)\b'
            ]
            
            # Rechercher les motifs
            for pattern in field_patterns:
                matches = re.search(pattern, text, re.IGNORECASE)
                if matches and matches.group(1):
                    # Valider si le nom de champ est significatif (pas de noms génériques)
                    field_name = matches.group(1).strip()
                    
                    # Vérifier si ce n'est pas un mot vide ou trop général
                    if field_name.lower() not in ['champ', 'field', 'input', 'form', 'the', 'les', 'des', 'pour']:
                        # Normaliser en format snake_case
                        field_name = re.sub(r'\s+', '_', field_name).lower()
                        field_name = re.sub(r'[^a-z0-9_]', '', field_name)
                        
                        # Vérifier longueur minimale après traitement
                        if len(field_name) >= 3:
                            return field_name
            
            # Si aucun pattern ne correspond, utiliser une approche plus simple mais robuste
            # Extraire le premier mot significatif (au moins 4 lettres)
            words = text.split()
            for word in words:
                # Nettoyer le mot
                clean_word = re.sub(r'[^a-zA-Z0-9]', '', word)
                if len(clean_word) >= 4 and not clean_word.isdigit():
                    # Convertir en format snake_case
                    field_name = clean_word.lower()
                    field_name = re.sub(r'[^a-z0-9]', '_', field_name)
                    
                    # Limiter la longueur
                    field_name = field_name[:30]
                    
                    return field_name
            
            # Dernier recours: générer un nom basé sur les premiers caractères du texte
            if len(text) >= 3:
                # Extraire les premiers caractères alphabétiques
                alpha_chars = ''.join(c for c in text[:20] if c.isalpha()).lower()
                if len(alpha_chars) >= 3:
                    return alpha_chars[:20]
            
            # Si tout échoue, générer un nom générique mais éviter field_xxx si possible
            seed = abs(hash(text)) % 1000
            return f"checkbox_{seed}"
            
        except Exception as e:
            logger.error(f"Erreur extraction nom de champ: {e}")
            return f"checkbox_{int(time.time()) % 1000}"
    
    async def _post_validate_checkboxes(self, checkboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Effectue une validation finale pour éliminer les faux positifs.
        
        Args:
            checkboxes: Liste de cases à cocher détectées
            
        Returns:
            Liste filtrée de cases à cocher validées
        """
        if not checkboxes:
            return []
            
        valid_checkboxes = []
        
        for checkbox in checkboxes:
            # Validation basée sur plusieurs critères
            
            # 1. Validation par score de confiance
            if checkbox["confidence"] < 0.55:  # Seuil plus élevé pour la confiance
                continue
                
            # 2. Validation par contexte textuel
            context = checkbox.get("text_context", "")
            if context:
                # Vérifier s'il y a un texte sensé (pas juste des caractères spéciaux)
                has_meaningful_text = bool(re.search(r'[a-zA-Z]{3,}', context))
                if not has_meaningful_text:
                    # Réduire la confiance pour les cases sans contexte textuel significatif
                    checkbox["confidence"] *= 0.8
                    if checkbox["confidence"] < 0.55:
                        continue
            
            # 3. Validation par cohérence de type
            checkbox_type = checkbox.get("type", "")
            if checkbox_type == "symbol":
                # Les cases de type symbol doivent avoir un score élevé pour être acceptées
                if checkbox["confidence"] < 0.6:
                    continue
            
            # 4. Validation des dimensions (rapport hauteur/largeur)
            w, h = checkbox.get("width", 0), checkbox.get("height", 0)
            if w > 0 and h > 0:
                aspect_ratio = w / h
                # Filtrer les formes trop allongées
                if aspect_ratio < 0.6 or aspect_ratio > 1.7:
                    continue
            
            # 5. Validation du nom de champ
            field_name = checkbox.get("field_name", "")
            if field_name:
                # Ignorer les noms de champs trop courts ou générés
                if len(field_name) < 3 or field_name.startswith("field_"):
                    # Essayer de générer un meilleur nom basé sur le contexte
                    better_name = await self._extract_better_field_name(context)
                    if better_name:
                        checkbox["field_name"] = better_name
            else:
                # Essayer de générer un nom basé sur le contexte
                checkbox["field_name"] = await self._extract_better_field_name(context)
            
            # 6. Validation du label
            label = checkbox.get("label", "")
            if not label and context:
                # Essayer de détecter un meilleur label
                better_label = await self._extract_better_label(context)
                if better_label:
                    checkbox["label"] = better_label
            
            # La case a passé toutes les validations
            valid_checkboxes.append(checkbox)
        
        return valid_checkboxes

    async def _extract_better_field_name(self, context: str) -> str:
        """Tente d'extraire un meilleur nom de champ du contexte."""
        if not context or len(context) < 3:
            return ""
            
        # Nettoyer le contexte
        context = re.sub(r'[^\w\s\-_]', ' ', context)
        context = re.sub(r'\s+', ' ', context).strip().lower()
        
        # Extraire des mots clés significatifs
        words = context.split()
        words = [w for w in words if len(w) >= 3 and not w.isdigit()]
        
        if not words:
            return ""
            
        # Utiliser le premier mot significatif comme nom de base
        base_name = words[0]
        
        # Si possible, combiner avec un second mot pour plus de spécificité
        if len(words) > 1:
            return f"{base_name}_{words[1]}"
        
        return base_name

    async def _extract_better_label(self, context: str) -> str:
        """Tente d'extraire un meilleur label du contexte."""
        if not context:
            return ""
            
        # Nettoyer le contexte
        clean_context = re.sub(r'[^\w\s\-_.,;:]', ' ', context)
        clean_context = re.sub(r'\s+', ' ', clean_context).strip()
        
        # Extraire les phrases ou groupes de mots
        parts = re.split(r'[.;:]', clean_context)
        parts = [p.strip() for p in parts if len(p.strip()) > 3]
        
        if parts:
            # Prendre la première partie qui a un sens
            for part in parts:
                # Vérifier si la partie contient des mots significatifs
                if re.search(r'[A-Za-z]{3,}', part):
                    # Limiter la longueur
                    if len(part) > 50:
                        return part[:47] + "..."
                    return part
        
        # Si rien n'est trouvé, prendre au moins quelques mots
        words = clean_context.split()
        if len(words) >= 3:
            return " ".join(words[:3]) + "..."
        elif words:
            return " ".join(words)
        
        return ""

    async def _detect_checkboxes_with_ai(
        self,
        image: np.ndarray,
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Détecte les cases à cocher en utilisant le modèle IA.
        
        Args:
            image: Image à analyser
            confidence_threshold: Seuil de confiance
            
        Returns:
            Liste des cases à cocher détectées
        """
        try:
            if not self.table_detector:
                return []
                
            # Utiliser le même modèle de détection de tableaux pour détecter les cases à cocher
            # en adaptant les paramètres pour les petits objets
            
            # Convertir en PIL Image pour le modèle
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Détecter les objets - on peut réutiliser la méthode de détection des tableaux
            # car les modèles DETR peuvent souvent détecter plusieurs types d'objets
            detections = await self.table_detector.detect_tables(image)
            
            checkboxes = []
            for detection in detections:
                # Filtrer par taille - les cases à cocher sont généralement plus petites que les tableaux
                if (self.min_size <= detection["width"] <= self.max_size and 
                    self.min_size <= detection["height"] <= self.max_size):
                    
                    # Vérifier si c'est un carré approximatif
                    aspect_ratio = detection["width"] / detection["height"]
                    if 0.7 <= aspect_ratio <= 1.3:
                        # Extraire la région
                        x, y = detection["x"], detection["y"]
                        w, h = detection["width"], detection["height"]
                        
                        checkbox_region = image[y:y+h, x:x+w]
                        
                        # Déterminer si la case est cochée
                        is_checked, check_confidence = await self._is_checkbox_checked(checkbox_region, "rectangular")
                        
                        # Ajouter à la liste des cases à cocher
                        if detection["score"] >= confidence_threshold:
                            checkboxes.append({
                                "x": detection["x"],
                                "y": detection["y"],
                                "width": detection["width"],
                                "height": detection["height"],
                                "confidence": detection["score"],
                                "type": "rectangular",
                                "is_checked": is_checked,
                                "check_confidence": check_confidence
                            })
            
            return checkboxes
            
        except Exception as e:
            logger.error(f"Erreur détection IA des cases à cocher: {e}")
            return []
            
    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            if self.executor:
                self.executor.shutdown(wait=False)
                
            if self.checkbox_model:
                self.checkbox_model = None
                
            if self.image_processor:
                self.image_processor.cleanup()
                
            logger.info("Ressources de l'extracteur de cases à cocher nettoyées")
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage de l'extracteur de cases à cocher: {e}")