# core/document_processing/table_extraction/checkbox_extractor.py
"""
Module d'extraction des cases à cocher dans les PDFs.
Version simplifiée et plus robuste pour détecter les cases à cocher et leur état.
"""
from typing import Dict, List, Any, Optional, Union, Tuple
import fitz  # PyMuPDF
import numpy as np
import cv2
import os
import io
import logging
from datetime import datetime
import re
import json

from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger("checkbox_extractor")

class CheckboxExtractor:
    """
    Classe simplifiée pour l'extraction des cases à cocher dans les documents PDF.
    Utilise une approche hybride combinant détection visuelle et analyse de texte.
    """
    
    def __init__(self):
        """Initialise l'extracteur avec des paramètres par défaut."""
        # Paramètres de base pour la détection
        self.min_size = 8       # Taille minimale d'une case à cocher en pixels
        self.max_size = 30      # Taille maximale d'une case à cocher en pixels
        self.confidence_threshold = 0.6  # Seuil de confiance par défaut
        self.dpi = 300          # Résolution pour la conversion en image
        self.cache = {}         # Cache des résultats
    
    async def extract_checkboxes_from_pdf(
        self, 
        pdf_path: Union[str, io.BytesIO],
        page_range: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extrait les cases à cocher d'un document PDF avec une approche simplifiée et robuste.
        
        Args:
            pdf_path: Chemin du fichier PDF ou objet BytesIO
            page_range: Liste optionnelle des pages à analyser (1-based)
            config: Configuration pour affiner la détection
                - confidence_threshold: seuil de confiance (défaut: 0.6)
                - enhance_detection: améliorer la détection (défaut: True)
                - include_images: inclure les images des cases (défaut: False)
        
        Returns:
            Dictionnaire contenant toutes les cases à cocher détectées et leur état
        """
        try:
            with metrics.timer("checkbox_extraction"):
                # Préparer la configuration
                conf = config or {}
                confidence_threshold = conf.get("confidence_threshold", self.confidence_threshold)
                enhance_detection = conf.get("enhance_detection", True)
                include_images = conf.get("include_images", False)
                
                # Vérifier si résultat en cache
                if isinstance(pdf_path, str):
                    cache_key = f"{pdf_path}:{str(page_range)}:{confidence_threshold}"
                    if cache_key in self.cache:
                        logger.info(f"Résultat trouvé en cache pour {pdf_path}")
                        return self.cache[cache_key]
                
                # Ouvrir le document PDF
                pdf_doc = self._open_pdf(pdf_path)
                if not pdf_doc:
                    return {"error": "Impossible d'ouvrir le PDF", "checkboxes": []}
                
                try:
                    # Normaliser le page_range
                    page_indices = self._normalize_page_range(pdf_doc, page_range)
                    
                    # Structure des résultats
                    results = {
                        "metadata": {
                            "filename": self._get_filename(pdf_path),
                            "page_count": len(pdf_doc),
                            "processed_pages": len(page_indices),
                            "extraction_date": datetime.now().isoformat(),
                            "config": {
                                "confidence_threshold": confidence_threshold,
                                "enhance_detection": enhance_detection
                            }
                        },
                        "checkboxes": [],
                        "form_values": {},
                        "form_sections": {}
                    }
                    
                    # Traiter chaque page
                    for page_idx in page_indices:
                        if page_idx >= len(pdf_doc):
                            continue
                            
                        page = pdf_doc[page_idx]
                        page_num = page_idx + 1  # Convertir en 1-based
                        
                        # Extraire le texte et convertir en image pour analyse visuelle
                        page_text = page.get_text("dict")
                        page_image = self._convert_page_to_image(page)
                        
                        # Détecter les cases à cocher (approche hybride)
                        checkboxes = await self._detect_checkboxes(
                            page, 
                            page_text, 
                            page_image, 
                            page_num, 
                            confidence_threshold,
                            enhance_detection
                        )
                        
                        # Ajouter au résultat
                        results["checkboxes"].extend(checkboxes)
                        
                        # Ajouter les images si demandé
                        if include_images and checkboxes:
                            if "checkbox_images" not in results:
                                results["checkbox_images"] = []
                            
                            for checkbox in checkboxes:
                                image_data = self._extract_checkbox_image(page, checkbox, page_image)
                                if image_data:
                                    results["checkbox_images"].append({
                                        "checkbox_id": checkbox.get("id", len(results["checkbox_images"])),
                                        "page": page_num,
                                        "data": image_data
                                    })
                    
                    # Post-traitement pour organiser les cases à cocher
                    self._organize_checkboxes(results)
                    
                    # Mettre en cache le résultat
                    if isinstance(pdf_path, str):
                        self.cache[cache_key] = results
                    
                    return results
                
                finally:
                    # Fermer le document
                    pdf_doc.close()
                    
        except Exception as e:
            logger.error(f"Erreur extraction cases à cocher: {e}")
            metrics.increment_counter("checkbox_extraction_errors")
            return {
                "error": str(e),
                "checkboxes": []
            }
    
    async def extract_form_checkboxes(
        self,
        pdf_path: Union[str, io.BytesIO],
        page_range: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Version simplifiée qui utilise la même méthode principale.
        Pour compatibilité avec l'API existante.
        """
        # Simplement appeler la méthode principale
        return await self.extract_checkboxes_from_pdf(pdf_path, page_range, config)
    
    def extract_selected_values(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Extrait uniquement les valeurs sélectionnées (cases cochées).
        
        Args:
            results: Résultats de l'extraction
            
        Returns:
            Dictionnaire avec les paires label:valeur pour les cases cochées
        """
        selected_values = {}
        
        # Parcourir les cases à cocher
        for checkbox in results.get("checkboxes", []):
            label = checkbox.get("label", "").strip()
            if not label:
                continue
                
            value = checkbox.get("value", "").strip()
            checked = checkbox.get("checked", False)
            
            # Déterminer la valeur en fonction du type
            if value.lower() in ["oui", "yes", "true", "non", "no", "false"]:
                # Pour les cases de type Oui/Non
                if checked:
                    selected_values[label] = value
            else:
                # Pour les cases à cocher simples
                selected_values[label] = "Oui" if checked else "Non"
        
        return selected_values
    
    def _open_pdf(self, pdf_path: Union[str, io.BytesIO]) -> Optional[fitz.Document]:
        """Ouvre un document PDF depuis un chemin ou un BytesIO."""
        try:
            if isinstance(pdf_path, str):
                return fitz.open(pdf_path)
            else:
                # Pour BytesIO, on doit réinitialiser la position
                if hasattr(pdf_path, 'seek'):
                    pdf_path.seek(0)
                
                # Ouvrir comme stream
                return fitz.open(stream=pdf_path.read(), filetype="pdf")
        except Exception as e:
            logger.error(f"Erreur ouverture PDF: {e}")
            return None
    
    def _normalize_page_range(self, pdf_doc: fitz.Document, page_range: Optional[List[int]]) -> List[int]:
        """Normalise la plage de pages pour l'extraction."""
        if page_range is None:
            return list(range(len(pdf_doc)))
        else:
            # Convertir en 0-based et filtrer les pages valides
            return [p-1 for p in page_range if 0 <= p-1 < len(pdf_doc)]
    
    def _get_filename(self, pdf_path: Union[str, io.BytesIO]) -> str:
        """Récupère le nom du fichier à partir du chemin ou de l'objet BytesIO."""
        if isinstance(pdf_path, str):
            return os.path.basename(pdf_path)
        elif hasattr(pdf_path, 'name'):
            return os.path.basename(pdf_path.name)
        else:
            return "unknown.pdf"
    
    def _convert_page_to_image(self, page: fitz.Page) -> np.ndarray:
        """Convertit une page PDF en image numpy."""
        try:
            # Création d'un pixmap avec une résolution adaptée
            pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi/72, self.dpi/72))
            
            # Conversion en array numpy
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            # Conversion en RGB ou niveaux de gris selon le nombre de canaux
            if pix.n == 4:  # RGBA
                return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1:  # Grayscale
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                return img
                
        except Exception as e:
            logger.error(f"Erreur conversion page en image: {e}")
            # Retourner une image vide en cas d'erreur
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    async def _detect_checkboxes(
        self, 
        page: fitz.Page, 
        page_texts: Dict, 
        page_image: np.ndarray, 
        page_num: int,
        confidence_threshold: float,
        enhance_detection: bool
    ) -> List[Dict[str, Any]]:
        """
        Détecte les cases à cocher avec une approche hybride avancée.
        
        Args:
            page: Page PDF
            page_texts: Textes extraits (différents formats)
            page_image: Image de la page
            page_num: Numéro de la page
            confidence_threshold: Seuil de confiance
            enhance_detection: Activer les améliorations de détection
            
        Returns:
            Liste des cases à cocher détectées
        """
        # Convertir en niveaux de gris pour traitement d'image
        if len(page_image.shape) > 2:
            gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = page_image
        
        # 1. Détection basée sur le texte (symboles de case à cocher)
        symbol_checkboxes = self._detect_by_symbols(page_texts["dict"], page_num)
        
        # 2. Détection visuelle (basée sur l'image)
        visual_checkboxes = self._detect_by_vision(gray, page_num, enhance_detection)
        
        # 3. Fusionner et dédupliquer les résultats
        merged_checkboxes = self._merge_checkbox_results(symbol_checkboxes, visual_checkboxes)
        
        # 4. Déterminer l'état de chaque case (cochée ou non)
        for checkbox in merged_checkboxes:
            # Si pas déjà déterminé par la détection symbolique
            if "checked" not in checkbox:
                bbox = checkbox.get("bbox", [0, 0, 0, 0])
                x, y, x2, y2 = bbox
                
                # Extraire la région de la case
                roi = gray[y:y2, x:x2] if 0 <= y < y2 <= gray.shape[0] and 0 <= x < x2 <= gray.shape[1] else None
                
                # Déterminer si cochée
                if roi is not None and roi.size > 0:
                    is_checked = self._is_checkbox_checked(roi)
                    checkbox["checked"] = is_checked
                else:
                    checkbox["checked"] = False
        
        # 5. Associer les étiquettes aux cases
        # (Cette étape initiale sera complétée par l'association contextuelle plus tard)
        for checkbox in merged_checkboxes:
            if not checkbox.get("label"):
                bbox = checkbox.get("bbox", [0, 0, 0, 0])
                label = self._find_closest_text(page_texts, bbox)
                checkbox["label"] = label
        
        # 6. Filtrer par seuil de confiance
        filtered_checkboxes = [
            cb for cb in merged_checkboxes 
            if cb.get("confidence", 0) >= confidence_threshold
        ]
        
        # Ajouter un identifiant unique à chaque case
        for i, checkbox in enumerate(filtered_checkboxes):
            checkbox["id"] = f"checkbox_{page_num}_{i}"
            
            # Assurer la présence des champs essentiels
            if "value" not in checkbox:
                checkbox["value"] = ""
            if "section" not in checkbox:
                checkbox["section"] = "Information"
        
        return filtered_checkboxes
    
    def _detect_by_symbols(self, page_text: Dict, page_num: int) -> List[Dict[str, Any]]:
        """
        Détecte les cases à cocher basées sur les symboles dans le texte.
        
        Args:
            page_text: Texte extrait de la page
            page_num: Numéro de la page
            
        Returns:
            Liste des cases à cocher trouvées
        """
        checkboxes = []
        
        # Symboles de case à cocher courants
        checkbox_symbols = ["☐", "☑", "☒", "□", "■", "▢", "▣", "▪", "▫"]
        
        # Parcourir les blocs de texte
        for block in page_text.get("blocks", []):
            if block.get("type", -1) != 0:  # Ignorer les blocs non-textuels
                continue
                
            for line in block.get("lines", []):
                line_text = ""
                has_checkbox = False
                is_checked = False
                
                # Récupérer le texte et vérifier la présence de symboles
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    line_text += span_text
                    
                    # Vérifier la présence de symboles
                    for symbol in checkbox_symbols:
                        if symbol in span_text:
                            has_checkbox = True
                            # Détecter si la case est cochée
                            if symbol in ["☑", "☒", "■"]:
                                is_checked = True
                
                if has_checkbox:
                    # Extraire les coordonnées
                    bbox = line.get("bbox", [0, 0, 0, 0])
                    
                    # Analyser le texte pour séparer label et case
                    label = line_text
                    for symbol in checkbox_symbols:
                        label = label.replace(symbol, "").strip()
                    
                    # Créer l'élément checkbox
                    checkbox = {
                        "label": label,
                        "checked": is_checked,
                        "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                        "page": page_num,
                        "confidence": 0.9,  # Haute confiance pour la détection par symboles
                        "method": "symbol"
                    }
                    
                    # Détection améliorée Oui/Non
                    if re.search(r'\b(oui|non|yes|no)\b', label.lower()):
                        match = re.search(r'\b(oui|non|yes|no)\b', label.lower())
                        checkbox["value"] = match.group(1).capitalize()
                    
                    checkboxes.append(checkbox)
        
        return checkboxes
    
    def _detect_by_vision(self, gray_img: np.ndarray, page_num: int, enhance: bool) -> List[Dict[str, Any]]:
        """
        Détecte les cases à cocher visuellement dans l'image.
        
        Args:
            gray_img: Image en niveaux de gris
            page_num: Numéro de la page
            enhance: Activer les améliorations
            
        Returns:
            Liste des cases à cocher détectées
        """
        try:
            checkboxes = []
            
            # 1. Prétraitement de l'image
            # Binarisation pour meilleure détection
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 2. Détection des contours
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # 3. Filtrer les contours
            for contour in contours:
                # Récupérer le rectangle englobant
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filtrer par taille
                if self.min_size <= w <= self.max_size and self.min_size <= h <= self.max_size:
                    # Vérifier si c'est approximativement un carré
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if 0.7 <= aspect_ratio <= 1.3:  # Tolérance pour quasi-carrés
                        # Calcul supplémentaire pour confirmer que c'est une case
                        area = cv2.contourArea(contour)
                        rect_area = w * h
                        solidity = float(area) / rect_area if rect_area > 0 else 0
                        
                        # Les cases à cocher ont généralement une solidité élevée
                        if solidity > 0.7:
                            # Créer la checkbox
                            checkbox = {
                                "bbox": [x, y, x+w, y+h],
                                "page": page_num,
                                "confidence": min(0.9, solidity),  # La solidité influence la confiance
                                "method": "vision"
                            }
                            
                            checkboxes.append(checkbox)
            
            # Améliorations supplémentaires si demandées
            if enhance and len(checkboxes) < 5:  # Si peu de cases détectées, utiliser des méthodes supplémentaires
                # Détection des lignes pour trouver des rectangles
                horizontal = self._detect_lines(binary, True)
                vertical = self._detect_lines(binary, False)
                
                # Combiner les lignes horizontales et verticales
                combined = cv2.bitwise_or(horizontal, vertical)
                
                # Chercher les intersections
                contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Filtrer par taille et forme (quasi-carré)
                    if (self.min_size <= w <= self.max_size and 
                        self.min_size <= h <= self.max_size and
                        0.7 <= float(w) / h <= 1.3):
                        
                        # Déjà dans la liste?
                        is_duplicate = False
                        for existing in checkboxes:
                            ex, ey, ex2, ey2 = existing["bbox"]
                            overlap = self._calculate_overlap(
                                [x, y, x+w, y+h],
                                [ex, ey, ex2, ey2]
                            )
                            if overlap > 0.5:  # 50% de chevauchement
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            checkbox = {
                                "bbox": [x, y, x+w, y+h],
                                "page": page_num,
                                "confidence": 0.7,  # Confiance plus faible pour cette méthode
                                "method": "lines"
                            }
                            checkboxes.append(checkbox)
            
            return checkboxes
            
        except Exception as e:
            logger.error(f"Erreur détection visuelle: {e}")
            return []
    
    def _detect_lines(self, binary_img: np.ndarray, is_horizontal: bool) -> np.ndarray:
        """Détecte les lignes horizontales ou verticales."""
        # Adapter le noyau selon l'orientation
        if is_horizontal:
            kernel_length = binary_img.shape[1] // 30
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        else:
            kernel_length = binary_img.shape[0] // 30
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        
        # Appliquer les opérations morphologiques
        detected = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        return detected
    
    def _calculate_overlap(self, bbox1, bbox2):
        """Calcule le chevauchement entre deux boîtes englobantes."""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        # Calculer l'intersection
        x_overlap = max(0, min(x2, x4) - max(x1, x3))
        y_overlap = max(0, min(y2, y4) - max(y1, y3))
        intersection = x_overlap * y_overlap
        
        # Calculer l'union
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        
        # Retourner le rapport intersection/union
        return intersection / union if union > 0 else 0
    
    def _merge_checkbox_results(self, symbol_checkboxes, visual_checkboxes):
        """Fusionne les résultats des deux méthodes de détection en évitant les doublons."""
        if not symbol_checkboxes:
            return visual_checkboxes
        if not visual_checkboxes:
            return symbol_checkboxes
        
        merged = symbol_checkboxes.copy()
        
        # Pour chaque case détectée visuellement
        for visual_cb in visual_checkboxes:
            vx1, vy1, vx2, vy2 = visual_cb["bbox"]
            
            # Vérifier si elle chevauche une case déjà détectée par symbole
            is_duplicate = False
            for symbol_cb in symbol_checkboxes:
                sx1, sy1, sx2, sy2 = symbol_cb["bbox"]
                
                # Calcul de chevauchement
                overlap = self._calculate_overlap(
                    [vx1, vy1, vx2, vy2],
                    [sx1, sy1, sx2, sy2]
                )
                
                if overlap > 0.3:  # 30% de chevauchement suffit
                    is_duplicate = True
                    break
            
            # Si ce n'est pas un doublon, l'ajouter
            if not is_duplicate:
                merged.append(visual_cb)
        
        return merged
    
    def _is_checkbox_checked(self, checkbox_img: np.ndarray) -> bool:
        """
        Détermine si une case à cocher est cochée ou non.
        
        Args:
            checkbox_img: Image de la case à cocher
            
        Returns:
            True si la case est cochée, False sinon
        """
        try:
            # S'assurer que l'image n'est pas vide
            if checkbox_img is None or checkbox_img.size == 0:
                return False
            
            # Si l'image est en couleur, la convertir en niveaux de gris
            if len(checkbox_img.shape) > 2:
                gray = cv2.cvtColor(checkbox_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = checkbox_img
                
            # 1. Binarisation pour détecter les marques
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 2. Calculer différentes métriques pour déterminer si la case est cochée
            
            # Pourcentage de pixels non-blancs (marques)
            total_pixels = binary.size
            marked_pixels = np.sum(binary > 0)
            fill_ratio = marked_pixels / total_pixels if total_pixels > 0 else 0
            
            # Vérifier si la marque est concentrée au centre (typique d'une coche)
            height, width = binary.shape
            center_margin = max(2, min(width, height) // 4)
            
            if width > 2*center_margin and height > 2*center_margin:
                # Extraire la région centrale
                center_binary = binary[
                    center_margin:height-center_margin, 
                    center_margin:width-center_margin
                ]
                
                # Calculer le ratio de remplissage central
                center_pixels = center_binary.size
                center_marked = np.sum(center_binary > 0)
                center_ratio = center_marked / center_pixels if center_pixels > 0 else 0
                
                # Combiner les deux ratios avec plus de poids pour le centre
                combined_ratio = (fill_ratio + 2 * center_ratio) / 3
                
                # La case est considérée cochée si le ratio combiné dépasse un seuil
                return combined_ratio > 0.15
            else:
                # Pour les petites cases, utiliser juste le ratio global
                return fill_ratio > 0.2
        
        except Exception as e:
            logger.debug(f"Erreur analyse case cochée: {e}")
            return False
    
    def _find_closest_text(self, page_text: Dict, bbox: List[int]) -> str:
        """
        Trouve le texte le plus proche d'une case à cocher avec des améliorations.
        
        Args:
            page_text: Texte structuré de la page
            bbox: Rectangle englobant de la case [x1, y1, x2, y2]
            
        Returns:
            Texte le plus proche
        """
        if not page_text or "blocks" not in page_text:
            return ""
        
        # Coordonnées du centre de la case
        x1, y1, x2, y2 = bbox
        checkbox_center_x = (x1 + x2) / 2
        checkbox_center_y = (y1 + y2) / 2
        
        closest_text = ""
        min_distance = float('inf')
        max_distance = 200  # Distance maximale à considérer (augmentée)
        
        # Pour stocker tous les textes candidats
        candidates = []
        
        # Chercher dans les blocs de texte
        for block in page_text["blocks"]:
            if block["type"] != 0:  # Ignorer les blocs non-textuels
                continue
                
            for line in block["lines"]:
                line_bbox = line["bbox"]
                line_center_x = (line_bbox[0] + line_bbox[2]) / 2
                line_center_y = (line_bbox[1] + line_bbox[3]) / 2
                
                # Calculer la distance euclidienne
                distance = ((line_center_x - checkbox_center_x) ** 2 + 
                        (line_center_y - checkbox_center_y) ** 2) ** 0.5
                
                if distance < max_distance:
                    line_text = " ".join([span["text"] for span in line["spans"]])
                    
                    # Nettoyer le texte (enlever les symboles de case à cocher)
                    cleaned_text = line_text
                    for symbol in ["☐", "☑", "☒", "□", "■", "▢", "▣", "▪", "▫"]:
                        cleaned_text = cleaned_text.replace(symbol, "").strip()
                    
                    if cleaned_text:
                        # Ajouter aux candidats
                        candidates.append({
                            "text": cleaned_text,
                            "distance": distance,
                            "coords": [line_center_x, line_center_y]
                        })

        # Analyser les candidats en considérant leur position relative
        for candidate in candidates:
            distance = candidate["distance"]
            x, y = candidate["coords"]
            
            # Facteurs de pondération selon la position relative
            position_factor = 1.0
            
            # Privilégier fortement les textes à droite (association classique étiquette-case)
            if x > checkbox_center_x:
                # À droite = meilleur candidat
                position_factor = 0.7
            elif x < checkbox_center_x and abs(y - checkbox_center_y) < 15:
                # À gauche sur la même ligne = bon candidat aussi
                position_factor = 0.8
            else:
                # Positions moins probables
                position_factor = 1.2
            
            # Appliquer la pondération
            adjusted_distance = distance * position_factor
            
            # Vérifier le texte (privilégier les textes courts, probablement des étiquettes)
            text = candidate["text"]
            if len(text) < 30:  # Probablement une étiquette
                adjusted_distance *= 0.9
            
            # Les textes contenant "Oui" ou "Non" sont de bons candidats pour les cases à cocher
            if re.search(r'\b(oui|non|yes|no)\b', text.lower()):
                adjusted_distance *= 0.8
            
            # Mettre à jour le texte le plus proche
            if adjusted_distance < min_distance:
                min_distance = adjusted_distance
                closest_text = text
        
        return closest_text
    
    def _extract_checkbox_image(self, page: fitz.Page, checkbox: Dict[str, Any], page_image: np.ndarray) -> Optional[str]:
        """
        Extrait l'image d'une case à cocher pour débogage.
        
        Args:
            page: Page du document
            checkbox: Information sur la case à cocher
            page_image: Image complète de la page
            
        Returns:
            Image encodée en base64 ou None
        """
        try:
            import base64
            
            # Extraire les coordonnées avec une marge
            bbox = checkbox.get("bbox", [0, 0, 0, 0])
            x, y, x2, y2 = bbox
            
            # Ajouter une marge
            margin = 5
            x_with_margin = max(0, x - margin)
            y_with_margin = max(0, y - margin)
            width_with_margin = (x2 - x) + 2 * margin
            height_with_margin = (y2 - y) + 2 * margin
            
            # S'assurer que les coordonnées sont dans les limites
            height, width = page_image.shape[:2]
            x_end = min(x_with_margin + width_with_margin, width)
            y_end = min(y_with_margin + height_with_margin, height)
            
            # Extraire la région
            if x_with_margin < x_end and y_with_margin < y_end:
                checkbox_img = page_image[y_with_margin:y_end, x_with_margin:x_end]
                
                # Convertir en PNG et encoder en base64
                _, buffer = cv2.imencode('.png', checkbox_img)
                return base64.b64encode(buffer).decode('utf-8')
            
            return None
            
        except Exception as e:
            logger.debug(f"Erreur extraction image case: {e}")
            return None
    
    def _organize_checkboxes(self, results: Dict[str, Any]) -> None:
        """
        Organise les cases à cocher par sections et crée les valeurs de formulaire.
        
        Args:
            results: Résultats de l'extraction à modifier in-place
        """
        # 1. Extraire les valeurs de formulaire
        results["form_values"] = self.extract_selected_values(results)
        
        # 2. Organiser par sections
        sections = {}
        
        for checkbox in results.get("checkboxes", []):
            section = checkbox.get("section", "Information")
            
            if section not in sections:
                sections[section] = {}
            
            label = checkbox.get("label", "")
            if label:
                value = checkbox.get("value", "")
                checked = checkbox.get("checked", False)
                
                # Pour les champs de type Oui/Non
                if value.lower() in ["oui", "yes", "true", "non", "no", "false"]:
                    if checked:  # Si coché, prendre la valeur explicite
                        sections[section][label] = value
                else:
                    # Pour les cases à cocher simples
                    sections[section][label] = "Oui" if checked else "Non"
        
        results["form_sections"] = sections