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
                        page_texts = page.get_text("dict")
                        page_image = self._convert_page_to_image(page)
                        
                        # Détecter les cases à cocher (approche hybride)
                        checkboxes = await self._detect_checkboxes(
                            page, 
                            page_texts, 
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
        Détecte les cases à cocher avec une approche hybride améliorée.
        
        Args:
            page: Page PDF
            page_texts: Textes extraits
            page_image: Image de la page
            page_num: Numéro de la page
            confidence_threshold: Seuil de confiance
            enhance_detection: Activer les améliorations
            
        Returns:
            Liste des cases à cocher détectées
        """
        # Convertir en niveaux de gris pour traitement d'image
        if len(page_image.shape) > 2:
            gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = page_image
        
        # 1. Détection basée sur le texte (symboles de case à cocher)
        symbol_checkboxes = self._detect_by_symbols(page_texts, page_num)
        
        # 2. Détection visuelle (basée sur l'image)
        visual_checkboxes = self._detect_by_vision(gray, page_num, enhance_detection)
        
        # 3. Fusionner et dédupliquer les résultats avec la méthode améliorée
        merged_checkboxes = self._merge_checkbox_results(symbol_checkboxes, visual_checkboxes)
        
        # 4. Déterminer l'état de chaque case (cochée ou non) avec la méthode améliorée
        for checkbox in merged_checkboxes:
            # Si pas déjà déterminé par la détection symbolique
            if "checked" not in checkbox:
                bbox = checkbox.get("bbox", [0, 0, 0, 0])
                x, y, x2, y2 = bbox
                
                # Ajouter une petite marge pour capturer toute la case
                margin = 2
                x = max(0, x - margin)
                y = max(0, y - margin)
                x2 = min(gray.shape[1], x2 + margin)
                y2 = min(gray.shape[0], y2 + margin)
                
                # Extraire la région de la case
                roi = gray[y:y2, x:x2] if 0 <= y < y2 <= gray.shape[0] and 0 <= x < x2 <= gray.shape[1] else None
                
                # Déterminer si cochée avec la méthode améliorée
                if roi is not None and roi.size > 0:
                    is_checked = self._is_checkbox_checked(roi)
                    checkbox["checked"] = is_checked
                else:
                    checkbox["checked"] = False
        
        # 5. Associer les étiquettes aux cases avec la méthode améliorée
        for checkbox in merged_checkboxes:
            if not checkbox.get("label"):
                bbox = checkbox.get("bbox", [0, 0, 0, 0])
                label = self._find_closest_text(page_texts, bbox)
                checkbox["label"] = label
        
        # 6. Détecter et normaliser les groupes Oui/Non
        self._normalize_yes_no_groups(merged_checkboxes)
        
        # 7. Filtrer par seuil de confiance
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
    
    def _normalize_yes_no_groups(self, checkboxes: List[Dict[str, Any]]) -> None:
        """
        Identifie et normalise les groupes de cases à cocher Oui/Non.
        
        Args:
            checkboxes: Liste des cases à cocher à normaliser in-place
        """
        # Regrouper les cases proches en position y (même ligne ou lignes adjacentes)
        y_groups = {}
        
        for i, checkbox in enumerate(checkboxes):
            bbox = checkbox.get("bbox", [0, 0, 0, 0])
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Arrondir la position y pour regrouper les cases sur la même ligne approximativement
            group_y = round(center_y / 20) * 20
            
            if group_y not in y_groups:
                y_groups[group_y] = []
            
            y_groups[group_y].append(i)
        
        # Pour chaque groupe en y, identifier les paires Oui/Non
        for group_indices in y_groups.values():
            if len(group_indices) < 2:
                continue
                
            yes_indices = []
            no_indices = []
            
            # Identifier les cases Oui et Non
            for idx in group_indices:
                label = checkboxes[idx].get("label", "").strip().lower()
                
                if re.match(r'^oui$|^yes$', label):
                    yes_indices.append(idx)
                    checkboxes[idx]["label"] = "Oui"
                    checkboxes[idx]["value"] = "Oui"
                elif re.match(r'^non$|^no$', label):
                    no_indices.append(idx)
                    checkboxes[idx]["label"] = "Non"
                    checkboxes[idx]["value"] = "Non"
            
            # Si nous avons au moins un Oui et un Non, les marquer comme faisant partie du même groupe
            if yes_indices and no_indices:
                group_id = f"group_yes_no_{min(group_indices)}"
                
                for idx in yes_indices + no_indices:
                    checkboxes[idx]["group_id"] = group_id

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
        """
        Fusionne les résultats des deux méthodes de détection de manière plus intelligente.
        
        Args:
            symbol_checkboxes: Cases détectées par symboles
            visual_checkboxes: Cases détectées visuellement
            
        Returns:
            Liste fusionnée de cases à cocher
        """
        if not symbol_checkboxes:
            return visual_checkboxes
        if not visual_checkboxes:
            return symbol_checkboxes
        
        # Donner priorité aux cases détectées par symboles (généralement plus précises)
        merged = symbol_checkboxes.copy()
        symbol_boxes = [cb["bbox"] for cb in symbol_checkboxes]
        
        # Pour chaque case détectée visuellement
        for visual_cb in visual_checkboxes:
            vx1, vy1, vx2, vy2 = visual_cb["bbox"]
            v_center_x, v_center_y = (vx1 + vx2) / 2, (vy1 + vy2) / 2
            
            # Vérifier si elle chevauche une case déjà détectée par symbole
            is_duplicate = False
            
            for i, s_bbox in enumerate(symbol_boxes):
                sx1, sy1, sx2, sy2 = s_bbox
                s_center_x, s_center_y = (sx1 + sx2) / 2, (sy1 + sy2) / 2
                
                # Calcul de chevauchement basé sur la distance entre centres
                center_distance = ((v_center_x - s_center_x) ** 2 + (v_center_y - s_center_y) ** 2) ** 0.5
                
                # Si les centres sont proches, c'est probablement la même case
                max_dimension = max(vx2 - vx1, vy2 - vy1, sx2 - sx1, sy2 - sy1)
                if center_distance < max_dimension * 0.7:  # Distance inférieure à 70% de la plus grande dimension
                    is_duplicate = True
                    
                    # Mettre à jour les informations si la détection visuelle a une meilleure confiance
                    if visual_cb.get("confidence", 0) > symbol_checkboxes[i].get("confidence", 0) + 0.15:
                        # Garder les attributs importants de la version symbolique
                        label = symbol_checkboxes[i].get("label", "")
                        if label:
                            visual_cb["label"] = label
                        
                        # Remplacer la case détectée par symbole par la version visuelle
                        merged[i] = visual_cb
                    
                    break
            
            # Si ce n'est pas un doublon, l'ajouter
            if not is_duplicate:
                merged.append(visual_cb)
        
        # Déduplication supplémentaire basée sur la proximité des cases
        # (pour gérer les cas où deux méthodes détectent la même case avec un léger décalage)
        final_merged = []
        used_indices = set()
        
        for i, checkbox1 in enumerate(merged):
            if i in used_indices:
                continue
            
            bbox1 = checkbox1["bbox"]
            center1_x = (bbox1[0] + bbox1[2]) / 2
            center1_y = (bbox1[1] + bbox1[3]) / 2
            
            duplicates = [i]
            
            for j, checkbox2 in enumerate(merged[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                bbox2 = checkbox2["bbox"]
                center2_x = (bbox2[0] + bbox2[2]) / 2
                center2_y = (bbox2[1] + bbox2[3]) / 2
                
                # Calculer la distance entre les centres
                distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
                
                # Si les centres sont très proches, c'est probablement la même case
                if distance < 20:  # 20 pixels de seuil
                    duplicates.append(j)
            
            # Prendre la case avec la meilleure confiance
            best_idx = max(duplicates, key=lambda idx: merged[idx].get("confidence", 0))
            final_merged.append(merged[best_idx])
            
            # Marquer tous les doublons comme utilisés
            used_indices.update(duplicates)
        
        return final_merged
    
    def _is_checkbox_checked(self, checkbox_img: np.ndarray) -> bool:
        """
        Détermine si une case à cocher est cochée avec une analyse d'image améliorée.
        
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
            thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 2. Calculer différentes métriques pour déterminer si la case est cochée
            
            # Pourcentage de pixels non-blancs (marques)
            total_pixels = binary.size
            marked_pixels = np.sum(binary > 0)
            fill_ratio = marked_pixels / total_pixels if total_pixels > 0 else 0
            
            # 3. Appliquer une analyse plus sophistiquée pour les cases à cocher
            
            # Vérifier la présence de lignes diagonales (typiques d'une croix)
            height, width = binary.shape
            has_diagonals = False
            
            # Utiliser la transformée de Hough pour détecter les lignes
            if width > 10 and height > 10:  # Éviter les cases trop petites
                edges = cv2.Canny(binary, 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                    threshold=max(5, min(width, height)//3), 
                                    minLineLength=max(3, min(width, height)//4), 
                                    maxLineGap=3)
                
                if lines is not None and len(lines) > 0:
                    diag_count = 0
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        # Calculer l'angle de la ligne
                        if x2 != x1:  # Éviter division par zéro
                            angle = abs(np.arctan((y2-y1)/(x2-x1)) * 180 / np.pi)
                            # Les diagonales ont des angles entre 30 et 60 degrés
                            if 30 <= angle <= 60 or 120 <= angle <= 150:
                                diag_count += 1
                    
                    has_diagonals = diag_count >= 2  # Au moins 2 lignes diagonales
            
            # 4. Analyse de la distribution spatiale des marques (concentrées au centre ou sur les bords)
            center_margin = max(2, min(width, height) // 4)
            border_region = np.zeros_like(binary)
            center_region = np.zeros_like(binary)
            
            if width > 2*center_margin and height > 2*center_margin:
                # Définir les régions d'intérêt
                center_region[center_margin:height-center_margin, center_margin:width-center_margin] = 1
                border_region = 1 - center_region
                
                # Calculer les ratios
                center_marked = np.sum(binary * center_region)
                border_marked = np.sum(binary * border_region)
                
                center_total = np.sum(center_region)
                border_total = np.sum(border_region)
                
                center_ratio = center_marked / center_total if center_total > 0 else 0
                border_ratio = border_marked / border_total if border_total > 0 else 0
                
                # Caractéristiques d'une case cochée: remplissage au centre OU présence de diagonales
                if has_diagonals:
                    # La présence de diagonales est un fort indicateur
                    return True
                elif center_ratio > 0.2:
                    # Remplissage significatif au centre
                    return True
                elif fill_ratio > 0.25:
                    # Remplissage global élevé
                    return True
                else:
                    return False
            else:
                # Pour les petites cases, utiliser juste le ratio global et la présence de diagonales
                return has_diagonals or fill_ratio > 0.22
        
        except Exception as e:
            logger.debug(f"Erreur analyse case cochée: {e}")
            return False
    
    def _group_checkboxes_by_question(self, checkboxes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Regroupe les cases à cocher par questions ou sections logiques.
        
        Args:
            checkboxes: Liste des cases à cocher détectées
            
        Returns:
            Dictionnaire groupant les cases par question
        """
        # Trier les cases par page puis par position y
        sorted_checkboxes = sorted(checkboxes, key=lambda cb: (cb.get("page", 0), cb.get("bbox", [0, 0, 0, 0])[1]))
        
        # Initialiser les groupes
        groups = {}
        current_group = None
        group_id = 0
        
        # Pour garder trace de la dernière position y
        last_y = None
        last_page = None
        
        # Seuil de distance verticale pour considérer des cases comme faisant partie d'une même question
        y_threshold = 50  # en pixels
        
        for checkbox in sorted_checkboxes:
            page = checkbox.get("page", 0)
            bbox = checkbox.get("bbox", [0, 0, 0, 0])
            y = bbox[1]  # Position y du haut de la case
            label = checkbox.get("label", "").strip()
            
            # Nouvelle page = nouveau groupe
            if page != last_page:
                current_group = None
            
            # Déterminer si cette case fait partie du groupe courant ou commence un nouveau groupe
            if current_group is None or (abs(y - last_y) > y_threshold if last_y is not None else True):
                # Nouveau groupe
                group_id += 1
                current_group = f"question_{group_id}"
                groups[current_group] = []
            
            # Ajouter la case au groupe courant
            groups[current_group].append(checkbox)
            
            # Mettre à jour last_y et last_page
            last_y = y
            last_page = page
        
        # Normaliser les groupes pour un usage pratique
        normalized_groups = {}
        
        for group_id, checkboxes in groups.items():
            # Identifier les groupes Oui/Non
            yes_no_labels = [cb for cb in checkboxes if re.search(r'\b(oui|non|yes|no)\b', cb.get("label", "").lower())]
            
            if len(yes_no_labels) >= 2:
                # C'est probablement un groupe de type question avec options Oui/Non
                question_text = self._extract_question_from_group(checkboxes)
                if question_text:
                    normalized_groups[question_text] = checkboxes
                else:
                    normalized_groups[group_id] = checkboxes
            else:
                # Groupe standard
                normalized_groups[group_id] = checkboxes
        
        return normalized_groups
    
    def _extract_question_from_group(self, checkboxes: List[Dict[str, Any]]) -> str:
        """
        Tente d'extraire le texte de la question à partir d'un groupe de cases à cocher.
        
        Args:
            checkboxes: Liste des cases à cocher du groupe
            
        Returns:
            Texte de la question ou chaîne vide
        """
        # Filtrer les étiquettes qui sont juste "Oui" ou "Non"
        non_yes_no_labels = []
        
        for checkbox in checkboxes:
            label = checkbox.get("label", "").strip().lower()
            if label and not re.match(r'^(oui|non|yes|no)$', label):
                non_yes_no_labels.append(checkbox.get("label", ""))
        
        # Si nous avons trouvé des labels qui ne sont pas juste Oui/Non
        if non_yes_no_labels:
            # Prendre le plus long comme probable question
            return max(non_yes_no_labels, key=len)
        
        return ""

    def _find_closest_text(self, page_text: Dict, bbox: List[int]) -> str:
        """
        Trouve le texte le plus proche d'une case à cocher avec des améliorations avancées.
        
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
        max_distance = 150  # Distance maximale à considérer
        
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
                            "coords": [line_center_x, line_center_y],
                            "bbox": line_bbox
                        })

        # Analyser les candidats en considérant leur position relative
        for candidate in candidates:
            distance = candidate["distance"]
            x, y = candidate["coords"]
            candidate_bbox = candidate["bbox"]
            
            # Facteurs de pondération selon la position relative
            position_factor = 1.0
            
            # Mesure de l'alignement horizontal/vertical avec la case
            h_aligned = abs(y - checkbox_center_y) < 15  # Alignement horizontal
            v_aligned = abs(x - checkbox_center_x) < 15  # Alignement vertical
            
            # Privilégier les textes correctement alignés
            if h_aligned and x > checkbox_center_x:
                # Texte à droite sur la même ligne (cas le plus commun)
                position_factor = 0.5
            elif h_aligned and x < checkbox_center_x:
                # Texte à gauche sur la même ligne (cas également commun)
                position_factor = 0.65
            elif v_aligned and y < checkbox_center_y:
                # Texte au-dessus, aligné verticalement (parfois le cas)
                position_factor = 0.8
            elif v_aligned and y > checkbox_center_y:
                # Texte en-dessous, aligné verticalement
                position_factor = 0.85
            else:
                # Positions moins probables
                position_factor = 1.3
            
            # Vérifier la taille du texte (préférer les courts textes comme "Oui"/"Non")
            text = candidate["text"]
            
            # Privilégier fortement les "Oui" et "Non" alignés horizontalement
            if h_aligned and re.search(r'\b(oui|non|yes|no)\b', text.lower()):
                position_factor *= 0.5
                
            # Limiter la longueur des étiquettes (pas de phrases entières)
            if len(text) > 80:
                # Tronquer les textes trop longs
                text = text[:77] + "..."
                position_factor *= 1.5
            
            # Rejeter les étiquettes qui ressemblent à des informations de contact/codes
            if re.search(r'(\d{2}\s*){3,}|@|www\.|\d+\.\d+\.\d+|\/\d+', text):
                position_factor *= 2.0
            
            # Appliquer la pondération
            adjusted_distance = distance * position_factor
            
            # Mettre à jour le texte le plus proche
            if adjusted_distance < min_distance:
                min_distance = adjusted_distance
                closest_text = text
        
        # Nettoyage final de l'étiquette
        if closest_text:
            # Supprimer les ponctuations en fin de texte
            closest_text = re.sub(r'[:\.\?;,]+$', '', closest_text)
            
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
        Organise les cases à cocher par sections et crée des groupes logiques.
        
        Args:
            results: Résultats de l'extraction à modifier in-place
        """
        checkboxes = results.get("checkboxes", [])
        
        # 1. Nettoyer et normaliser les étiquettes
        for checkbox in checkboxes:
            label = checkbox.get("label", "").strip()
            
            # Normaliser les étiquettes Oui/Non
            if re.match(r'^oui$|^yes$', label, re.IGNORECASE):
                checkbox["label"] = "Oui"
                checkbox["value"] = "Oui"
            elif re.match(r'^non$|^no$', label, re.IGNORECASE):
                checkbox["label"] = "Non"
                checkbox["value"] = "Non"
        
        # 2. Regrouper les cases à cocher par questions
        groups = self._group_checkboxes_by_question(checkboxes)
        
        # 3. Créer une structure plus intuitive pour les résultats
        structured_results = {
            "groups": {},
            "form_values": {}
        }
        
        # Traiter chaque groupe
        for group_id, group_checkboxes in groups.items():
            group_info = {
                "checkboxes": group_checkboxes,
                "values": {}
            }
            
            # Déterminer s'il s'agit d'une question Oui/Non
            yes_no_boxes = [cb for cb in group_checkboxes if cb.get("label") in ["Oui", "Non"]]
            
            if len(yes_no_boxes) >= 2:
                # Traiter comme une question Oui/Non
                question = group_id if isinstance(group_id, str) and not group_id.startswith("question_") else "Question"
                
                checked_values = [cb.get("label") for cb in yes_no_boxes if cb.get("checked", False)]
                if checked_values:
                    group_info["values"][question] = checked_values[0]
                    structured_results["form_values"][question] = checked_values[0]
            else:
                # Traiter comme cases à cocher individuelles
                for checkbox in group_checkboxes:
                    label = checkbox.get("label", "")
                    if label and label not in ["Oui", "Non"]:
                        is_checked = checkbox.get("checked", False)
                        group_info["values"][label] = "Oui" if is_checked else "Non"
                        if is_checked:
                            structured_results["form_values"][label] = "Oui"
            
            # Ajouter le groupe aux résultats
            structured_results["groups"][str(group_id)] = group_info
        
        # 4. Ajouter la structure au résultat final
        results["structured_checkboxes"] = structured_results
        
        # 5. Extraire et mettre à jour les valeurs de formulaire
        results["form_values"] = structured_results["form_values"]