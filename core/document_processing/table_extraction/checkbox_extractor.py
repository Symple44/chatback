# core/document_processing/table_extraction/checkbox_extractor
"""
Module d'extraction des cases à cocher dans les PDFs.
Permet de détecter l'état des cases à cocher et d'associer chaque étiquette à sa valeur.
"""
from typing import Dict, List, Tuple, Any, Optional, Union
import fitz  # PyMuPDF
import numpy as np
import cv2
from pathlib import Path
import re
import json
import os
import io
import logging
from datetime import datetime

from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger("checkbox_extractor")

class CheckboxExtractor:
    """
    Classe spécialisée dans l'extraction des cases à cocher dans les documents PDF.
    Détecte les cases à cocher, détermine leur état (cochée ou non) et les associe à leur étiquette.
    """

    def __init__(self):
        """
        Initialise l'extracteur de cases à cocher.
        """
        # Configuration des paramètres de détection
        self.checkbox_min_size = 8     # Taille minimale d'une case à cocher en pixels
        self.checkbox_max_size = 25    # Taille maximale d'une case à cocher en pixels
        self.label_distance = 100      # Distance maximale entre une case et son étiquette
        self.checkbox_tolerance = 0.2  # Tolérance pour la taille carrée d'une case
        self.min_checkbox_area = 64    # Surface minimale d'une case en pixels carrés
        self.checked_threshold = 0.15  # Seuil pour déterminer si une case est cochée
        self.dpi = 300                 # Résolution pour la conversion en image
        self.cache = {}                # Cache des résultats (pour éviter de refaire l'analyse)

    async def _detect_checkboxes_by_symbols(self, page, text_dict, page_img, page_num):
        """
        Détecte les cases à cocher basées sur les symboles spécifiques dans le texte.
        
        Args:
            page: Page PDF (fitz page)
            text_dict: Dictionnaire de texte extrait
            page_img: Image de la page convertie en array numpy
            page_num: Numéro de la page
            
        Returns:
            Liste des cases à cocher détectées
        """
        checkboxes = []
        
        # Recherche de motifs typiques des cases à cocher dans le texte
        checkbox_symbols = ["☐", "☑", "☒", "□", "■", "▢", "▣", "▪", "▫"]
        checkbox_keywords = ["oui", "non", "yes", "no", "cochez", "check"]
        
        # Parcourir les blocs de texte
        for block in text_dict["blocks"]:
            if block["type"] != 0:  # Ignorer les non-texte
                continue
                
            for line in block["lines"]:
                line_text = " ".join([span["text"] for span in line["spans"]])
                
                # Vérifier si la ligne contient un symbole de case à cocher
                has_symbol = any(symbol in line_text for symbol in checkbox_symbols)
                has_keyword = any(keyword in line_text.lower() for keyword in checkbox_keywords)
                
                if has_symbol or has_keyword:
                    # Extraire la région de la ligne
                    x0, y0, x1, y1 = line["bbox"]
                    
                    # Déterminer si la case est cochée (par le contenu du texte pour l'instant)
                    is_checked = any(checked in line_text for checked in ["☑", "☒", "■"])
                    
                    # Extraire l'étiquette (texte sans les symboles)
                    label = line_text
                    for symbol in checkbox_symbols:
                        label = label.replace(symbol, "")
                    label = label.strip()
                    
                    # Créer l'élément case à cocher
                    checkbox = {
                        "label": label,
                        "checked": is_checked,
                        "bbox": [x0, y0, x1, y1],
                        "page": page_num,
                        "type": "symbol",
                        "confidence": 0.8 if has_symbol else 0.6
                    }
                    
                    checkboxes.append(checkbox)
        
        return checkboxes

    async def extract_checkboxes_from_pdf(
        self, 
        pdf_path: Union[str, Path, io.BytesIO],
        page_range: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None  # Ajout du paramètre config
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extrait les cases à cocher d'un document PDF.
        
        Args:
            pdf_path: Chemin ou objet BytesIO du fichier PDF
            page_range: Liste optionnelle des pages à analyser (1-based)
            config: Configuration optionnelle pour l'extraction
            
        Returns:
            Dictionnaire des résultats avec les sections et leurs cases à cocher
        """
        try:
            with metrics.timer("checkbox_extraction"):
                # Utiliser la configuration si fournie
                conf = config or {}
                
                # Calcul d'un hash de cache simple
                if isinstance(pdf_path, (str, Path)):
                    cache_key = f"{pdf_path}:{str(page_range)}"
                    if cache_key in self.cache:
                        return self.cache[cache_key]
                
                # Ouvrir le document PDF - CORRECTION ICI
                if isinstance(pdf_path, (str, Path)):
                    pdf_doc = fitz.open(pdf_path)
                else:
                    # Pour BytesIO, on doit réinitialiser la position et l'ouvrir comme stream
                    pdf_path.seek(0)
                    pdf_doc = fitz.open(stream=pdf_path.read(), filetype="pdf")
                
                # Si page_range n'est pas spécifié, analyser tout le document
                if page_range is None:
                    page_range = list(range(len(pdf_doc)))
                else:
                    # Convertir page_range de 1-based à 0-based
                    page_range = [p-1 for p in page_range if 0 <= p-1 < len(pdf_doc)]
                
                results = {
                    "metadata": {
                        "filename": getattr(pdf_path, "name", str(pdf_path)) if hasattr(pdf_path, "name") else "unknown",
                        "page_count": len(pdf_doc),
                        "processed_pages": len(page_range),
                        "extraction_date": datetime.now().isoformat()
                    },
                    "sections": {},
                    "checkboxes": []
                }
                
                # Paramètres de config
                confidence_threshold = conf.get("confidence_threshold", 0.6)
                enhance_detection = conf.get("enhance_detection", True)
                
                # Traiter chaque page
                for page_idx in page_range:
                    page = pdf_doc[page_idx]
                    
                    # Étape 1: Extraire le texte de la page pour les étiquettes
                    page_text = page.get_text("dict")
                    
                    # Étape 2: Convertir la page en image pour la détection visuelle
                    pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi/72, self.dpi/72))
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    
                    # Convertir en niveaux de gris si l'image est en couleur
                    if img.shape[2] == 3 or img.shape[2] == 4:
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img
                    
                    # Étape 3: Détecter les cases à cocher
                    checkbox_regions = self._detect_checkboxes(gray)
                    
                    # Étape 4: Identifier les étiquettes et déterminer l'état des cases
                    page_results = self._match_labels_to_checkboxes(
                        checkbox_regions, page_text, gray, page_idx + 1
                    )
                    
                    # Organiser par section
                    for item in page_results:
                        section = item.get("section", "default")
                        if section not in results["sections"]:
                            results["sections"][section] = []
                        
                        # Ajouter à la section
                        results["sections"][section].append(item)
                        
                        # Ajouter à la liste complète
                        results["checkboxes"].append(item)
                
                # Fermer le document
                pdf_doc.close()
                
                # Mise en cache du résultat
                if isinstance(pdf_path, (str, Path)):
                    self.cache[cache_key] = results
                
                return results
                
        except Exception as e:
            logger.error(f"Erreur extraction cases à cocher: {e}")
            metrics.increment_counter("checkbox_extraction_errors")
            return {
                "error": str(e),
                "sections": {},
                "checkboxes": []
            }
        
    def _detect_checkboxes(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Détecte les cases à cocher dans une image avec des algorithmes améliorés.
        
        Args:
            img: Image en niveaux de gris
            
        Returns:
            Liste des régions de cases à cocher détectées
        """
        try:
            binary_methods = [
                # Méthode 1: Binarisation Otsu standard
                lambda x: cv2.threshold(x, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
                
                # Méthode 2: Binarisation adaptative (plus sensible aux détails)
                lambda x: cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY_INV, 11, 2),
                                                
                # Méthode 3: Flou puis Otsu (plus robuste au bruit)
                lambda x: cv2.threshold(cv2.GaussianBlur(x, (5, 5), 0), 
                                        0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            ]
            
            # NOUVEL AJOUT: détection multi-échelle
            scales = [0.8, 1.0, 1.2]  # Différentes échelles pour détecter des cases de tailles variables
            
            all_checkbox_regions = []
            
            # Pour chaque échelle
            for scale in scales:
                # Redimensionner l'image
                if scale != 1.0:
                    height, width = img.shape[:2]
                    resized = cv2.resize(img, (int(width * scale), int(height * scale)))
                else:
                    resized = img
                    
                # Appliquer chaque méthode de binarisation
                for method in binary_methods:
                    binary = method(resized)
                    
                    # AMÉLIORATION: détection forme-spécifique pour cases à cocher
                    # 1. Détection de formes carrées (checkboxes traditionnelles)
                    square_regions = self._detect_square_checkboxes(binary, scale)
                    all_checkbox_regions.extend(square_regions)
                    
                    # 2. Détection de formes circulaires (radio buttons)
                    circle_regions = self._detect_circle_checkboxes(binary, scale)
                    all_checkbox_regions.extend(circle_regions)
            
            # Filtrer les doublons et valider les détections
            return self._filter_checkbox_candidates(all_checkbox_regions)
            
        except Exception as e:
            logger.error(f"Erreur détection cases à cocher: {e}")
            return []
    
    def _detect_checkbox_groups(self, checkbox_regions: List[Dict[str, Any]], page_text: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Détecte les groupes de cases à cocher (notamment les paires Oui/Non).
        
        Args:
            checkbox_regions: Régions des cases à cocher détectées
            page_text: Texte de la page au format dict de PyMuPDF
            
        Returns:
            Dictionnaire de groupes de cases à cocher
        """
        # Extraire les blocs de texte
        blocks = page_text.get("blocks", [])
        
        # Extraire tous les spans de texte avec leurs coordonnées
        oui_non_spans = []
        for block in blocks:
            if block["type"] == 0:  # Type 0 = text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip().lower()
                        if span_text in ["oui", "non", "yes", "no"]:
                            # Calculer le centre du span
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            span_center_x = (bbox[0] + bbox[2]) / 2
                            span_center_y = (bbox[1] + bbox[3]) / 2
                            
                            oui_non_spans.append({
                                "text": span_text,
                                "center_x": span_center_x,
                                "center_y": span_center_y,
                                "bbox": bbox
                            })
        
        # Organiser les checkboxes par groupes
        groups = {}
        
        # Associer chaque case à cocher avec le texte Oui/Non le plus proche
        for i, region in enumerate(checkbox_regions):
            # Coordonnées du centre de la case
            checkbox_center_x = region.get("center_x", region["x"] + region["width"] // 2)
            checkbox_center_y = region.get("center_y", region["y"] + region["height"] // 2)
            
            # Trouver le texte Oui/Non le plus proche
            closest_oui_non = None
            min_distance = float('inf')
            
            for span in oui_non_spans:
                distance = np.sqrt(
                    (span["center_x"] - checkbox_center_x)**2 + 
                    (span["center_y"] - checkbox_center_y)**2
                )
                
                # Limiter la distance à 100 pixels (ajustable)
                if distance < 100 and distance < min_distance:
                    min_distance = distance
                    closest_oui_non = span
            
            if closest_oui_non:
                # Déterminer le type d'étiquette (Oui/Non)
                label_type = closest_oui_non["text"]
                
                # Ajouter à un groupe basé sur la position Y pour regrouper les cases sur la même ligne
                y_key = int(checkbox_center_y / 10) * 10  # Regrouper par tranches de 10 pixels
                
                if y_key not in groups:
                    groups[y_key] = []
                    
                groups[y_key].append({
                    "region": region,
                    "label_type": label_type,
                    "distance": min_distance
                })
        
        return groups

    def _is_checkbox_checked(self, img: np.ndarray, region: Dict[str, Any]) -> bool:
        """
        Détermine si une case à cocher est cochée avec des méthodes optimisées pour les formulaires techniques.
        
        Cette version améliorée utilise plusieurs techniques complémentaires:
        - Analyse précise du contenu intérieur de la case
        - Détection de motifs de type X ou ✓
        - Analyse de la densité du centre de la case
        - Différentes méthodes de binarisation pour une meilleure détection
        
        Args:
            img: Image en niveaux de gris
            region: Région de la case à cocher
            
        Returns:
            True si la case est cochée, False sinon
        """
        try:
            # Extraire la région de la case avec marge
            x = region["x_with_margin"] if "x_with_margin" in region else region["x"]
            y = region["y_with_margin"] if "y_with_margin" in region else region["y"]
            w = region["width_with_margin"] if "width_with_margin" in region else region["width"]
            h = region["height_with_margin"] if "height_with_margin" in region else region["height"]
            
            # S'assurer que les limites sont valides
            if y + h > img.shape[0] or x + w > img.shape[1]:
                return False
            
            # Extraire la région de la case
            checkbox_img = img[y:y+h, x:x+w]
            
            # 1. Analyse du contenu intérieur (évite de prendre les bordures)
            inner_margin = max(2, min(w, h) // 10)  # Marge adaptative selon la taille
            if w > 2*inner_margin and h > 2*inner_margin:
                # Créer un masque pour extraire uniquement la partie intérieure
                inner_mask = np.zeros_like(checkbox_img)
                inner_mask[inner_margin:-inner_margin, inner_margin:-inner_margin] = 1
                
                # Appliquer différentes méthodes de binarisation pour plus de robustesse
                # Méthode 1: Seuillage Otsu
                _, binary_otsu = cv2.threshold(checkbox_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Méthode 2: Seuillage adaptatif - meilleur pour les documents numérisés
                binary_adaptive = cv2.adaptiveThreshold(
                    checkbox_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
                )
                
                # Appliquer le masque intérieur aux deux versions binaires
                inner_binary_otsu = binary_otsu * inner_mask
                inner_binary_adaptive = binary_adaptive * inner_mask
                
                # Calculer les ratios de pixels blancs (potentiellement cochés)
                inner_area = np.sum(inner_mask)
                if inner_area > 0:
                    ratio_otsu = np.sum(inner_binary_otsu > 0) / inner_area
                    ratio_adaptive = np.sum(inner_binary_adaptive > 0) / inner_area
                    
                    # Utiliser le ratio le plus élevé des deux méthodes
                    inner_ratio = max(ratio_otsu, ratio_adaptive)
                    
                    # Utiliser un seuil adaptatif basé sur la taille de la case
                    adaptive_threshold = 0.15 if inner_area > 100 else 0.2
                    
                    if inner_ratio > adaptive_threshold:
                        return True
                
                # 2. Analyse spécifique du centre (où se trouve généralement la marque)
                center_margin = max(inner_margin * 2, min(w, h) // 4)
                if w > 2*center_margin and h > 2*center_margin:
                    # Créer un masque pour le centre uniquement
                    center_mask = np.zeros_like(checkbox_img)
                    center_mask[center_margin:-center_margin, center_margin:-center_margin] = 1
                    
                    # Appliquer aux deux versions
                    center_binary_otsu = binary_otsu * center_mask
                    center_binary_adaptive = binary_adaptive * center_mask
                    
                    # Calculer les ratios
                    center_area = np.sum(center_mask)
                    if center_area > 0:
                        center_ratio_otsu = np.sum(center_binary_otsu > 0) / center_area
                        center_ratio_adaptive = np.sum(center_binary_adaptive > 0) / center_area
                        
                        # Prendre le ratio le plus élevé
                        center_ratio = max(center_ratio_otsu, center_ratio_adaptive)
                        
                        # Une case est probablement cochée si le ratio central est élevé
                        if center_ratio > 0.15:  # Seuil abaissé pour être plus sensible
                            return True
                
                # 3. Recherche d'éléments de type "X" ou "✓" avec des opérations morphologiques
                # Cela permet de détecter des marques spécifiques même avec un faible contraste
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(inner_binary_otsu, kernel, iterations=1)
                dilated = cv2.dilate(eroded, kernel, iterations=1)
                
                # Si après érosion et dilatation on garde une structure significative
                # c'est probablement une marque comme X ou ✓ plutôt que du bruit
                structure_ratio = np.sum(dilated > 0) / inner_area if inner_area > 0 else 0
                
                if structure_ratio > 0.08:  # Seuil bas pour détecter même les marques légères
                    return True
                    
                # 4. Analyse de l'histogramme pour détecter les cas ambigus
                # Un histogramme bimodal peut indiquer une case cochée même avec un faible contraste
                hist = cv2.calcHist([checkbox_img], [0], inner_mask.astype(np.uint8), [256], [0, 256])
                hist_normalized = hist / np.sum(hist)
                
                # Calculer le nombre de pics significatifs dans l'histogramme
                peaks, _ = find_peaks(hist_normalized.flatten(), height=0.01, distance=20)
                if len(peaks) >= 2:  # Histogramme bimodal = probable marque
                    # Vérifier si les pics sont suffisamment éloignés (contraste)
                    if np.max(peaks) - np.min(peaks) > 50:
                        return True
            
            # 5. Analyse globale comme fallback (pour les très petites cases)
            # Utiliser Otsu qui est plus robuste pour la détection globale
            _, binary = cv2.threshold(checkbox_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            total_pixels = checkbox_img.size
            if total_pixels > 0:
                global_ratio = np.sum(binary > 0) / total_pixels
                
                # Seuil adaptatif selon la taille de la case
                adaptive_global_threshold = 0.15 if total_pixels > 400 else 0.20
                
                if global_ratio > adaptive_global_threshold:
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Erreur analyse case cochée: {e}")
            # En cas d'erreur, considérer comme non cochée
            return False
    
    # Fonction utilitaire pour l'analyse d'histogramme - à importer de scipy si disponible
    def find_peaks(x, height=None, threshold=None, distance=None):
        """
        Version simplifiée de scipy.signal.find_peaks pour détecter les pics dans un histogramme.
        """
        results = []
        if len(x) < 3:
            return np.array(results), {}
        
        # Trouver les maxima locaux
        for i in range(1, len(x)-1):
            if x[i-1] < x[i] and x[i] > x[i+1]:
                # Vérifier le seuil minimal
                if height is None or x[i] >= height:
                    results.append(i)
        
        # Filtrer par distance si spécifiée
        if distance is not None and len(results) > 1:
            filtered = [results[0]]
            for i in range(1, len(results)):
                if results[i] - filtered[-1] >= distance:
                    filtered.append(results[i])
            results = filtered
        
        return np.array(results), {"peak_heights": [x[i] for i in results]}

    def _match_labels_to_checkboxes(
        self, 
        checkbox_regions: List[Dict[str, Any]], 
        page_text: Dict[str, Any],
        img: np.ndarray,
        page_num: int
    ) -> List[Dict[str, Any]]:
        """
        Associe les étiquettes aux cases à cocher et détermine leur état.
        
        Args:
            checkbox_regions: Régions des cases à cocher détectées
            page_text: Texte de la page au format dict de PyMuPDF
            img: Image en niveaux de gris
            page_num: Numéro de la page
            
        Returns:
            Liste des cases à cocher avec leurs étiquettes et états
        """
        results = []
        
        # Extraction améliorée du texte de la page
        text_blocks = self._extract_text_blocks(page_text)
        
        # Identification avancée des groupes de cases à cocher
        checkbox_groups = self._identify_checkbox_groups(checkbox_regions, text_blocks)
        
        # Pour chaque case ou groupe
        for checkbox in checkbox_regions:
            # Méthode 1: Recherche spatiale améliorée
            nearest_labels = self._find_nearest_labels(checkbox, text_blocks)
            
            # Méthode 2: Analyse contextuelle (recherche d'étiquettes type Oui/Non, etc.)
            context_labels = self._analyze_label_context(checkbox, text_blocks)
            
            # Méthode 3: Analyse des alignements (les étiquettes sont souvent alignées verticalement/horizontalement)
            aligned_labels = self._find_aligned_labels(checkbox, text_blocks)
            
            # Fusionner les résultats des trois méthodes avec pondération
            best_label = self._select_best_label(nearest_labels, context_labels, aligned_labels)
            
            # Déterminer si la case est cochée (avec méthode améliorée)
            is_checked = self._is_checkbox_checked_enhanced(img, checkbox)
            
            # Ajout du résultat avec métadonnées enrichies
            results.append({
                "label": best_label["text"],
                "value": best_label.get("value", ""),
                "checked": is_checked,
                "section": best_label.get("section", "Information"),
                "page": page_num,
                "position": self._get_position_data(checkbox),
                "confidence": self._calculate_confidence(checkbox, best_label, is_checked),
                "group_id": checkbox.get("group_id"),
                "type": checkbox.get("type", "checkbox")  # checkbox ou radio
            })
        
        return results
    
    def format_to_table(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Formate les résultats en tableau pour faciliter l'export.
        
        Args:
            results: Résultats de l'extraction
            
        Returns:
            Liste formatée pour export en tableau
        """
        table_data = []
        
        # Parcourir chaque section
        for section_name, checkboxes in results.get("sections", {}).items():
            for checkbox in checkboxes:
                if checkbox.get("checked"):
                    table_data.append({
                        "Section": section_name,
                        "Label": checkbox.get("label", ""),
                        "Valeur": checkbox.get("value", ""),
                        "Page": checkbox.get("page", "")
                    })
        
        return table_data
    
    def format_to_text(self, results: Dict[str, Any], include_unchecked: bool = False) -> str:
        """
        Formate les résultats en texte.
        
        Args:
            results: Résultats de l'extraction
            include_unchecked: Si True, inclut aussi les cases non cochées
            
        Returns:
            Texte formaté
        """
        output = []
        
        # Ajouter les métadonnées
        output.append("# Résultats d'extraction des cases à cocher")
        output.append(f"Document: {results.get('metadata', {}).get('filename', 'inconnu')}")
        output.append(f"Date d'extraction: {results.get('metadata', {}).get('extraction_date', '')}")
        output.append(f"Pages analysées: {results.get('metadata', {}).get('processed_pages', 0)}")
        output.append("")
        
        # Parcourir les sections
        for section_name, checkboxes in results.get("sections", {}).items():
            output.append(f"## Section: {section_name}")
            
            section_items = []
            for checkbox in checkboxes:
                if checkbox.get("checked") or include_unchecked:
                    label = checkbox.get("label", "")
                    value = checkbox.get("value", "")
                    section_items.append(f"{label}: {value}")
            
            if section_items:
                output.extend(section_items)
                output.append("")
            else:
                output.append("Aucun élément trouvé\n")
        
        return "\n".join(output)
    
    def _is_checkbox_checked_enhanced(self, img: np.ndarray, region: Dict[str, Any]) -> bool:
        """
        Détermine si une case à cocher est cochée avec une approche multi-techniques.
        
        Args:
            img: Image en niveaux de gris
            region: Région de la case à cocher
            
        Returns:
            True si la case est cochée, False sinon
        """
        try:
            # Extraire la région de la case avec marge
            x = region["x_with_margin"] if "x_with_margin" in region else region["x"]
            y = region["y_with_margin"] if "y_with_margin" in region else region["y"]
            w = region["width_with_margin"] if "width_with_margin" in region else region["width"]
            h = region["height_with_margin"] if "height_with_margin" in region else region["height"]
            
            # S'assurer que les limites sont valides
            if y + h > img.shape[0] or x + w > img.shape[1]:
                return False
            
            # Extraire la région de la case
            checkbox_img = img[y:y+h, x:x+w]
            
            # AMÉLIORATION 1: Analyse du contenu intérieur (évite la bordure)
            inner_margin = 3
            if w > 2*inner_margin and h > 2*inner_margin:
                # Masque pour extraire uniquement la partie intérieure
                inner_mask = np.zeros_like(checkbox_img)
                inner_mask[inner_margin:-inner_margin, inner_margin:-inner_margin] = 1
                
                # Binarisation adaptative
                binary = cv2.adaptiveThreshold(
                    checkbox_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                
                # Appliquer le masque intérieur
                inner_binary = binary * inner_mask
                
                # Calculer le ratio de pixels blancs à l'intérieur
                inner_area = np.sum(inner_mask)
                if inner_area > 0:
                    inner_ratio = np.sum(inner_binary > 0) / inner_area
                    
                    # Si beaucoup de contenu à l'intérieur, c'est probablement coché
                    if inner_ratio > 0.15:  # Seuil abaissé pour être plus sensible
                        return True
                
                # AMÉLIORATION 2: Analyse morphologique pour détecter X et ✓
                # Appliquer des opérations morphologiques pour trouver des structures
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(inner_binary, kernel, iterations=1)
                dilated = cv2.dilate(eroded, kernel, iterations=1)
                
                # Si après érosion et dilatation on garde une structure
                # c'est probablement une marque comme X ou ✓
                morph_ratio = np.sum(dilated > 0) / inner_area if inner_area > 0 else 0
                if morph_ratio > 0.08:
                    return True
            
            # AMÉLIORATION 3: Analyse de densité du centre (où se trouve généralement la marque)
            center_margin = max(inner_margin * 2, min(w, h) // 4)
            if w > 2*center_margin and h > 2*center_margin:
                # Masque pour le centre uniquement
                center_mask = np.zeros_like(checkbox_img)
                center_mask[center_margin:-center_margin, center_margin:-center_margin] = 1
                
                # Binarisation 
                _, binary = cv2.threshold(checkbox_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Analyser le centre
                center_binary = binary * center_mask
                center_area = np.sum(center_mask)
                if center_area > 0:
                    center_ratio = np.sum(center_binary > 0) / center_area
                    if center_ratio > 0.2:  # Plus de contenu au centre
                        return True
            
            # Analyse globale (fallback)
            _, binary = cv2.threshold(checkbox_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            total_pixels = checkbox_img.size
            global_ratio = np.sum(binary > 0) / total_pixels if total_pixels > 0 else 0
            
            return global_ratio > 0.15  # Seuil légèrement plus bas
            
        except Exception as e:
            logger.debug(f"Erreur analyse case cochée: {e}")
            return False
    
    def export_to_csv(self, results: Dict[str, Any], output_file: str) -> bool:
        """
        Exporte les résultats au format CSV.
        
        Args:
            results: Résultats de l'extraction
            output_file: Chemin du fichier de sortie
            
        Returns:
            True si l'export a réussi, False sinon
        """
        try:
            import csv
            
            table_data = self.format_to_table(results)
            if not table_data:
                return False
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["Section", "Label", "Valeur", "Page"])
                writer.writeheader()
                writer.writerows(table_data)
            
            return True
        except Exception as e:
            logger.error(f"Erreur export CSV: {e}")
            return False
    
    def extract_selected_values(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Extrait uniquement les valeurs sélectionnées (cases cochées) dans un format clair et normalisé.
        
        Cette version améliorée:
        - Gère correctement les groupes Oui/Non
        - Détecte les boutons radio et options mutuellement exclusives
        - Utilise une structure plus intuitive pour l'exploitation des données
        
        Args:
            results: Résultats de l'extraction
            
        Returns:
            Dictionnaire avec les paires label:valeur pour les cases cochées
        """
        selected_values = {}
        
        # Pré-traitement pour identifier les groupes et relations
        groups = {}
        option_groups = {}
        
        # 1. Première passe: identifier les relations et groupes
        for idx, checkbox in enumerate(results.get("checkboxes", [])):
            label = checkbox.get("label", "").strip()
            if not label:
                continue
                
            value = checkbox.get("value", "").strip()
            checked = checkbox.get("checked", False)
            group_id = checkbox.get("group_id")
            
            # Identifier les groupes Oui/Non par proximité spatiale ou contexte textuel
            if value.lower() in ["oui", "yes", "true", "non", "no", "false"]:
                # Tenter de trouver une clé de groupe basée sur le label sans le "Oui" ou "Non"
                group_key = label
                for prefix in ["oui", "yes", "true", "non", "no", "false"]:
                    if group_key.lower().startswith(prefix):
                        # Enlever le préfixe et les espaces
                        group_key = group_key[len(prefix):].strip()
                        break
                    elif group_key.lower().endswith(prefix):
                        # Enlever le suffixe et les espaces
                        group_key = group_key[:-len(prefix)].strip()
                        break
                
                # Si un groupe clé a été identifié, l'ajouter aux groupes d'options
                if group_key:
                    if group_key not in option_groups:
                        option_groups[group_key] = []
                    option_groups[group_key].append({
                        "idx": idx,
                        "value": value,
                        "checked": checked
                    })
            
            # Collecter les groupes explicites (par ID de groupe)
            if group_id:
                if group_id not in groups:
                    groups[group_id] = []
                groups[group_id].append({
                    "idx": idx,
                    "label": label,
                    "value": value,
                    "checked": checked
                })
        
        # 2. Deuxième passe: traiter chaque case
        for checkbox in results.get("checkboxes", []):
            label = checkbox.get("label", "").strip()
            if not label:
                continue
                
            value = checkbox.get("value", "").strip()
            checked = checkbox.get("checked", False)
            group_id = checkbox.get("group_id")
            
            # Cas 1: Traitement des groupes explicites (par ID de groupe)
            if group_id and checked:
                # Vérifier si ce groupe a déjà été traité
                if group_id not in selected_values:
                    # Utiliser le groupe comme clé et la valeur/label comme valeur
                    if value:
                        selected_values[group_id] = value
                    else:
                        selected_values[group_id] = label
            
            # Cas 2: Option parmi groupe Oui/Non détecté
            elif value.lower() in ["oui", "yes", "true", "non", "no", "false"]:
                # Chercher dans les groupes d'options identifiés
                found_in_group = False
                for group_key, options in option_groups.items():
                    for option in options:
                        if option["value"] == value and option["checked"] == checked:
                            if checked:
                                # Ajouter uniquement si coché
                                if group_key not in selected_values:
                                    selected_values[group_key] = value
                            found_in_group = True
                            break
                    
                    if found_in_group:
                        break
                        
                # Si pas trouvé dans un groupe, traiter comme case individuelle
                if not found_in_group and checked:
                    selected_values[label] = value
            
            # Cas 3: Case à cocher standard (non groupée)
            else:
                # Pour les cases simples, ajouter Oui/Non selon l'état
                selected_values[label] = "Oui" if checked else "Non"
        
        # 3. Nettoyage et normalisation finale
        # Supprimer les valeurs ambiguës ou incomplètes
        clean_values = {}
        for key, value in selected_values.items():
            # Nettoyer les clés et valeurs
            clean_key = key.strip().rstrip(':')
            clean_value = value.strip()
            
            # Ne pas inclure les clés vides ou valeurs vides
            if clean_key and clean_value:
                clean_values[clean_key] = clean_value
        
        return clean_values
    
    async def extract_form_checkboxes(
        self, 
        pdf_path: Union[str, Path, io.BytesIO],
        page_range: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Méthode optimisée pour extraire les cases à cocher des formulaires PDF structurés.
        Utilise des heuristiques spécifiques et techniques avancées pour une meilleure détection.
        
        Args:
            pdf_path: Chemin ou objet BytesIO du fichier PDF
            page_range: Liste optionnelle des pages à analyser (1-based)
            config: Configuration avancée pour la détection
            
        Returns:
            Dictionnaire des résultats avec les cases à cocher
        """
        try:
            with metrics.timer("form_checkbox_extraction"):
                # Configuration par défaut
                default_config = {
                    "confidence_threshold": 0.6,
                    "enhance_detection": True,
                    "detect_groups": True,
                    "include_images": False,
                    "use_ml_detection": False  # Utiliser une approche par apprentissage si disponible
                }
                
                # Fusionner avec la config fournie
                config = {**default_config, **(config or {})}
                
                # Ouvrir le document PDF
                if isinstance(pdf_path, (str, Path)):
                    pdf_doc = fitz.open(pdf_path)
                else:
                    # Pour BytesIO, on doit réinitialiser la position et l'ouvrir comme stream
                    pdf_path.seek(0)
                    pdf_doc = fitz.open(stream=pdf_path.read(), filetype="pdf")
                
                # Si page_range n'est pas spécifié, analyser tout le document
                if page_range is None:
                    page_range = list(range(len(pdf_doc)))
                else:
                    # Convertir page_range de 1-based à 0-based
                    page_range = [p-1 for p in page_range if 0 <= p-1 < len(pdf_doc)]
                
                results = {
                    "metadata": {
                        "filename": getattr(pdf_path, "name", str(pdf_path)) if hasattr(pdf_path, "name") else "unknown",
                        "page_count": len(pdf_doc),
                        "processed_pages": len(page_range),
                        "extraction_date": datetime.now().isoformat(),
                        "config": config
                    },
                    "sections": {},
                    "checkboxes": []
                }
                
                # AMÉLIORATION 1: Détection des zones de formulaire pour une analyse plus ciblée
                form_regions = await self._detect_form_regions(pdf_doc, page_range)
                
                # AMÉLIORATION 2: Analyse multi-techniques pour les cases à cocher
                # Utiliser plusieurs approches et fusionner les résultats
                detection_methods = [
                    self._detect_checkboxes_by_symbols,    # Détection basée sur les symboles spécifiques
                    self._detect_checkboxes_by_structure,  # Détection basée sur la structure du document
                    self._detect_checkboxes_visual         # Détection visuelle basée sur OpenCV
                ]
                
                # Si la détection ML est activée et disponible
                if config["use_ml_detection"] and hasattr(self, "_detect_checkboxes_ml"):
                    detection_methods.append(self._detect_checkboxes_ml)
                
                # Traiter chaque page avec toutes les méthodes
                all_checkboxes = []
                
                for page_idx in page_range:
                    page = pdf_doc[page_idx]
                    
                    # Obtenir le texte structuré de la page
                    text_dict = page.get_text("dict")
                    
                    # Convertir la page en image pour l'analyse visuelle
                    if config["enhance_detection"]:
                        page_img = await self._convert_page_to_image(page)
                    else:
                        page_img = None
                    
                    # Appliquer chaque méthode de détection
                    page_checkboxes = []
                    for method in detection_methods:
                        try:
                            method_results = await method(page, text_dict, page_img, page_idx + 1)
                            if method_results:
                                page_checkboxes.extend(method_results)
                        except Exception as method_error:
                            logger.debug(f"Erreur dans méthode de détection: {method_error}")
                    
                    # Filtrer et fusionner les résultats redondants
                    if page_checkboxes:
                        filtered_checkboxes = self._filter_duplicate_checkboxes(page_checkboxes)
                        all_checkboxes.extend(filtered_checkboxes)
                
                # AMÉLIORATION 3: Post-traitement avancé des cases détectées
                processed_checkboxes = await self._process_checkbox_candidates(
                    all_checkboxes, 
                    pdf_doc, 
                    config["detect_groups"]
                )
                
                # AMÉLIORATION 4: Organisation des cases par section et groupes logiques
                # Identifier les sections et associations de cases
                organized_checkboxes = self._organize_checkboxes_in_sections(processed_checkboxes)
                
                # Ajouter les résultats
                results["checkboxes"] = organized_checkboxes
                
                # Organiser par sections
                section_map = {}
                for checkbox in organized_checkboxes:
                    section = checkbox.get("section", "Information")
                    if section not in section_map:
                        section_map[section] = []
                    section_map[section].append(checkbox)
                
                results["sections"] = section_map
                
                # AMÉLIORATION 5: Générer les représentations visuelles si demandé
                if config["include_images"]:
                    images = []
                    for checkbox in organized_checkboxes:
                        if "page" in checkbox and "position" in checkbox:
                            try:
                                page_num = checkbox["page"] - 1  # Convertir en 0-based
                                if 0 <= page_num < len(pdf_doc):
                                    # Extraire et encoder l'image de la case
                                    checkbox_img = await self._extract_checkbox_image(
                                        pdf_doc[page_num], 
                                        checkbox["position"]
                                    )
                                    if checkbox_img is not None:
                                        images.append({
                                            "checkbox_id": checkbox.get("id", str(len(images))),
                                            "page": checkbox["page"],
                                            "label": checkbox.get("label", ""),
                                            "data": checkbox_img
                                        })
                            except Exception as img_error:
                                logger.debug(f"Erreur extraction image case: {img_error}")
                    
                    results["checkbox_images"] = images
                
                # Générer l'extraction des valeurs pour faciliter l'utilisation
                form_values = self.extract_selected_values(results)
                results["form_values"] = form_values
                
                # Fermer le document
                pdf_doc.close()
                
                return results
                
        except Exception as e:
            logger.error(f"Erreur extraction cases à cocher formulaire: {e}")
            metrics.increment_counter("checkbox_extraction_errors")
            return {
                "error": str(e),
                "sections": {},
                "checkboxes": []
            }
        
    async def _detect_form_regions(self, pdf_doc, page_range) -> List[Dict[str, Any]]:
        """
        Détecte les régions de formulaire dans les pages du document.
        Utilisé pour cibler l'analyse des cases à cocher.
        
        Args:
            pdf_doc: Document PDF ouvert
            page_range: Liste des indices de pages à analyser
            
        Returns:
            Liste des régions de formulaire détectées
        """
        form_regions = []
        
        for page_idx in page_range:
            page = pdf_doc[page_idx]
            
            # 1. Recherche de champs de formulaire (PDF interactif)
            widget_count = 0
            field_rect = None
            
            # Vérifier si la page contient des champs de formulaire
            for widget in page.widgets():
                widget_count += 1
                if field_rect is None:
                    field_rect = widget.rect
                else:
                    field_rect |= widget.rect  # Union des rectangles
            
            # Si des champs sont trouvés, ajouter la région
            if widget_count > 0 and field_rect:
                form_regions.append({
                    "page": page_idx + 1,
                    "x": field_rect.x0,
                    "y": field_rect.y0,
                    "width": field_rect.width,
                    "height": field_rect.height,
                    "type": "interactive_form",
                    "confidence": 0.9
                })
                continue  # Passer à la page suivante
            
            # 2. Analyse du texte pour les formulaires non interactifs
            text_dict = page.get_text("dict")
            
            # Recherche de motifs textuels typiques des formulaires (étiquettes, cases à cocher)
            form_indicators = 0
            form_rect = None
            
            for block in text_dict["blocks"]:
                if block["type"] == 0:  # Block de texte
                    block_text = " ".join([span["text"] for line in block["lines"] for span in line["spans"]])
                    
                    # Recherche de motifs de formulaire
                    if any(pattern in block_text.lower() for pattern in 
                        ["oui", "non", "yes", "no", "cochez", "check", "select", "option"]):
                        form_indicators += 1
                        
                        block_rect = fitz.Rect(block["bbox"])
                        if form_rect is None:
                            form_rect = block_rect
                        else:
                            form_rect |= block_rect
            
            # Si des indicateurs de formulaire sont trouvés
            if form_indicators >= 3 and form_rect:
                form_regions.append({
                    "page": page_idx + 1,
                    "x": form_rect.x0,
                    "y": form_rect.y0,
                    "width": form_rect.width,
                    "height": form_rect.height,
                    "type": "text_form",
                    "confidence": 0.7
                })
        
        return form_regions
    
    def _detect_checkboxes_by_structure(self, page, text_dict, page_img, page_num):
        """
        Détecte les cases à cocher basées sur la structure du document.
        
        Args:
            page: Page PDF (fitz page)
            text_dict: Dictionnaire de texte extrait
            page_img: Image de la page convertie en array numpy
            page_num: Numéro de la page
            
        Returns:
            Liste des cases à cocher détectées
        """
        checkboxes = []
        
        # Détecter les cases à cocher basées sur la structure du document
        # Recherche de motifs structurels typiques comme les listes à puces, etc.
        
        # Paramètres de détection
        min_bullet_size = 8
        max_bullet_size = 25
        
        # Extraire les blocs de texte
        blocks = text_dict.get("blocks", [])
        
        for block_idx, block in enumerate(blocks):
            if block["type"] != 0:  # Ignorer les non-texte
                continue
            
            # Récupérer les lignes du bloc
            for line_idx, line in enumerate(block.get("lines", [])):
                line_spans = line.get("spans", [])
                
                # Recherche de modèles typiques de cases à cocher dans les spans
                for span_idx, span in enumerate(line_spans):
                    span_text = span.get("text", "").strip()
                    span_font = span.get("font", "")
                    
                    # 1. Vérifier si le span contient des symboles de cases à cocher
                    has_checkbox_symbol = any(symbol in span_text for symbol in ["☐", "☑", "☒", "□", "■", "▢", "▣"])
                    
                    # 2. Détecter les structures typiques de cases à cocher
                    is_checkbox_structure = False
                    
                    # 2.1 Vérifier si c'est un élément court suivi d'un texte plus long (typique d'une structure de liste)
                    if len(span_text) <= 2 and span_idx + 1 < len(line_spans):
                        next_span = line_spans[span_idx + 1]
                        next_text = next_span.get("text", "").strip()
                        if len(next_text) > 3:  # Le texte suivant est significativement plus long
                            is_checkbox_structure = True
                    
                    # 2.2 Vérifier les spans de type puce/bullet
                    if span_text in ["•", "○", "◦", "▪", "▫", "◆", "◇", "–", "-", "*"]:
                        is_checkbox_structure = True
                    
                    # Si c'est un candidat de case à cocher
                    if has_checkbox_symbol or is_checkbox_structure:
                        # Récupérer les coordonnées du span
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        x0, y0, x1, y1 = bbox
                        
                        # Déterminer l'état (coché/non coché)
                        is_checked = any(symbol in span_text for symbol in ["☑", "☒", "■"])
                        
                        # Obtenir l'étiquette de la case (souvent dans le span suivant)
                        label = ""
                        if span_idx + 1 < len(line_spans):
                            label = line_spans[span_idx + 1].get("text", "").strip()
                        
                        # Si aucune étiquette trouvée, utiliser le texte de la ligne entière
                        if not label:
                            label = " ".join([s.get("text", "") for s in line_spans]).strip()
                            # Retirer les symboles de case à cocher de l'étiquette
                            for symbol in ["☐", "☑", "☒", "□", "■", "▢", "▣", "•", "○", "◦", "▪", "▫"]:
                                label = label.replace(symbol, "").strip()
                        
                        # Créer l'objet case à cocher
                        checkbox = {
                            "label": label,
                            "checked": is_checked,
                            "bbox": bbox,
                            "page": page_num,
                            "confidence": 0.7 if has_checkbox_symbol else 0.5,
                            "type": "structure"
                        }
                        
                        checkboxes.append(checkbox)
        
        return checkboxes
    
    def _detect_square_checkboxes(self, binary_img, scale=1.0):
        """
        Détecte les formes carrées qui pourraient être des cases à cocher.
        
        Args:
            binary_img: Image binaire
            scale: Facteur d'échelle pour les coordonnées
            
        Returns:
            Liste des régions de cases à cocher détectées
        """
        # Recherche de contours
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        checkbox_regions = []
        
        for contour in contours:
            # Ignorer les contours trop petits
            if cv2.contourArea(contour) < self.min_checkbox_area:
                continue
            
            # Calculer le rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            
            # Vérifier si c'est approximativement un carré
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if (self.checkbox_min_size <= w <= self.checkbox_max_size and 
                self.checkbox_min_size <= h <= self.checkbox_max_size and
                abs(aspect_ratio - 1.0) <= self.checkbox_tolerance):
                
                # Vérifier la solidité (rapport entre l'aire du contour et l'aire du rectangle englobant)
                contour_area = cv2.contourArea(contour)
                rect_area = w * h
                solidity = float(contour_area) / rect_area if rect_area > 0 else 0
                
                if solidity > 0.7:  # Les carrés ont une solidité élevée
                    # Appliquer le facteur d'échelle
                    x_scaled = int(x / scale)
                    y_scaled = int(y / scale)
                    w_scaled = int(w / scale)
                    h_scaled = int(h / scale)
                    
                    # Ajouter des marges pour capturer un peu plus de contexte
                    margin = max(2, int(w_scaled * 0.1))
                    x_with_margin = max(0, x_scaled - margin)
                    y_with_margin = max(0, y_scaled - margin)
                    width_with_margin = w_scaled + 2 * margin
                    height_with_margin = h_scaled + 2 * margin
                    
                    checkbox_regions.append({
                        "x": x_scaled,
                        "y": y_scaled,
                        "width": w_scaled,
                        "height": h_scaled,
                        "x_with_margin": x_with_margin,
                        "y_with_margin": y_with_margin,
                        "width_with_margin": width_with_margin,
                        "height_with_margin": height_with_margin,
                        "area": contour_area / (scale * scale),
                        "solidity": solidity
                    })
        
        return checkbox_regions
    
    async def _convert_page_to_image(self, page):
        """
        Convertit une page PDF en image pour l'analyse visuelle.
        
        Args:
            page: Page PDF (fitz page)
            
        Returns:
            Image numpy de la page
        """
        try:
            # Créer un pixmap avec une résolution suffisante
            pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi/72, self.dpi/72))
            
            # Convertir en array numpy
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            # Convertir en RGB si nécessaire
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            return img
        except Exception as e:
            self.logger.error(f"Erreur conversion page en image: {e}")
            return None

    def _filter_checkbox_candidates(self, checkbox_regions):
        """
        Filtre les candidats de cases à cocher pour éliminer les faux positifs.
        
        Args:
            checkbox_regions: Liste des régions candidates
            
        Returns:
            Liste filtrée des régions de cases à cocher
        """
        if not checkbox_regions:
            return []
        
        filtered_regions = []
        
        # Trier les régions par score de confiance
        sorted_regions = sorted(checkbox_regions, key=lambda x: -(x.get("confidence", 0) or 0))
        
        # Filtrer les régions qui se chevauchent trop
        for region in sorted_regions:
            x1, y1 = region.get("x", 0), region.get("y", 0)
            w1, h1 = region.get("width", 0), region.get("height", 0)
            area1 = w1 * h1
            
            # Ignorer les régions trop petites ou trop grandes
            if area1 < self.min_checkbox_area or w1 > self.checkbox_max_size * 2 or h1 > self.checkbox_max_size * 2:
                continue
            
            # Vérifier si cette région chevauche une région déjà acceptée
            is_overlapping = False
            for accepted in filtered_regions:
                x2, y2 = accepted.get("x", 0), accepted.get("y", 0)
                w2, h2 = accepted.get("width", 0), accepted.get("height", 0)
                
                # Calculer le chevauchement
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = x_overlap * y_overlap
                
                # Si le chevauchement est significatif
                if overlap_area > 0.5 * min(area1, w2 * h2):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                # Vérifier le ratio largeur/hauteur pour s'assurer que c'est approximativement carré
                aspect_ratio = w1 / h1 if h1 > 0 else 0
                if 0.7 <= aspect_ratio <= 1.3:  # Approximativement carré
                    region["confidence"] = min(1.0, region.get("confidence", 0.7) * 1.2)  # Bonus de confiance
                else:
                    region["confidence"] = max(0.1, region.get("confidence", 0.7) * 0.8)  # Malus de confiance
                
                # Ajouter à la liste filtrée
                filtered_regions.append(region)
        
        # Post-traitement: stabiliser les tailles des régions similaires
        if filtered_regions:
            # Calculer la taille médiane
            widths = [r.get("width", 0) for r in filtered_regions]
            heights = [r.get("height", 0) for r in filtered_regions]
            median_width = sorted(widths)[len(widths) // 2]
            median_height = sorted(heights)[len(heights) // 2]
            
            # Ajuster les régions proches de la médiane
            for region in filtered_regions:
                w, h = region.get("width", 0), region.get("height", 0)
                
                # Si la taille est proche de la médiane, l'ajuster pour plus de cohérence
                if 0.8 * median_width <= w <= 1.2 * median_width and 0.8 * median_height <= h <= 1.2 * median_height:
                    # Ajuster la taille tout en conservant le centre
                    x, y = region.get("x", 0), region.get("y", 0)
                    center_x = x + w / 2
                    center_y = y + h / 2
                    
                    # Définir la nouvelle taille normalisée
                    new_width = median_width
                    new_height = median_height
                    
                    # Recalculer les coordonnées à partir du centre
                    region["x"] = max(0, int(center_x - new_width / 2))
                    region["y"] = max(0, int(center_y - new_height / 2))
                    region["width"] = new_width
                    region["height"] = new_height
        
        return filtered_regions

    # Méthode supplémentaire pour la détection de checkbox par forme
    def find_peaks(x, height=None, threshold=None, distance=None):
        """
        Version simplifiée de scipy.signal.find_peaks pour détecter les pics dans un histogramme.
        """
        results = []
        if len(x) < 3:
            return np.array(results), {}
        
        # Trouver les maxima locaux
        for i in range(1, len(x)-1):
            if x[i-1] < x[i] and x[i] > x[i+1]:
                # Vérifier le seuil minimal
                if height is None or x[i] >= height:
                    results.append(i)
        
        # Filtrer par distance si spécifiée
        if distance is not None and len(results) > 1:
            filtered = [results[0]]
            for i in range(1, len(results)):
                if results[i] - filtered[-1] >= distance:
                    filtered.append(results[i])
            results = filtered
        
        return np.array(results), {"peak_heights": [x[i] for i in results]}

    def _detect_checkboxes_visual(self, page, text_dict, page_img, page_num):
        """
        Détecte les cases à cocher visuellement dans l'image de la page.
        
        Args:
            page: Page PDF (fitz page)
            text_dict: Dictionnaire de texte extrait
            page_img: Image de la page convertie en array numpy
            page_num: Numéro de la page
            
        Returns:
            Liste des cases à cocher détectées
        """
        checkboxes = []
        
        # Vérifier si l'image est valide
        if page_img is None or page_img.size == 0:
            return checkboxes
        
        # Convertir en niveaux de gris si nécessaire
        if len(page_img.shape) > 2:
            gray = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = page_img
        
        # Binarisation pour la détection de formes
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 1. Détecter les cases à cocher carrées
        square_checkboxes = self._detect_square_checkboxes(binary)
        
        # 2. Détecter les cases à cocher circulaires (boutons radio)
        circle_checkboxes = self._detect_circle_checkboxes(binary)
        
        # Combiner les deux types de cases
        all_regions = square_checkboxes + circle_checkboxes
        
        # Trier les régions par position (haut en bas, gauche à droite)
        all_regions.sort(key=lambda r: (r.get("y", 0), r.get("x", 0)))
        
        # Associer les cases détectées avec les textes
        for region in all_regions:
            # Coordonnées de la région
            x, y = region.get("x", 0), region.get("y", 0)
            width, height = region.get("width", 0), region.get("height", 0)
            
            # Détecter l'état (cochée ou non)
            region_img = gray[y:y+height, x:x+width] if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1] else None
            is_checked = self._is_checkbox_checked(gray, region) if region_img is not None else False
            
            # Chercher l'étiquette associée dans le texte proche
            label = self._find_nearest_label(x, y, width, height, text_dict)
            
            # Ajouter à la liste des cases détectées
            checkbox = {
                "label": label,
                "checked": is_checked,
                "bbox": [x, y, x+width, y+height],
                "page": page_num,
                "confidence": region.get("confidence", 0.7),
                "type": region.get("type", "visual"),
                "area": region.get("area", width * height)
            }
            
            checkboxes.append(checkbox)
        
        return checkboxes

    def _detect_circle_checkboxes(self, binary_img, scale=1.0):
        """
        Détecte les formes circulaires qui pourraient être des boutons radio.
        
        Args:
            binary_img: Image binaire
            scale: Facteur d'échelle pour les coordonnées
            
        Returns:
            Liste des régions de boutons radio détectées
        """
        # Paramètres pour la détection de cercles
        min_radius = 5
        max_radius = 15
        
        # Détection des cercles avec la transformation de Hough
        circles = cv2.HoughCircles(
            binary_img, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20, 
            param1=50, 
            param2=30, 
            minRadius=min_radius, 
            maxRadius=max_radius
        )
        
        checkbox_regions = []
        
        if circles is not None:
            # Convertir les coordonnées en entiers
            circles = np.uint16(np.around(circles))
            
            for circle in circles[0, :]:
                # Extraire les coordonnées et le rayon
                center_x, center_y, radius = circle
                
                # Calculer les coordonnées du rectangle englobant
                x = int((center_x - radius) / scale)
                y = int((center_y - radius) / scale)
                diameter = int(2 * radius / scale)
                
                # Ajouter des marges pour capturer un peu plus de contexte
                margin = max(2, int(diameter * 0.1))
                x_with_margin = max(0, x - margin)
                y_with_margin = max(0, y - margin)
                width_with_margin = diameter + 2 * margin
                height_with_margin = diameter + 2 * margin
                
                # Vérifier si c'est réellement un cercle en examinant le contenu
                is_circle = True  # On pourrait ajouter une vérification plus poussée ici
                
                if is_circle:
                    checkbox_regions.append({
                        "x": x,
                        "y": y,
                        "width": diameter,
                        "height": diameter,
                        "x_with_margin": x_with_margin,
                        "y_with_margin": y_with_margin,
                        "width_with_margin": width_with_margin,
                        "height_with_margin": height_with_margin,
                        "radius": int(radius / scale),
                        "type": "circle",
                        "confidence": 0.7
                    })
        
        return checkbox_regions

    def _find_nearest_label(self, x, y, width, height, text_dict):
        """
        Trouve l'étiquette de texte la plus proche d'une case à cocher.
        
        Args:
            x, y, width, height: Coordonnées de la case à cocher
            text_dict: Dictionnaire de texte extrait
            
        Returns:
            Étiquette trouvée ou chaîne vide
        """
        # Calculer le centre de la case à cocher
        checkbox_center_x = x + width // 2
        checkbox_center_y = y + height // 2
        
        # Distance maximale pour considérer un texte comme étiquette
        max_distance = 100
        
        nearest_text = ""
        min_distance = float('inf')
        
        # Parcourir les blocs de texte
        for block in text_dict.get("blocks", []):
            if block.get("type", -1) != 0:  # Ignorer les non-texte
                continue
            
            # Parcourir les lignes du bloc
            for line in block.get("lines", []):
                # Récupérer le texte de la ligne
                line_text = " ".join([span.get("text", "") for span in line.get("spans", [])])
                line_text = line_text.strip()
                
                if not line_text:
                    continue
                
                # Coordonnées du centre de la ligne
                line_bbox = line.get("bbox", [0, 0, 0, 0])
                line_center_x = (line_bbox[0] + line_bbox[2]) / 2
                line_center_y = (line_bbox[1] + line_bbox[3]) / 2
                
                # Calculer la distance
                distance = ((line_center_x - checkbox_center_x) ** 2 + 
                            (line_center_y - checkbox_center_y) ** 2) ** 0.5
                
                # Vérifier si c'est l'étiquette la plus proche
                if distance < min_distance and distance < max_distance:
                    # Vérifier que le texte ne contient pas lui-même un symbole de case à cocher
                    if not any(symbol in line_text for symbol in ["☐", "☑", "☒", "□", "■", "▢", "▣"]):
                        min_distance = distance
                        nearest_text = line_text
        
        # Nettoyer l'étiquette
        if nearest_text:
            # Supprimer les caractères spéciaux ou symboles qui pourraient être présents
            for symbol in ["•", "○", "◦", "▪", "▫", "◆", "◇", "–", "-", "*"]:
                nearest_text = nearest_text.replace(symbol, "").strip()
        
        return nearest_text

    def _extract_text_blocks(self, page_text):
        """
        Extrait les blocs de texte depuis le dictionnaire de texte de page.
        
        Args:
            page_text: Dictionnaire de texte de la page (obtenu via page.get_text("dict"))
            
        Returns:
            Liste des blocs de texte normalisés
        """
        text_blocks = []
        
        for block in page_text.get("blocks", []):
            if block["type"] != 0:  # Ignorer les non-texte
                continue
            
            block_text = ""
            block_bbox = list(block.get("bbox", [0, 0, 0, 0]))
            
            # Extraire le texte des spans
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if span_text:
                        block_text += span_text + " "
            
            block_text = block_text.strip()
            
            if block_text:
                # Analyser le bloc pour déterminer si c'est un titre, une case à cocher, etc.
                block_type = "text"
                if any(symbol in block_text for symbol in ["☐", "☑", "☒", "□", "■", "▢", "▣"]):
                    block_type = "checkbox"
                elif block_text.endswith(":") and len(block_text) < 50:
                    block_type = "title"
                
                # Ajouter le bloc normalisé
                text_blocks.append({
                    "text": block_text,
                    "bbox": block_bbox,
                    "type": block_type,
                    "font_size": block.get("lines", [{}])[0].get("spans", [{}])[0].get("size", 0) if block.get("lines") else 0
                })
        
        # Trier les blocs par position verticale
        text_blocks.sort(key=lambda b: b["bbox"][1])
        
        return text_blocks

    def _identify_checkbox_groups(self, checkbox_regions, text_blocks):
        """
        Identifie les groupes logiques de cases à cocher.
        
        Args:
            checkbox_regions: Liste des régions de cases à cocher
            text_blocks: Blocs de texte extraits
            
        Returns:
            Liste des cases à cocher avec information de groupe
        """
        # Paramètres de regroupement
        vertical_tolerance = 20  # Tolérance verticale pour considérer des cases sur la même ligne
        horizontal_min_gap = 50  # Écart horizontal minimum entre des cases d'un même groupe
        
        # Trier les cases à cocher par position Y
        checkbox_regions = sorted(checkbox_regions, key=lambda c: c.get("y", 0))
        
        # Initialiser les groupes
        groups = []
        current_group = []
        last_y = -vertical_tolerance * 2
        
        # Regrouper par proximité verticale
        for checkbox in checkbox_regions:
            y = checkbox.get("y", 0)
            
            # Si on change de ligne
            if y > last_y + vertical_tolerance:
                if current_group:
                    groups.append(current_group)
                current_group = [checkbox]
                last_y = y
            else:
                current_group.append(checkbox)
        
        # Ajouter le dernier groupe
        if current_group:
            groups.append(current_group)
        
        # Pour chaque groupe, trier horizontalement et assigner des IDs
        checkboxes_with_groups = []
        for group_id, group in enumerate(groups):
            # Trier les cases du groupe par position X
            group.sort(key=lambda c: c.get("x", 0))
            
            # Identifier les sous-groupes si les cases sont bien espacées
            subgroups = []
            current_subgroup = [group[0]] if group else []
            
            for i in range(1, len(group)):
                current_x = group[i-1].get("x", 0) + group[i-1].get("width", 0)
                next_x = group[i].get("x", 0)
                
                # Si l'écart horizontal est significatif, c'est un nouveau sous-groupe
                if next_x - current_x > horizontal_min_gap:
                    if current_subgroup:
                        subgroups.append(current_subgroup)
                    current_subgroup = [group[i]]
                else:
                    current_subgroup.append(group[i])
            
            # Ajouter le dernier sous-groupe
            if current_subgroup:
                subgroups.append(current_subgroup)
            
            # Assigner les informations de groupe à chaque case
            for subgroup_id, subgroup in enumerate(subgroups):
                for checkbox in subgroup:
                    # Créer une copie enrichie avec les informations de groupe
                    checkbox_with_group = checkbox.copy()
                    checkbox_with_group["group_id"] = f"group_{group_id}"
                    checkbox_with_group["subgroup_id"] = f"group_{group_id}_sub_{subgroup_id}"
                    
                    # Déterminer le type de sous-groupe (Oui/Non, options multiples, etc.)
                    if len(subgroup) == 2:
                        # Probablement un groupe Oui/Non
                        checkbox_with_group["group_type"] = "binary"
                    elif len(subgroup) > 2:
                        # Probablement un groupe d'options multiples
                        checkbox_with_group["group_type"] = "multiple"
                    else:
                        # Case unique
                        checkbox_with_group["group_type"] = "single"
                    
                    checkboxes_with_groups.append(checkbox_with_group)
        
        return checkboxes_with_groups

    def _filter_duplicate_checkboxes(self, checkboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filtre les cases à cocher en supprimant les doublons.
        
        Args:
            checkboxes: Liste des cases à cocher détectées
            
        Returns:
            Liste filtrée sans doublons
        """
        if not checkboxes:
            return []
        
        # Trier par confiance décroissante
        sorted_checkboxes = sorted(checkboxes, key=lambda x: -(x.get("confidence", 0) or 0))
        
        filtered = []
        seen_positions = set()
        
        for checkbox in sorted_checkboxes:
            # Créer une clé unique basée sur la position
            if "page" in checkbox and "position" in checkbox:
                pos = checkbox["position"]
                pos_key = (
                    checkbox["page"],
                    round(pos.get("x", 0) / 5) * 5,  # Arrondi pour tolérer de petites différences
                    round(pos.get("y", 0) / 5) * 5
                )
                
                if pos_key not in seen_positions:
                    seen_positions.add(pos_key)
                    filtered.append(checkbox)
            else:
                # Si pas de position, utiliser le label comme clé
                label = checkbox.get("label", "")
                if label and all(cb.get("label", "") != label for cb in filtered):
                    filtered.append(checkbox)
        
        return filtered

    async def _extract_checkbox_image(self, page, position: Dict[str, Any]) -> Optional[str]:
        """
        Extrait l'image d'une case à cocher à partir de sa position.
        
        Args:
            page: Page du document PDF
            position: Position de la case à cocher
            
        Returns:
            Image encodée en base64 ou None si échec
        """
        try:
            # Extraire les coordonnées avec une marge
            x = position.get("x", 0)
            y = position.get("y", 0)
            width = position.get("width", 0)
            height = position.get("height", 0)
            
            # Ajouter une marge
            margin = 5
            rect = fitz.Rect(x - margin, y - margin, x + width + margin, y + height + margin)
            
            # Ajuster au cadre de la page
            page_rect = page.rect
            rect &= page_rect  # Intersection
            
            # Extraire l'image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
            img_data = pix.tobytes("png")
            
            # Encoder en base64
            import base64
            return base64.b64encode(img_data).decode("utf-8")
        
        except Exception as e:
            logger.debug(f"Erreur extraction image case: {e}")
            return None

    def _organize_checkboxes_in_sections(self, checkboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Organise les cases à cocher en sections logiques et détecte les groupes.
        
        Args:
            checkboxes: Liste des cases à cocher
            
        Returns:
            Liste organisée avec sections et groupes
        """
        # Identifier les en-têtes potentiels de section
        section_headers = {}
        for checkbox in checkboxes:
            label = checkbox.get("label", "").strip()
            if not label:
                continue
                
            # Les labels terminant par ":" sont souvent des en-têtes
            if label.endswith(":"):
                section_name = label.rstrip(":")
                page = checkbox.get("page", 0)
                pos_y = checkbox.get("position", {}).get("y", 0)
                
                if page not in section_headers:
                    section_headers[page] = []
                    
                section_headers[page].append({
                    "name": section_name,
                    "y": pos_y
                })
        
        # Trier les en-têtes par position Y
        for page in section_headers:
            section_headers[page].sort(key=lambda x: x["y"])
        
        # Assigner des sections basées sur la position
        for checkbox in checkboxes:
            page = checkbox.get("page", 0)
            pos_y = checkbox.get("position", {}).get("y", 0)
            
            # Chercher la section correspondante
            section = "Information"  # Section par défaut
            if page in section_headers:
                for header in section_headers[page]:
                    if header["y"] <= pos_y:
                        section = header["name"]
                    else:
                        break
            
            checkbox["section"] = section
        
        # Détecter et associer les groupes Oui/Non
        self._associate_oui_non_groups(checkboxes)
        
        return checkboxes

    def _associate_oui_non_groups(self, checkboxes: List[Dict[str, Any]]) -> None:
        """
        Détecte et associe les groupes de cases à cocher Oui/Non.
        
        Args:
            checkboxes: Liste des cases à cocher à analyser et modifier
        """
        # 1. Recherche de paires Oui/Non basées sur les labels et valeurs
        oui_cases = []
        non_cases = []
        
        for i, checkbox in enumerate(checkboxes):
            label = checkbox.get("label", "").lower().strip()
            value = checkbox.get("value", "").lower().strip()
            
            # Cas 1: Le label contient explicitement Oui/Non
            if "oui" in label or "yes" in label:
                oui_cases.append((i, checkbox))
            elif "non" in label or "no" in label:
                non_cases.append((i, checkbox))
            # Cas 2: La valeur est Oui/Non
            elif value in ["oui", "yes", "true"]:
                oui_cases.append((i, checkbox))
            elif value in ["non", "no", "false"]:
                non_cases.append((i, checkbox))
        
        # 2. Associer les pairs par proximité spatiale
        associated = set()  # Indices des cases déjà associées
        
        for oui_idx, oui_checkbox in oui_cases:
            if oui_idx in associated:
                continue
                
            oui_page = oui_checkbox.get("page", 0)
            oui_pos = oui_checkbox.get("position", {})
            oui_y = oui_pos.get("y", 0)
            
            # Trouver le "Non" correspondant le plus proche
            best_match = None
            min_distance = float('inf')
            
            for non_idx, non_checkbox in non_cases:
                if non_idx in associated:
                    continue
                    
                non_page = non_checkbox.get("page", 0)
                non_pos = non_checkbox.get("position", {})
                non_y = non_pos.get("y", 0)
                
                # Doivent être sur la même page et à peu près au même niveau
                if oui_page == non_page and abs(oui_y - non_y) < 20:
                    # Calculer distance horizontale
                    oui_x = oui_pos.get("x", 0)
                    non_x = non_pos.get("x", 0)
                    distance = abs(oui_x - non_x)
                    
                    if distance < min_distance and distance < 200:  # Moins de 200 pixels
                        min_distance = distance
                        best_match = (non_idx, non_checkbox)
            
            # Si un match a été trouvé, créer le groupe
            if best_match:
                non_idx, non_checkbox = best_match
                
                # Générer un ID de groupe unique
                group_id = f"oui_non_group_{oui_page}_{oui_y}"
                
                # Extraire l'étiquette commune (du texte avant les options Oui/Non)
                oui_label = oui_checkbox.get("label", "")
                non_label = non_checkbox.get("label", "")
                
                # Tenter de trouver une étiquette commune
                common_label = self._extract_common_label(oui_label, non_label)
                
                # Mettre à jour les cases
                oui_checkbox["group_id"] = group_id
                non_checkbox["group_id"] = group_id
                
                if common_label:
                    oui_checkbox["group_label"] = common_label
                    non_checkbox["group_label"] = common_label
                
                # Marquer comme associés
                associated.add(oui_idx)
                associated.add(non_idx)

    def _extract_common_label(self, label1: str, label2: str) -> Optional[str]:
        """
        Extrait une étiquette commune à partir de deux étiquettes de case à cocher.
        
        Args:
            label1, label2: Étiquettes à comparer
            
        Returns:
            Étiquette commune ou None si pas de correspondance
        """
        # Nettoyage des labels
        label1 = label1.lower().strip()
        label2 = label2.lower().strip()
        
        # Cas 1: Un label contient l'autre
        if label1 in label2:
            return label2
        if label2 in label1:
            return label1
        
        # Cas 2: Les labels sont des variantes avec Oui/Non
        for prefix in ["oui", "yes", "non", "no"]:
            if label1.startswith(prefix):
                rest1 = label1[len(prefix):].strip()
                if rest1 and label2.endswith(rest1):
                    return rest1
            if label1.endswith(prefix):
                rest1 = label1[:-len(prefix)].strip()
                if rest1 and label2.startswith(rest1):
                    return rest1
        
        # Cas 3: Recherche du plus long préfixe/suffixe commun
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, label1, label2)
        match = matcher.find_longest_match(0, len(label1), 0, len(label2))
        
        if match.size > 3:  # Au moins 3 caractères communs
            common = label1[match.a:match.a + match.size].strip()
            if common:
                return common
        
        return None