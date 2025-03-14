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

    async def extract_checkboxes_from_pdf(
        self, 
        pdf_path: Union[str, Path, io.BytesIO],
        page_range: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extrait les cases à cocher d'un document PDF.
        
        Args:
            pdf_path: Chemin ou objet BytesIO du fichier PDF
            page_range: Liste optionnelle des pages à analyser (1-based)
            
        Returns:
            Dictionnaire des résultats avec les sections et leurs cases à cocher
        """
        try:
            with metrics.timer("checkbox_extraction"):
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
            # Binarisation avec diverses méthodes pour améliorer la détection
            methods = [
                # Méthode 1: Binarisation Otsu standard
                lambda x: cv2.threshold(x, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
                
                # Méthode 2: Binarisation adaptative
                lambda x: cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2),
                                            
                # Méthode 3: Flou puis Otsu
                lambda x: cv2.threshold(cv2.GaussianBlur(x, (5, 5), 0), 
                                    0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            ]
            
            all_checkbox_regions = []
            
            # Appliquer chaque méthode de binarisation
            for method_idx, method in enumerate(methods):
                binary = method(img)
                
                # Détection des contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Analyser chaque contour pour détecter les cases à cocher
                for contour in contours:
                    # Calculer l'aire et le périmètre
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    # Ignorer les contours trop petits
                    if area < self.min_checkbox_area:
                        continue
                    
                    # Approximer pour obtenir une forme polygonale
                    epsilon = 0.05 * perimeter
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Les cases à cocher peuvent être des quadrilatères (4 côtés) ou des cercles
                    if 4 <= len(approx) <= 8:
                        x, y, w, h = cv2.boundingRect(approx)
                        
                        # Vérifier la taille et les proportions
                        if (self.checkbox_min_size <= w <= self.checkbox_max_size and 
                            self.checkbox_min_size <= h <= self.checkbox_max_size):
                            
                            # Vérifier si c'est approximativement carré ou rond
                            aspect_ratio = float(w) / h
                            if abs(aspect_ratio - 1.0) <= self.checkbox_tolerance:
                                # Calculer un score de confiance basé sur la forme
                                # Plus la forme est proche d'un carré parfait ou d'un cercle, plus le score est élevé
                                shape_score = 1.0 - abs(aspect_ratio - 1.0)
                                
                                # C'est probablement une case à cocher
                                # Ajouter une marge pour l'extraction
                                margin = 2
                                x_with_margin = max(0, x - margin)
                                y_with_margin = max(0, y - margin)
                                w_with_margin = w + 2 * margin
                                h_with_margin = h + 2 * margin
                                
                                checkbox_regions = {
                                    "x": x,
                                    "y": y,
                                    "width": w,
                                    "height": h,
                                    "x_with_margin": x_with_margin,
                                    "y_with_margin": y_with_margin,
                                    "width_with_margin": w_with_margin,
                                    "height_with_margin": h_with_margin,
                                    "area": area,
                                    "method": method_idx,
                                    "confidence": shape_score,
                                    "center_x": x + w // 2,
                                    "center_y": y + h // 2
                                }
                                all_checkbox_regions.append(checkbox_regions)
            
            # Filtrer les doublons (cases détectées par plusieurs méthodes)
            unique_regions = []
            centers = []
            
            for region in sorted(all_checkbox_regions, key=lambda r: -r["confidence"]):
                center = (region["center_x"], region["center_y"])
                
                # Vérifier si c'est un doublon (même centre ou très proche)
                is_duplicate = False
                for existing_center in centers:
                    distance = np.sqrt((center[0] - existing_center[0])**2 + (center[1] - existing_center[1])**2)
                    if distance < self.checkbox_min_size:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_regions.append(region)
                    centers.append(center)
            
            return unique_regions
            
        except Exception as e:
            logger.error(f"Erreur détection cases à cocher: {e}")
            return []
    
    def _is_checkbox_checked(self, img: np.ndarray, region: Dict[str, Any]) -> bool:
        """
        Détermine si une case à cocher est cochée avec une méthode améliorée.
        
        Args:
            img: Image en niveaux de gris
            region: Région de la case à cocher
            
        Returns:
            True si la case est cochée, False sinon
        """
        try:
            # Extraire la région de la case avec marge
            x = region["x_with_margin"]
            y = region["y_with_margin"]
            w = region["width_with_margin"]
            h = region["height_with_margin"]
            
            # S'assurer que les limites sont valides
            if y + h > img.shape[0] or x + w > img.shape[1]:
                return False
            
            checkbox_img = img[y:y+h, x:x+w]
            
            # Binariser l'image de la case
            _, binary = cv2.threshold(checkbox_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Enlever les bordures qui peuvent affecter la détection
            # Créer un masque intérieur
            inner_margin = 2
            if w > 2*inner_margin and h > 2*inner_margin:
                mask = np.zeros_like(binary)
                mask[inner_margin:-inner_margin, inner_margin:-inner_margin] = 1
                binary = binary * mask
            
            # Calculer le ratio de pixels noirs (contenu) par rapport à l'aire totale de l'intérieur
            black_pixels = np.sum(binary > 0)
            masked_area = np.sum(mask) if 'mask' in locals() else binary.size
            total_area = binary.shape[0] * binary.shape[1]
            
            # Le ratio dépend aussi de la densité des pixels noirs dans la région centrale
            fill_ratio = black_pixels / masked_area if masked_area > 0 else black_pixels / total_area
            
            # Analyse de la distribution des pixels noirs
            # Une case cochée a généralement des pixels plus concentrés au centre
            if 'mask' in locals() and np.sum(mask) > 0:
                center_binary = binary[inner_margin*2:-inner_margin*2, inner_margin*2:-inner_margin*2] if w > 4*inner_margin and h > 4*inner_margin else binary
                center_ratio = np.sum(center_binary > 0) / center_binary.size if center_binary.size > 0 else 0
                
                # Une case est cochée si le ratio central ou le ratio global dépasse le seuil
                return center_ratio > self.checked_threshold * 1.2 or fill_ratio > self.checked_threshold
            
            # Si on ne peut pas analyser le centre, on utilise juste le ratio global
            return fill_ratio > self.checked_threshold
            
        except Exception as e:
            logger.error(f"Erreur analyse case cochée: {e}")
            return False
    
    def _match_labels_to_checkboxes(
        self, 
        checkbox_regions: List[Dict[str, Any]], 
        page_text: Dict[str, Any],
        img: np.ndarray,
        page_num: int
    ) -> List[Dict[str, Any]]:
        """
        Associe les étiquettes aux cases à cocher avec une méthode améliorée.
        
        Args:
            checkbox_regions: Régions des cases à cocher détectées
            page_text: Texte de la page au format dict de PyMuPDF
            img: Image en niveaux de gris
            page_num: Numéro de la page
            
        Returns:
            Liste des cases à cocher avec leurs étiquettes et états
        """
        results = []
        
        # Extraire les blocs de texte
        blocks = page_text.get("blocks", [])
        
        # Extraire tous les spans de texte avec leurs coordonnées
        all_spans = []
        for block in blocks:
            if block["type"] == 0:  # Type 0 = text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if span_text:
                            # Calculer le centre du span
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            span_center_x = (bbox[0] + bbox[2]) / 2
                            span_center_y = (bbox[1] + bbox[3]) / 2
                            
                            all_spans.append({
                                "text": span_text,
                                "center_x": span_center_x,
                                "center_y": span_center_y,
                                "bbox": bbox,
                                "is_value": span_text.lower() in ["oui", "non", "yes", "no", "true", "false"],
                                "is_header": span_text.endswith(":") or span_text.endswith("/"),
                            })
        
        # Structure typique des cases à cocher: "Option Oui Non"
        for region in checkbox_regions:
            # Coordonnées du centre de la case
            checkbox_center_x = region.get("center_x", region["x"] + region["width"] // 2)
            checkbox_center_y = region.get("center_y", region["y"] + region["height"] // 2)
            
            # Déterminer si la case est cochée
            is_checked = self._is_checkbox_checked(img, region)
            
            # Rechercher les spans les plus proches
            nearby_spans = []
            for span in all_spans:
                # Calculer la distance
                distance = np.sqrt(
                    (span["center_x"] - checkbox_center_x)**2 + 
                    (span["center_y"] - checkbox_center_y)**2
                )
                
                if distance < self.label_distance:
                    nearby_spans.append({
                        **span,
                        "distance": distance
                    })
            
            # Trier par distance
            nearby_spans.sort(key=lambda s: s["distance"])
            
            # Valeurs par défaut
            closest_label = None
            closest_value = None
            closest_section = None
            
            # Analyse des spans proches
            for span in nearby_spans:
                text = span["text"].strip()
                
                # Détermination du type de texte
                if span["is_value"]:
                    # C'est une valeur (Oui/Non)
                    if not closest_value:
                        closest_value = text
                elif span["is_header"]:
                    # C'est un en-tête de section
                    if not closest_section:
                        closest_section = text.rstrip(":/")
                else:
                    # C'est probablement une étiquette
                    if not closest_label:
                        closest_label = text
            
            # Si on a trouvé une étiquette et une valeur
            if closest_label and closest_value:
                # Vérifier que la valeur correspond à l'état de la case
                value_matches_state = (
                    (closest_value.lower() in ["oui", "yes", "true"] and is_checked) or 
                    (closest_value.lower() in ["non", "no", "false"] and not is_checked)
                )
                
                if value_matches_state:
                    checkbox_info = {
                        "label": closest_label,
                        "value": closest_value,
                        "checked": is_checked,
                        "section": closest_section or "Information",
                        "page": page_num,
                        "position": {
                            "x": region["x"],
                            "y": region["y"],
                            "width": region["width"],
                            "height": region["height"]
                        },
                        "confidence": region.get("confidence", 0.6)
                    }
                    results.append(checkbox_info)
            
            # Si on a juste trouvé une étiquette sans valeur explicite (cas plus simple)
            elif closest_label and not closest_value:
                checkbox_info = {
                    "label": closest_label,
                    "value": "Coché" if is_checked else "Non coché",
                    "checked": is_checked,
                    "section": closest_section or "Information",
                    "page": page_num,
                    "position": {
                        "x": region["x"],
                        "y": region["y"],
                        "width": region["width"],
                        "height": region["height"]
                    },
                    "confidence": region.get("confidence", 0.5)
                }
                results.append(checkbox_info)
        
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
        Extrait uniquement les valeurs sélectionnées (cases cochées).
        
        Args:
            results: Résultats de l'extraction
            
        Returns:
            Dictionnaire avec les paires label:valeur pour les cases cochées
        """
        selected_values = {}
        
        # Parcourir tous les checkboxes
        for checkbox in results.get("checkboxes", []):
            if checkbox.get("checked"):
                label = checkbox.get("label", "")
                value = checkbox.get("value", "")
                
                if label and value:
                    selected_values[label] = value
        
        return selected_values