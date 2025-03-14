from typing import List, Dict, Any, Optional, Union, Tuple
import re
import os
import io
from pathlib import Path
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Bibliothèques externes
import numpy as np
import cv2
import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageEnhance

# Imports internes
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings

class TechnicalFormExtractor:
    """
    Extracteur spécialisé pour les fiches techniques CANAMETAL et documents similaires.
    Cette classe analyse spécifiquement les formats de formulaire type fiche d'affaire.
    """
    
    def __init__(self):
        self.logger = get_logger("tech_form_extractor")
    
    async def extract_form_data(self, file_path: Union[str, Path, io.BytesIO]) -> Dict[str, Any]:
        """
        Point d'entrée principal pour extraire les données d'une fiche technique.
        
        Args:
            file_path: Chemin ou objet BytesIO du fichier PDF
            
        Returns:
            Dictionnaire des valeurs extraites structurées par sections
        """
        try:
            # Ouvrir le document PDF - correction pour gérer correctement BytesIO
            pdf_doc = None
            if isinstance(file_path, (str, Path)):
                pdf_doc = fitz.open(file_path)
            else:
                # Pour BytesIO, on doit réinitialiser la position et l'ouvrir comme stream
                if hasattr(file_path, 'seek'):
                    file_path.seek(0)
                    
                # Lire le contenu du BytesIO
                if hasattr(file_path, 'read'):
                    pdf_content = file_path.read()
                    # Ouvrir à partir du contenu binaire
                    pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
            
            if pdf_doc is None:
                raise ValueError(f"Impossible d'ouvrir le document PDF: {type(file_path)}")
            
            try:
                # Analyser toutes les pages
                form_data = {
                    "metadata": self._extract_metadata(pdf_doc),
                    "sections": {}
                }
                
                # Pour chaque page
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    
                    # Extraire les sections de la page
                    page_sections = await self._extract_page_sections(page, page_num)
                    
                    # Fusionner avec les données existantes
                    for section_name, section_data in page_sections.items():
                        if section_name not in form_data["sections"]:
                            form_data["sections"][section_name] = {}
                        
                        # Mettre à jour la section
                        form_data["sections"][section_name].update(section_data)
                
                # Nettoyage final et post-traitement
                form_data = self._post_process_form_data(form_data)
                
                return form_data
            finally:
                # Fermer le document PDF proprement
                if pdf_doc:
                    pdf_doc.close()
                    
        except Exception as e:
            self.logger.error(f"Erreur dans l'extraction de la fiche technique: {e}")
            return {"error": str(e)}
    
    def _extract_metadata(self, pdf_doc) -> Dict[str, Any]:
        """Extrait les métadonnées du document"""
        metadata = {
            "title": "",
            "date": "",
            "reference": "",
            "client": "",
            "page_count": len(pdf_doc)
        }
        
        # Analyser la première page pour les métadonnées
        if len(pdf_doc) > 0:
            page = pdf_doc[0]
            text = page.get_text()
            
            # Chercher des patterns spécifiques aux fiches CANAMETAL
            ref_match = re.search(r'[Nn][°\s.:]*\s*([A-Z0-9]+)', text)
            if ref_match:
                metadata["reference"] = ref_match.group(1)
            
            date_match = re.search(r'(?:le|date)\s*:?\s*(\d{1,2}\/\d{1,2}\/\d{4}|\d{1,2}\/\d{1,2}\/\d{2})', text, re.IGNORECASE)
            if date_match:
                metadata["date"] = date_match.group(1)
            
            client_match = re.search(r'Client\s*:?\s*([^\n\r]{2,30})', text)
            if client_match:
                metadata["client"] = client_match.group(1).strip()
            
            # Recherche du titre (généralement en haut de page ou après "Fiche")
            title_match = re.search(r'(?:Fiche|LOT)\s+([^\n\r]{2,50})', text)
            if title_match:
                metadata["title"] = title_match.group(1).strip()
        
        return metadata
    
    async def _extract_page_sections(self, page, page_num: int) -> Dict[str, Dict[str, Any]]:
        """
        Extrait les sections et leurs champs à partir d'une page.
        
        Args:
            page: Page PDF (fitz)
            page_num: Numéro de la page
            
        Returns:
            Dictionnaire des sections et leurs champs
        """
        sections = {}
        
        # 1. Extraire le texte de la page et les détails des éléments
        text_dict = page.get_text("dict")
        
        # 2. Identifier les sections
        section_blocks = []
        current_section = None
        
        for block in text_dict["blocks"]:
            # Ignorer les blocks non-textuels
            if block["type"] != 0:
                continue
                
            # Extraire le texte du bloc
            block_text = " ".join([span["text"] for line in block["lines"] for span in line["spans"]])
            block_text = block_text.strip()
            
            # Vérifier si c'est un en-tête de section
            if re.match(r'^\d+\/\s*[A-Z]', block_text) or block_text.endswith(":"):
                # C'est un entête de section
                current_section = {
                    "name": block_text.rstrip(":"),
                    "bbox": block["bbox"],
                    "fields": []
                }
                section_blocks.append(current_section)
            elif current_section is not None:
                # Ajouter aux champs de la section courante
                current_section["fields"].append({
                    "text": block_text,
                    "bbox": block["bbox"]
                })
        
        # 3. Analyser chaque section pour détecter les champs et cases à cocher
        for section in section_blocks:
            section_name = section["name"]
            fields_data = {}
            
            # Ajouter les champs individuels
            for field in section["fields"]:
                # Analyser le champ pour déterminer s'il contient des cases à cocher
                field_data = await self._analyze_field(field["text"], field["bbox"], page)
                
                if field_data:
                    # Ajouter à la section
                    fields_data.update(field_data)
            
            if fields_data:
                sections[section_name] = fields_data
        
        # 4. Analyser également les zones spécifiques de formulaire (tables formatées)
        form_fields = await self._extract_form_tables(page, page_num)
        if form_fields:
            section_name = "Form_Fields"
            if section_name not in sections:
                sections[section_name] = {}
            sections[section_name].update(form_fields)
        
        return sections
    
    async def _analyze_field(self, text: str, bbox: List[float], page) -> Optional[Dict[str, Any]]:
        """
        Analyse un champ pour détecter les cases à cocher et extraire les valeurs.
        
        Args:
            text: Texte du champ
            bbox: Rectangle englobant (x0, y0, x1, y1)
            page: Page contenant le champ
            
        Returns:
            Données du champ ou None
        """
        # Ignorer les champs vides ou trop courts
        if not text or len(text) < 3:
            return None
        
        result = {}
        
        # 1. Détecter les patterns de cases à cocher
        checkbox_patterns = [
            # Pattern avec "Oui" "Non" explicite
            r'([^:]+):\s*(?:\[?)(Oui|Non)(?:\]?)',
            # Pattern avec case à cocher
            r'([^:]+)\s*(?:☐|☑|☒|■|□|▣|▢|▪|▫)\s*(Oui|Non)',
            # Pattern plus générique
            r'([^:]+)\s*:\s*(\w+)'
        ]
        
        # Tester chaque pattern
        for pattern in checkbox_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 2:
                    label, value = match
                    label = label.strip()
                    value = value.strip()
                    
                    # Vérifier si la valeur est Oui/Non ou autre
                    if value.lower() in ["oui", "non", "yes", "no"]:
                        result[label] = value
        
        # 2. Si aucun match des patterns, analyser visuellement pour les cases à cocher
        if not result:
            # Convertir la page en image pour analyse visuelle
            # Cette partie utilise l'image pour détecter les cases cochées
            img = await self._get_region_image(page, bbox)
            if img is not None:
                checkboxes = await self._detect_checkboxes_in_image(img, text)
                if checkboxes:
                    result.update(checkboxes)
        
        return result
    
    async def _get_region_image(self, page, bbox: List[float]) -> Optional[np.ndarray]:
        """Obtient l'image d'une région de la page"""
        try:
            # Créer un rectangle à partir des coordonnées
            rect = fitz.Rect(bbox)
            
            # Ajuster aux limites de la page pour éviter les erreurs
            page_rect = page.rect
            rect = rect.intersect(page_rect)  # S'assurer que le rectangle est dans les limites
            
            # Vérifier si le rectangle est valide
            if rect.is_empty or rect.width < 5 or rect.height < 5:
                return None
            
            # Obtenir un pixmap de la région
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
            except Exception as pix_error:
                self.logger.debug(f"Erreur lors de la génération du pixmap: {pix_error}")
                # Essayer avec une matrice plus petite en cas d'erreur
                pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), clip=rect)
            
            # Vérifier si le pixmap est valide
            if pix.width < 1 or pix.height < 1:
                return None
                
            # Convertir en np.array pour OpenCV
            try:
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                
                # Convertir en niveaux de gris
                if img.shape[2] == 3 or img.shape[2] == 4:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img
                    
                return gray
            except Exception as array_error:
                self.logger.debug(f"Erreur conversion pixmap en array: {array_error}")
                return None
        except Exception as e:
            self.logger.debug(f"Erreur récupération image: {e}")
        return None
    
    async def _detect_checkboxes_in_image(self, img: np.ndarray, text: str) -> Dict[str, str]:
        """
        Détecte les cases à cocher dans une image et les associe au texte.
        
        Args:
            img: Image en niveaux de gris
            text: Texte associé à l'image
            
        Returns:
            Dictionnaire des champs détectés et leur état
        """
        # Algorithme simplifié pour détecter les cases à cocher
        results = {}
        
        # Rechercher des motifs typiques dans le texte
        field_matches = re.findall(r'([A-Za-zÀ-ÿ\s\-\_\d]+)(?:\s*:)?', text)
        
        if not field_matches:
            return results
            
        # Binarisation pour détection
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Recherche de rectangles/carrés (cases à cocher)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer les contours de forme carrée/rectangulaire
        checkboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Vérifier si c'est proche d'un carré et de taille raisonnable
            if 0.7 <= aspect_ratio <= 1.3 and 10 <= w <= 30 and 10 <= h <= 30:
                area = cv2.contourArea(contour)
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Les carrés ont une solidité élevée
                if solidity > 0.8:
                    # Région de la case
                    checkbox_region = binary[y:y+h, x:x+w]
                    # Calculer le ratio de remplissage (si coché)
                    fill_ratio = np.sum(checkbox_region > 0) / (w * h)
                    
                    checkboxes.append({
                        'x': x, 'y': y,
                        'width': w, 'height': h,
                        'fill_ratio': fill_ratio,
                        'checked': fill_ratio > 0.2,  # Seuil pour case cochée
                    })
        
        # Associer les cases aux champs
        if checkboxes and field_matches:
            # Simplification: associer les N premières cases aux N premiers champs
            for i, field in enumerate(field_matches):
                if i < len(checkboxes):
                    checkbox = checkboxes[i]
                    field_name = field.strip()
                    # Déterminer l'état (cochée/non cochée)
                    results[field_name] = "Oui" if checkbox['checked'] else "Non"
        
        return results
    
    async def _extract_form_tables(self, page, page_num: int) -> Dict[str, Any]:
        """
        Extrait les tableaux de formulaire et détecte les cases à cocher dans les cellules.
        
        Args:
            page: Page du document
            page_num: Numéro de la page
            
        Returns:
            Dictionnaire des champs extraits
        """
        form_fields = {}
        
        # 1. Rechercher des tableaux structurés
        # Cette méthode est spécifique pour les fiches CANAMETAL
        
        # Obtenir le texte organisé par blocs pour une meilleure analyse
        text_dict = page.get_text("dict")
        
        # Rechercher des motifs spécifiques aux formulaires techniques
        # Par exemple "Oui Non" côte à côte
        oui_non_pattern = re.compile(r'(Oui|Non)\s+(Oui|Non)', re.IGNORECASE)
        
        # Convertir la page en image pour la détection visuelle des cases
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        if img.shape[2] == 3 or img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Détection des lignes horizontales et verticales (structure de tableau)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10
        )
        
        # Si des lignes sont détectées, c'est probablement un tableau
        if lines is not None and len(lines) > 10:
            # 2. Traiter les blocs de texte pour extraire le contenu des cellules
            rows = []
            current_row = []
            last_y = 0
            y_tolerance = 20  # Tolérance pour grouper des éléments sur la même ligne
            
            # Trier les blocs par position Y
            blocks = [b for b in text_dict["blocks"] if b["type"] == 0]  # Type 0 = text block
            blocks.sort(key=lambda b: b["bbox"][1])  # Trier par Y1
            
            # Regrouper les blocs par ligne
            for block in blocks:
                block_text = " ".join([span["text"] for line in block["lines"] for span in line["spans"]])
                block_text = block_text.strip()
                
                if not block_text:
                    continue
                
                y1 = block["bbox"][1]
                
                # Nouvelle ligne si saut vertical significatif
                if len(current_row) > 0 and abs(y1 - last_y) > y_tolerance:
                    rows.append(current_row)
                    current_row = []
                
                current_row.append({
                    "text": block_text,
                    "bbox": block["bbox"]
                })
                last_y = y1
            
            # Ajouter la dernière ligne
            if current_row:
                rows.append(current_row)
            
            # 3. Analyser chaque ligne pour trouver les champs de formulaire
            for row in rows:
                # Rechercher d'abord le libellé (généralement premier élément de la ligne)
                if not row:
                    continue
                    
                # Extraire les textes de la ligne
                row_texts = [cell["text"] for cell in row]
                line_text = " ".join(row_texts)
                
                # Vérifier si la ligne contient un libellé et des options Oui/Non
                if oui_non_pattern.search(line_text):
                    # Détecter les cases cochées
                    checkboxes = []
                    
                    # Analyser visuellement la ligne
                    for cell in row:
                        # Extraire l'image de la cellule
                        cell_img = gray[
                            int(cell["bbox"][1]):int(cell["bbox"][3]), 
                            int(cell["bbox"][0]):int(cell["bbox"][2])
                        ]
                        
                        # Détection spécifique pour "Oui" et "Non"
                        is_checked = False
                        cell_text = cell["text"].strip()
                        
                        if cell_text in ["Oui", "Non"]:
                            # Analyser la cellule pour voir si elle est cochée
                            # Binariser l'image
                            _, cell_bin = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            
                            # Calculer le ratio de remplissage
                            if cell_img.size > 0:
                                fill_ratio = np.sum(cell_bin > 0) / cell_img.size
                                # Détection améliorée pour cases à cocher dans des cellules
                                if fill_ratio > 0.05:  # Seuil bas pour détecter même des marques légères
                                    # Recherche de formes géométriques (X, ✓, etc.)
                                    contours, _ = cv2.findContours(
                                        cell_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                                    )
                                    if contours and len(contours) > 0:
                                        largest_cnt = max(contours, key=cv2.contourArea)
                                        if cv2.contourArea(largest_cnt) > 25:  # Aire minimale
                                            is_checked = True
                            
                            checkboxes.append({
                                "text": cell_text,
                                "checked": is_checked
                            })
                    
                    # Extraire le libellé (texte avant les options Oui/Non)
                    label = ""
                    for text in row_texts:
                        if text not in ["Oui", "Non"]:
                            label = text.strip()
                            # Nettoyer le libellé
                            label = re.sub(r'\s+', ' ', label)
                            break
                    
                    if label:
                        # Déterminer l'état selon les cases cochées
                        selected_value = "Non"  # Valeur par défaut
                        
                        # Parcourir les cases détectées
                        for checkbox in checkboxes:
                            if checkbox["checked"]:
                                selected_value = checkbox["text"]
                                break
                        
                        # Ajouter au résultat
                        form_fields[label] = selected_value
        
        return form_fields
    
    def _post_process_form_data(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-traitement des données du formulaire pour les nettoyer et standardiser.
        
        Args:
            form_data: Données brutes extraites
            
        Returns:
            Données nettoyées et structurées
        """
        # Nettoyer et restructurer les données
        cleaned_data = {
            "metadata": form_data["metadata"],
            "sections": {}
        }
        
        # Nettoyer les sections
        for section_name, section_data in form_data["sections"].items():
            # Nettoyer le nom de section
            clean_section_name = section_name.strip()
            
            # Créer une nouvelle section nettoyée
            if clean_section_name not in cleaned_data["sections"]:
                cleaned_data["sections"][clean_section_name] = {}
            
            # Nettoyer les champs
            for field_name, field_value in section_data.items():
                # Nettoyer le nom du champ
                clean_field_name = field_name.strip()
                
                # Nettoyer la valeur
                clean_value = field_value.strip() if isinstance(field_value, str) else field_value
                
                # Ajouter à la section nettoyée
                cleaned_data["sections"][clean_section_name][clean_field_name] = clean_value
        
        # Aplatir les sections redondantes (Form_Fields)
        if "Form_Fields" in cleaned_data["sections"]:
            form_fields = cleaned_data["sections"].pop("Form_Fields")
            
            # Essayer de les associer aux sections appropriées
            for field_name, field_value in form_fields.items():
                # Rechercher une section existante par préfixe
                section_found = False
                
                for section_prefix in ["1/", "2/", "3/", "4/", "5/"]:
                    if field_name.startswith(section_prefix):
                        section_name = next(
                            (s for s in cleaned_data["sections"] if s.startswith(section_prefix)), 
                            None
                        )
                        
                        if section_name:
                            # Ajouter à la section existante
                            cleaned_data["sections"][section_name][field_name] = field_value
                            section_found = True
                            break
                
                # Si aucune section trouvée, ajouter à une section "Autres"
                if not section_found:
                    if "Autres" not in cleaned_data["sections"]:
                        cleaned_data["sections"]["Autres"] = {}
                    cleaned_data["sections"]["Autres"][field_name] = field_value
        
        return cleaned_data