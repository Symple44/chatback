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
import math

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
        Version robuste de l'extracteur de cases à cocher avec résolution des incohérences.
        
        Args:
            pdf_path: Chemin du fichier PDF ou objet BytesIO
            page_range: Liste optionnelle des pages à analyser (1-based)
            config: Configuration pour affiner la détection
        
        Returns:
            Dictionnaire contenant les cases à cocher détectées et leur état
        """
        try:
            with metrics.timer("checkbox_extraction"):
                # Préparer la configuration avec des valeurs plus strictes par défaut
                conf = config or {}
                confidence_threshold = conf.get("confidence_threshold", 0.65)
                strict_mode = conf.get("strict_mode", True)
                enhance_detection = conf.get("enhance_detection", True)
                include_images = conf.get("include_images", False)
                
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
                                "strict_mode": strict_mode,
                                "enhance_detection": enhance_detection
                            }
                        },
                        "checkboxes": [],
                    }
                    
                    # Traiter chaque page
                    for page_idx in page_indices:
                        if page_idx >= len(pdf_doc):
                            continue
                            
                        try:
                            page = pdf_doc[page_idx]
                            page_num = page_idx + 1  # Convertir en 1-based
                            
                            # Extraire le texte et convertir en image pour analyse visuelle
                            page_texts = page.get_text("dict")
                            page_image = self._convert_page_to_image(page)
                            
                            # Détecter les cases à cocher
                            checkboxes = await self._detect_checkboxes(
                                page, 
                                page_texts, 
                                page_image, 
                                page_num, 
                                confidence_threshold,
                                enhance_detection
                            )
                            
                            # En mode strict, appliquer un traitement post-détection
                            if strict_mode:
                                try:
                                    # Limiter le nombre de cases cochées par page
                                    checkboxes = self._apply_strict_filtering(checkboxes, page_num)
                                except Exception as e:
                                    logger.error(f"Erreur lors du filtrage strict page {page_num}: {e}")
                            
                            # Ajouter au résultat
                            results["checkboxes"].extend(checkboxes)
                            
                        except Exception as e:
                            logger.error(f"Erreur traitement page {page_idx+1}: {e}")
                            # Continuer avec la page suivante
                    
                    # Post-traitement pour organiser les cases à cocher
                    try:
                        # Stocker temporairement une référence au PDF pour l'extraction des questions
                        self._current_pdf_doc = pdf_doc
                        
                        # Améliorer les étiquettes et grouper les paires Oui/Non
                        results["checkboxes"] = self._detect_and_group_yes_no_pairs(results["checkboxes"])
                        
                        # Filtrer les cases redondantes ou inutiles
                        results["checkboxes"] = self._filter_redundant_checkboxes(results["checkboxes"])
                        
                        # Essayer la structure avancée avec résolution des incohérences
                        try:
                            # Résoudre les incohérences et améliorer la structure
                            self._enhance_checkbox_structure(results, pdf_doc)
                        except Exception as e:
                            logger.error(f"Erreur lors du post-traitement avancé: {e}")
                            # Fallback sur l'organisation simple
                            self._organize_checkboxes(results)
                        
                        # Nettoyer la référence temporaire
                        if hasattr(self, "_current_pdf_doc"):
                            delattr(self, "_current_pdf_doc")
                    except Exception as e:
                        logger.error(f"Erreur organisation des cases: {e}")
                    
                    # En mode strict, effectuer une validation globale
                    if strict_mode:
                        try:
                            self._validate_global_checkbox_results(results)
                        except Exception as e:
                            logger.error(f"Erreur validation globale: {e}")
                    
                    # Vérifier qu'on a des résultats valides
                    if not results.get("checkboxes") and len(page_indices) > 0:
                        results["warning"] = "Extraction terminée mais aucune case à cocher détectée"
                    
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
    
    def _find_questions_for_checkboxes(self, pdf_doc: fitz.Document, checkboxes: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Trouve les questions associées aux paires Oui/Non.
        
        Args:
            pdf_doc: Document PDF
            checkboxes: Liste des cases à cocher
            
        Returns:
            Dictionnaire associant les IDs de paires aux questions
        """
        # Regrouper les cases par paires
        pairs = {}
        questions = {}
        
        # Identifier les paires Oui/Non
        for checkbox in checkboxes:
            pair_id = checkbox.get("pair_id", "")
            if pair_id:
                if pair_id not in pairs:
                    pairs[pair_id] = []
                pairs[pair_id].append(checkbox)
        
        # Pour chaque paire, trouver la question associée
        for pair_id, pair_checkboxes in pairs.items():
            if len(pair_checkboxes) != 2:
                continue
                
            # Récupérer la page et les positions moyennes de la paire
            page_num = pair_checkboxes[0].get("page", 0)
            avg_y = sum(cb.get("bbox", [0, 0, 0, 0])[1] for cb in pair_checkboxes) / 2
            
            try:
                # Obtenir le texte de la page
                page = pdf_doc[page_num - 1]  # Convertir en 0-based
                page_text = page.get_text("dict")
                
                # Chercher des textes qui pourraient être des questions
                question_candidates = []
                
                for block in page_text.get("blocks", []):
                    if block.get("type", -1) != 0:  # Ignorer les blocs non-textuels
                        continue
                        
                    for line in block.get("lines", []):
                        line_bbox = line.get("bbox", [0, 0, 0, 0])
                        line_text = " ".join([span.get("text", "") for span in line.get("spans", [])])
                        
                        # Nettoyer le texte
                        cleaned_text = line_text.strip()
                        if not cleaned_text or cleaned_text.lower() in ["oui", "non", "yes", "no"]:
                            continue
                        
                        # Calculer la distance verticale
                        line_y = (line_bbox[1] + line_bbox[3]) / 2
                        y_distance = line_y - avg_y
                        
                        # Chercher des textes au-dessus de la paire
                        if -150 < y_distance < 0:
                            # Score basé sur la proximité et le contenu
                            score = 100 - abs(y_distance)
                            
                            # Bonus pour les textes qui ressemblent à des questions
                            if "?" in cleaned_text:
                                score += 50
                            elif len(cleaned_text) > 10 and len(cleaned_text) < 100:
                                score += 20
                            
                            question_candidates.append({
                                "text": cleaned_text,
                                "score": score,
                                "distance": abs(y_distance)
                            })
                
                # Trier les candidats par score
                question_candidates.sort(key=lambda x: x["score"], reverse=True)
                
                # Prendre le meilleur candidat
                if question_candidates:
                    questions[pair_id] = question_candidates[0]["text"]
                else:
                    # Fallback: utiliser un identifiant générique
                    questions[pair_id] = f"Question {len(questions) + 1}"
                    
            except Exception as e:
                logger.debug(f"Erreur recherche question pour paire {pair_id}: {e}")
                questions[pair_id] = f"Question {len(questions) + 1}"
        
        return questions


    def _resolve_conflicting_checkboxes(self, checkboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Résout les incohérences pour les paires Oui/Non où les deux options sont cochées.
        
        Args:
            checkboxes: Liste des cases à cocher
            
        Returns:
            Liste des cases à cocher avec résolution des conflits
        """
        # Regrouper par paires
        pairs = {}
        
        for checkbox in checkboxes:
            pair_id = checkbox.get("pair_id", "")
            if pair_id:
                if pair_id not in pairs:
                    pairs[pair_id] = []
                pairs[pair_id].append(checkbox)
        
        # Résoudre les conflits pour chaque paire
        for pair_id, pair_checkboxes in pairs.items():
            if len(pair_checkboxes) != 2:
                continue
                
            # Vérifier si les deux cases sont cochées
            both_checked = all(cb.get("checked", False) for cb in pair_checkboxes)
            
            if both_checked:
                # Trouver quelle case a la plus haute confiance
                pair_checkboxes.sort(key=lambda cb: cb.get("confidence", 0), reverse=True)
                
                # Garder uniquement la case avec la plus haute confiance comme cochée
                pair_checkboxes[1]["checked"] = False
                pair_checkboxes[1]["auto_corrected"] = True
                
                logger.info(f"Résolution de conflit pour paire {pair_id}: maintien de '{pair_checkboxes[0].get('label')}' comme cochée")
        
        # Reconstruire la liste des cases à cocher
        return checkboxes

    def _enhance_checkbox_structure(self, results: Dict[str, Any], pdf_doc: fitz.Document) -> None:
        """
        Améliore la structure des résultats avec l'association des questions.
        
        Args:
            results: Résultats d'extraction
            pdf_doc: Document PDF
        """
        checkboxes = results.get("checkboxes", [])
        
        # 1. Résoudre les conflits (deux cases cochées dans une paire Oui/Non)
        checkboxes = self._resolve_conflicting_checkboxes(checkboxes)
        
        # 2. Trouver les questions pour chaque paire
        questions = self._find_questions_for_checkboxes(pdf_doc, checkboxes)
        
        # 3. Organiser les résultats par questions
        structured_results = {
            "questions": [],
            "individual_checkboxes": []
        }
        
        # Traiter les paires Oui/Non
        processed_checkboxes = set()
        
        for pair_id, question_text in questions.items():
            # Trouver les cases associées à cette paire
            pair_checkboxes = [cb for cb in checkboxes if cb.get("pair_id", "") == pair_id]
            
            if len(pair_checkboxes) != 2:
                continue
                
            # Identifier Oui et Non
            oui_checkbox = next((cb for cb in pair_checkboxes if cb.get("label", "").lower() == "oui"), None)
            non_checkbox = next((cb for cb in pair_checkboxes if cb.get("label", "").lower() == "non"), None)
            
            if not oui_checkbox or not non_checkbox:
                continue
                
            # Déterminer la réponse
            answer = None
            if oui_checkbox.get("checked", False) and not non_checkbox.get("checked", False):
                answer = "Oui"
            elif non_checkbox.get("checked", False) and not oui_checkbox.get("checked", False):
                answer = "Non"
            
            # Ajouter à la structure des questions
            structured_results["questions"].append({
                "text": question_text,
                "answer": answer,
                "page": oui_checkbox.get("page", 0),
                "checkboxes": [oui_checkbox, non_checkbox]
            })
            
            # Marquer ces cases comme traitées
            processed_checkboxes.add(id(oui_checkbox))
            processed_checkboxes.add(id(non_checkbox))
        
        # Ajouter les cases individuelles restantes
        for checkbox in checkboxes:
            if id(checkbox) not in processed_checkboxes:
                structured_results["individual_checkboxes"].append(checkbox)
        
        # 4. Créer un format simplifié des valeurs
        form_values = {}
        
        # Ajouter les réponses aux questions
        for question in structured_results["questions"]:
            if question["answer"]:
                form_values[question["text"]] = question["answer"]
        
        # Ajouter les cases individuelles cochées
        for checkbox in structured_results["individual_checkboxes"]:
            if checkbox.get("checked", False) and checkbox.get("label"):
                form_values[checkbox["label"]] = "Oui"
        
        # 5. Mettre à jour les résultats
        results["structured_results"] = structured_results
        results["form_values"] = form_values

    def _validate_global_checkbox_results(self, results: Dict[str, Any]) -> None:
        """
        Valide et corrige les incohérences au niveau global.
        Version corrigée pour éviter l'erreur numpy.
        
        Args:
            results: Résultats d'extraction à corriger in-place
        """
        checkboxes = results.get("checkboxes", [])
        if not checkboxes:
            return
        
        # 1. Vérifier le ratio global de cases cochées
        checked_count = sum(1 for cb in checkboxes if cb.get("checked", False))
        total_count = len(checkboxes)
        checked_ratio = checked_count / total_count if total_count > 0 else 0
        
        # Si plus de 40% des cases sont cochées globalement, c'est suspect
        if checked_ratio > 0.4 and checked_count > 5:
            logger.warning(f"Trop de cases cochées globalement ({checked_count}/{total_count}), "
                        f"application d'une correction globale")
            
            # Garder seulement les cases cochées les plus confiantes
            sorted_checked = sorted(
                [cb for cb in checkboxes if cb.get("checked", False)],
                key=lambda cb: cb.get("confidence", 0),
                reverse=True
            )
            
            # Calculer un seuil adaptatif basé sur la distribution des confiances
            # S'assurer que les confidences sont des nombres flottants simples
            confidences = [float(cb.get("confidence", 0)) for cb in sorted_checked]
            
            if confidences:
                try:
                    # Utiliser la médiane + écart-type comme seuil adaptatif
                    # Calculer manuellement pour éviter les problèmes numpy
                    confidences.sort()
                    if len(confidences) % 2 == 0:
                        median_conf = (confidences[len(confidences)//2-1] + confidences[len(confidences)//2])/2
                    else:
                        median_conf = confidences[len(confidences)//2]
                    
                    # Calculer l'écart-type manuellement
                    mean_conf = sum(confidences) / len(confidences)
                    variance = sum((x - mean_conf) ** 2 for x in confidences) / len(confidences)
                    std_conf = variance ** 0.5 if variance > 0 else 0.1
                    
                    # Seuil adaptatif
                    adaptive_threshold = median_conf + 0.2 * std_conf
                    
                    # Limiter le nombre de cases cochées
                    max_checked = min(max(3, total_count // 5), 15)  # Entre 3 et 15, ~20% max
                    
                    # Deux stratégies : par seuil ou par nombre max
                    for i, checkbox in enumerate(checkboxes):
                        if checkbox.get("checked", False):
                            in_top_n = any(id(checkbox) == id(cb) for cb in sorted_checked[:max_checked])
                            above_threshold = checkbox.get("confidence", 0) >= adaptive_threshold
                            
                            # Si la case ne satisfait ni le classement ni le seuil
                            if not (in_top_n or above_threshold):
                                # Modifier l'état
                                checkbox["checked"] = False
                                checkbox["auto_corrected"] = True
                
                except Exception as e:
                    # En cas d'erreur, approche plus simple
                    logger.warning(f"Erreur dans le calcul adaptatif: {e}, utilisation d'une approche simplifiée")
                    
                    # Garder seulement les N cases les plus confiantes
                    max_checked = min(max(3, total_count // 5), 15)
                    
                    # Marquer toutes les cases comme non cochées d'abord
                    for checkbox in checkboxes:
                        if checkbox.get("checked", False):
                            checkbox["checked"] = False
                            checkbox["auto_corrected"] = True
                    
                    # Puis marquer les top N comme cochées
                    for i, checkbox in enumerate(sorted_checked[:max_checked]):
                        checkbox["checked"] = True
                        if "auto_corrected" in checkbox:
                            del checkbox["auto_corrected"]
        
        # 2. Corriger les incohérences dans les groupes Oui/Non
        try:
            # Regrouper les cases par proximité
            groups = self._group_checkboxes_by_proximity(checkboxes)
            
            for group in groups:
                # Chercher les paires Oui/Non dans le groupe
                oui_boxes = [cb for cb in group if cb.get("label", "").lower() in ["oui", "yes"]]
                non_boxes = [cb for cb in group if cb.get("label", "").lower() in ["non", "no"]]
                
                if len(oui_boxes) == 1 and len(non_boxes) == 1:
                    # Si les deux sont cochées, c'est incohérent
                    if oui_boxes[0].get("checked", False) and non_boxes[0].get("checked", False):
                        # Garder celle avec la plus haute confiance
                        if oui_boxes[0].get("confidence", 0) > non_boxes[0].get("confidence", 0):
                            non_boxes[0]["checked"] = False
                            non_boxes[0]["auto_corrected"] = True
                        else:
                            oui_boxes[0]["checked"] = False
                            oui_boxes[0]["auto_corrected"] = True
        except Exception as e:
            logger.warning(f"Erreur dans la correction des incohérences Oui/Non: {e}")

    def _group_checkboxes_by_proximity(self, checkboxes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Regroupe les cases à cocher par proximité spatiale.
        
        Args:
            checkboxes: Liste des cases à cocher
            
        Returns:
            Liste de groupes (chaque groupe étant une liste de cases)
        """
        if not checkboxes:
            return []
        
        # Trier par page puis par position y
        sorted_boxes = sorted(checkboxes, key=lambda cb: (cb.get("page", 0), cb.get("bbox", [0, 0, 0, 0])[1]))
        
        groups = []
        current_group = [sorted_boxes[0]]
        current_page = sorted_boxes[0].get("page", 0)
        last_y = sorted_boxes[0].get("bbox", [0, 0, 0, 0])[1]
        
        # Seuil de proximité en pixels
        proximity_threshold = 50
        
        for checkbox in sorted_boxes[1:]:
            page = checkbox.get("page", 0)
            bbox = checkbox.get("bbox", [0, 0, 0, 0])
            y = bbox[1]
            
            # Nouvelle page ou trop éloigné verticalement
            if page != current_page or abs(y - last_y) > proximity_threshold:
                # Sauvegarder le groupe courant et en commencer un nouveau
                groups.append(current_group)
                current_group = [checkbox]
                current_page = page
            else:
                # Ajouter à la groupe courant
                current_group.append(checkbox)
            
            last_y = y
        
        # Ajouter le dernier groupe
        if current_group:
            groups.append(current_group)
        
        return groups

    def _apply_strict_filtering(self, checkboxes: List[Dict[str, Any]], page_num: int) -> List[Dict[str, Any]]:
        """
        Applique des règles de filtrage strict pour réduire les faux positifs.
        
        Args:
            checkboxes: Liste des cases à cocher détectées
            page_num: Numéro de la page
            
        Returns:
            Liste filtrée des cases à cocher
        """
        if not checkboxes:
            return []
        
        # 1. Limitation du nombre de cases cochées
        checked_boxes = [cb for cb in checkboxes if cb.get("checked", False)]
        unchecked_boxes = [cb for cb in checkboxes if not cb.get("checked", False)]
        
        # Si plus de 50% des cases sont cochées, c'est suspect
        checked_ratio = len(checked_boxes) / len(checkboxes) if checkboxes else 0
        
        if checked_ratio > 0.5 and len(checked_boxes) > 3:
            logger.warning(f"Page {page_num}: Trop de cases cochées ({len(checked_boxes)}/{len(checkboxes)}), "
                        f"application d'un filtrage plus strict")
            
            # Ne garder que les cases cochées avec la plus haute confiance
            checked_boxes.sort(key=lambda cb: cb.get("confidence", 0), reverse=True)
            max_checked = max(2, len(checkboxes) // 4)  # Limiter à 25% ou au moins 2
            filtered_checked = checked_boxes[:max_checked]
            
            # Recombiner avec les cases non cochées
            return filtered_checked + unchecked_boxes
        
        # 2. Vérification de cohérence pour les groupes Oui/Non
        final_boxes = []
        
        # Regrouper par position Y approximative (même ligne)
        y_groups = {}
        
        for checkbox in checkboxes:
            bbox = checkbox.get("bbox", [0, 0, 0, 0])
            y_center = (bbox[1] + bbox[3]) / 2
            # Arrondir pour regrouper
            y_group = round(y_center / 20) * 20
            
            if y_group not in y_groups:
                y_groups[y_group] = []
                
            y_groups[y_group].append(checkbox)
        
        # Vérifier la cohérence dans chaque groupe
        for y_group, group_boxes in y_groups.items():
            # Chercher les paires Oui/Non
            oui_boxes = [cb for cb in group_boxes if cb.get("label", "").lower() in ["oui", "yes"]]
            non_boxes = [cb for cb in group_boxes if cb.get("label", "").lower() in ["non", "no"]]
            
            if len(oui_boxes) == 1 and len(non_boxes) == 1:
                # C'est une paire Oui/Non, vérifier qu'une seule est cochée
                both_checked = oui_boxes[0].get("checked", False) and non_boxes[0].get("checked", False)
                
                if both_checked:
                    # Contradiction! Choisir la plus confiante
                    if oui_boxes[0].get("confidence", 0) > non_boxes[0].get("confidence", 0):
                        non_boxes[0]["checked"] = False
                    else:
                        oui_boxes[0]["checked"] = False
                
                # Ajouter la paire
                final_boxes.extend(oui_boxes + non_boxes)
            else:
                # Pas une paire Oui/Non claire, ajouter tel quel
                final_boxes.extend(group_boxes)
        
        return final_boxes
    
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
                label = self._find_closest_text_improved(page_texts, bbox)
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
        Version améliorée de la détection visuelle des cases à cocher.
        
        Args:
            gray_img: Image en niveaux de gris
            page_num: Numéro de la page
            enhance: Activer les améliorations
            
        Returns:
            Liste des cases à cocher détectées
        """
        try:
            checkboxes = []
            
            # 1. Prétraitement multi-étapes pour améliorer la détection des bords
            # Amélioration 1: Égalisation d'histogramme adaptative
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(gray_img)
            
            # Filtre bilatéral amélioré (paramètres ajustés)
            smooth = cv2.bilateralFilter(equalized, 7, 100, 100)
            
            # 2. Détection des bords avec Canny modifié
            # Calculer les seuils adaptatifs basés sur l'histogramme de l'image
            med_val = np.median(smooth)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * med_val))
            upper = int(min(255, (1.0 + sigma) * med_val))
            
            edges = cv2.Canny(smooth, lower, upper)
            
            # 3. Amélioration: Détecter les lignes horizontales et verticales séparément
            horizontal = self._detect_lines(smooth.copy(), True, enhance)
            vertical = self._detect_lines(smooth.copy(), False, enhance)
            
            # Combiner les lignes détectées avec les bords
            combined_edges = cv2.bitwise_or(edges, cv2.bitwise_or(horizontal, vertical))
            
            # 4. Fermeture morphologique pour connecter les contours interrompus
            # Kernel plus grand pour mieux connecter les contours
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
            
            # 5. Détection des contours avec hiérarchie
            contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            # 6. Filtrer les contours pour trouver les cases à cocher potentielles
            for i, contour in enumerate(contours):
                # Amélioration: Utiliser la hiérarchie des contours
                # Les cases à cocher ont souvent un contour parent (externe)
                has_parent = hierarchy[0][i][3] != -1
                
                # Approcher le contour par un polygone avec précision améliorée
                epsilon = 0.03 * cv2.arcLength(contour, True)  # Précision accrue
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Vérifier si c'est un quadrilatère (4 côtés comme un carré/rectangle)
                is_quadrilateral = len(approx) == 4
                
                # Récupérer le rectangle englobant
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculer diverses caractéristiques
                area = cv2.contourArea(contour)
                rect_area = w * h
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Amélioration: Filtres plus précis
                # 1. Taille appropriée avec plage étendue
                valid_size = self.min_size-2 <= w <= self.max_size+2 and self.min_size-2 <= h <= self.max_size+2
                
                # 2. Forme carrée ou presque avec tolérance ajustée
                is_square_like = 0.7 <= aspect_ratio <= 1.3
                
                # 3. Remplissage approprié (amélioré)
                fill_ratio = float(area) / rect_area if rect_area > 0 else 0
                valid_fill = 0.4 <= fill_ratio <= 0.98
                
                # 4. Analyse des angles pour détecter les formes rectangulaires
                corners_ok = is_quadrilateral
                if not is_quadrilateral and len(approx) > 4:
                    # Chercher des points qui forment approximativement des angles droits
                    angles = []
                    for i in range(len(approx)):
                        pt1 = approx[i][0]
                        pt2 = approx[(i+1) % len(approx)][0]
                        pt3 = approx[(i+2) % len(approx)][0]
                        
                        # Calculer les vecteurs
                        v1 = [pt2[0] - pt1[0], pt2[1] - pt1[1]]
                        v2 = [pt3[0] - pt2[0], pt3[1] - pt2[1]]
                        
                        # Calculer l'angle (produit scalaire)
                        dot = v1[0]*v2[0] + v1[1]*v2[1]
                        mag1 = (v1[0]**2 + v1[1]**2)**0.5
                        mag2 = (v2[0]**2 + v2[1]**2)**0.5
                        
                        if mag1 * mag2 > 0:
                            cos_angle = dot / (mag1 * mag2)
                            cos_angle = max(-1, min(1, cos_angle))  # Assurer que c'est dans [-1, 1]
                            angle = math.acos(cos_angle) * 180 / math.pi
                            angles.append(angle)
                    
                    # Compter les angles proches de 90°
                    right_angles = sum(1 for angle in angles if 80 <= angle <= 100)
                    corners_ok = right_angles >= 2
                
                # Amélioration: combiner tous les critères avec pondération
                # Certains critères sont plus importants que d'autres
                if valid_size and is_square_like and valid_fill:
                    # Calculer un score de confiance basé sur plusieurs facteurs
                    square_factor = 1 - abs(aspect_ratio - 1) / 0.3  # 1 pour un carré parfait
                    
                    # Amélioration: Analyse détaillée de la région
                    region = gray_img[y:y+h, x:x+w]
                    is_checkbox_like, region_score = self._analyze_visual_checkbox_improved(region)
                    
                    # Score combiné avec pondération ajustée
                    base_score = (
                        square_factor * 0.3 + 
                        fill_ratio * 0.2 + 
                        region_score * 0.3 + 
                        (0.2 if corners_ok else 0.0)
                    )
                    
                    # Bonus pour caractéristiques spécifiques
                    # Bonus pour les quadrilatères
                    if is_quadrilateral:
                        base_score *= 1.1
                    
                    # Bonus pour les formes qui ressemblent vraiment à des cases
                    if is_checkbox_like:
                        base_score *= 1.2
                    
                    # Limiter à 0.95 (garder en dessous de la détection symbolique)
                    confidence_score = min(0.95, base_score)
                    
                    # Ajouter seulement si la confiance est suffisante
                    if confidence_score >= 0.55:  # Seuil légèrement abaissé pour augmenter la détection
                        checkbox = {
                            "bbox": [x, y, x+w, y+h],
                            "page": page_num,
                            "confidence": confidence_score,
                            "method": "vision"
                        }
                        checkboxes.append(checkbox)
            
            # 7. Détection complémentaire basée sur les intersections de lignes
            if enhance and len(checkboxes) < 15:  # Augmenté pour être plus inclusif
                line_checkboxes = self._detect_by_line_intersections(smooth, gray_img, page_num)
                checkboxes.extend(line_checkboxes)
            
            # 8. Appliquer une déduplication améliorée
            return self._deduplicate_checkboxes_improved(checkboxes)
            
        except Exception as e:
            logger.error(f"Erreur détection visuelle: {e}")
            return []
        
    def _deduplicate_checkboxes_improved(self, checkboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Version améliorée de la déduplication des cases détectées.
        
        Args:
            checkboxes: Liste des cases à cocher détectées
            
        Returns:
            Liste dédupliquée des cases à cocher
        """
        if not checkboxes:
            return []
        
        # 1. Trier par confiance décroissante
        sorted_boxes = sorted(checkboxes, key=lambda cb: cb.get("confidence", 0), reverse=True)
        
        # 2. Utiliser un algorithme de clustering pour regrouper les détections similaires
        # Initialiser les groupes
        groups = []
        
        for checkbox in sorted_boxes:
            bbox = checkbox.get("bbox", [0, 0, 0, 0])
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Vérifier si la case appartient à un groupe existant
            found_group = False
            
            for group in groups:
                # Calculer la distance moyenne aux cases du groupe
                distances = []
                for existing in group:
                    ex_bbox = existing.get("bbox", [0, 0, 0, 0])
                    ex_center_x = (ex_bbox[0] + ex_bbox[2]) / 2
                    ex_center_y = (ex_bbox[1] + ex_bbox[3]) / 2
                    
                    # Distance euclidienne
                    distance = ((center_x - ex_center_x) ** 2 + (center_y - ex_center_y) ** 2) ** 0.5
                    distances.append(distance)
                
                avg_distance = sum(distances) / len(distances) if distances else float('inf')
                
                # Si la distance moyenne est inférieure au seuil, ajouter au groupe
                if avg_distance < 20:  # Seuil de 20 pixels
                    group.append(checkbox)
                    found_group = True
                    break
            
            # Si aucun groupe existant ne convient, créer un nouveau groupe
            if not found_group:
                groups.append([checkbox])
        
        # 3. Sélectionner la meilleure case de chaque groupe
        deduplicated = []
        
        for group in groups:
            if len(group) == 1:
                # Groupe avec une seule case, l'ajouter directement
                deduplicated.append(group[0])
            else:
                # Filtrer d'abord par méthode: privilégier les méthodes plus fiables
                symbol_candidates = [cb for cb in group if cb.get("method") == "symbol"]
                if symbol_candidates:
                    # Priorité aux détections par symbole
                    best_candidate = max(symbol_candidates, key=lambda cb: cb.get("confidence", 0))
                else:
                    # Choisir la meilleure détection visuelle
                    best_candidate = max(group, key=lambda cb: cb.get("confidence", 0))
                
                deduplicated.append(best_candidate)
        
        return deduplicated
            
    def _analyze_visual_checkbox_improved(self, region: np.ndarray) -> Tuple[bool, float]:
        """
        Analyse améliorée pour déterminer si une région ressemble à une case à cocher.
        
        Args:
            region: Image de la région à analyser
            
        Returns:
            Tuple (est_une_case, score)
        """
        try:
            if region is None or region.size == 0 or region.shape[0] < 5 or region.shape[1] < 5:
                return False, 0.3
            
            # 1. Prétraitement de l'image
            # Égalisation d'histogramme pour améliorer le contraste
            if region.shape[0] > 10 and region.shape[1] > 10:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                enhanced = clahe.apply(region)
            else:
                enhanced = cv2.equalizeHist(region)
            
            # 2. Calculer les gradients
            sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # 3. Amélioration: Détecter les bords avec Canny
            edges = cv2.Canny(enhanced, 50, 150)
            
            # 4. Caractéristiques d'une case à cocher
            h, w = region.shape[:2]
            
            # A. Définir les masques pour bords et centre de manière plus précise
            # Calculer border_width adaptatif
            border_width = max(1, min(w, h) // 5)
            
            # Créer les masques
            border_mask = np.zeros_like(region)
            border_mask[:border_width, :] = 1  # Haut
            border_mask[-border_width:, :] = 1  # Bas
            border_mask[:, :border_width] = 1  # Gauche
            border_mask[:, -border_width:] = 1  # Droite
            
            center_mask = 1 - border_mask
            
            # B. Caractéristiques de bord-centre améliorées
            # Gradient moyen sur les bords
            border_gradient = np.mean(gradient_magnitude * border_mask)
            # Gradient moyen au centre
            center_gradient = np.mean(gradient_magnitude * center_mask)
            # Intensité moyenne du centre et des bords
            border_intensity = np.mean(enhanced * border_mask)
            center_intensity = np.mean(enhanced * center_mask)
            
            # C. Calculer le contraste bord/centre (important pour les cases à cocher vides)
            # Pour les cases vides, le centre est généralement plus clair que les bords
            intensity_ratio = center_intensity / border_intensity if border_intensity > 0 else 1
            gradient_ratio = border_gradient / center_gradient if center_gradient > 0 else 10
            
            # D. Analyse d'histogramme améliorée
            # Histogramme à plus haute résolution pour mieux capturer la bimodalité
            hist = cv2.calcHist([enhanced], [0], None, [64], [0, 256])
            hist_normalized = hist / np.sum(hist)
            
            # Détection plus sophistiquée des pics et creux
            peaks = []
            valleys = []
            
            for i in range(1, len(hist_normalized)-1):
                if hist_normalized[i] > hist_normalized[i-1] and hist_normalized[i] > hist_normalized[i+1]:
                    peaks.append((i, hist_normalized[i][0]))
                if hist_normalized[i] < hist_normalized[i-1] and hist_normalized[i] < hist_normalized[i+1]:
                    valleys.append((i, hist_normalized[i][0]))
            
            # Trier par hauteur
            peaks.sort(key=lambda x: x[1], reverse=True)
            
            # Calculer les caractéristiques de l'histogramme
            bimodality_score = 0.0
            if len(peaks) >= 2:
                # Distance entre les deux pics principaux
                peak_distance = abs(peaks[0][0] - peaks[1][0]) / 64
                peak_heights = peaks[0][1] + peaks[1][1]
                
                # La bimodalité est élevée si les pics sont éloignés et prononcés
                bimodality_score = peak_distance * peak_heights * 2.0
            
            # 5. Détecter les contours dans l'image binaire
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 6. Analyser les formes détectées
            contour_score = 0.0
            if contours:
                # Trouver le contour principal
                main_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(main_contour)
                rect_area = w * h
                
                # Calculer la solidité (remplissage)
                solidity = contour_area / rect_area if rect_area > 0 else 0
                
                # Les cases à cocher ont souvent une solidité modérée (ni trop pleine ni trop vide)
                if 0.2 <= solidity <= 0.9:
                    contour_score = 0.5 + (0.5 - abs(solidity - 0.5)) * 0.8
                else:
                    contour_score = 0.3
                
                # Vérifier si le contour est approximativement rectangulaire
                epsilon = 0.04 * cv2.arcLength(main_contour, True)
                approx = cv2.approxPolyDP(main_contour, epsilon, True)
                if len(approx) == 4:
                    contour_score += 0.3
            
            # 7. Calculer un score combiné avec des pondérations améliorées
            edge_score = min(1.0, np.sum(edges > 0) / (edges.size * 0.2))
            gradient_score = min(1.0, gradient_ratio / 15.0)
            intensity_score = min(1.0, intensity_ratio if intensity_ratio < 1.2 else 2.4 - intensity_ratio)
            
            # 8. Score final combinant tous les indicateurs
            final_score = (
                edge_score * 0.2 +
                gradient_score * 0.25 +
                intensity_score * 0.2 +
                bimodality_score * 0.15 +
                contour_score * 0.2
            )
            
            # Déterminer si c'est vraiment une case à cocher
            is_checkbox = final_score > 0.5
            
            return is_checkbox, min(1.0, final_score)
        
        except Exception as e:
            logger.debug(f"Erreur analyse visuelle case: {e}")
            return False, 0.3
    
    def _detect_by_line_intersections(self, gray_img: np.ndarray, orig_img: np.ndarray, page_num: int) -> List[Dict[str, Any]]:
        """
        Nouvelle méthode de détection basée sur les intersections de lignes.
        Utile pour les cases à cocher qui sont formées par des lignes perpendiculaires.
        
        Args:
            gray_img: Image prétraitée en niveaux de gris
            orig_img: Image originale
            page_num: Numéro de la page
            
        Returns:
            Liste des cases à cocher détectées
        """
        try:
            # 1. Détecter séparément les lignes horizontales et verticales
            horizontal = self._detect_lines(gray_img, True, True)
            vertical = self._detect_lines(gray_img, False, True)
            
            # 2. Trouver les points d'intersection
            intersections = cv2.bitwise_and(horizontal, vertical)
            
            # 3. Dilatation pour renforcer les intersections
            kernel = np.ones((3, 3), np.uint8)
            intersections = cv2.dilate(intersections, kernel, iterations=1)
            
            # 4. Détecter les composantes connexes dans l'image d'intersection
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersections)
            
            checkboxes = []
            
            # 5. Analyser chaque composante connexe
            for i in range(1, num_labels):  # Commencer à 1 pour sauter le fond
                # Récupérer les statistiques de la composante
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                
                # Filtrer par taille
                if self.min_size-2 <= max(w, h) <= self.max_size+5 and self.min_size-2 <= min(w, h):
                    # Calculer un score de confiance basé sur la forme
                    aspect_ratio = float(w) / h if h > 0 else 0
                    
                    # Les cases carrées ont un meilleur score
                    square_factor = 1 - min(0.8, abs(aspect_ratio - 1))
                    
                    # Analyser la région pour confirmer qu'il s'agit d'une case
                    region = orig_img[y:y+h, x:x+w]
                    is_checkbox, region_score = self._analyze_visual_checkbox_improved(region)
                    
                    # Score combiné
                    confidence = (square_factor * 0.4 + region_score * 0.6) * 0.9  # 0.9 pour rester sous la détection symbolique
                    
                    # Ajouter si score suffisant
                    if confidence >= 0.55 and is_checkbox:
                        checkbox = {
                            "bbox": [x, y, x+w, y+h],
                            "page": page_num,
                            "confidence": confidence,
                            "method": "lines"
                        }
                        checkboxes.append(checkbox)
            
            return checkboxes
        
        except Exception as e:
            logger.error(f"Erreur détection par intersections: {e}")
            return []
    
    def _analyze_visual_checkbox(self, region: np.ndarray) -> float:
        """
        Analyse si une région ressemble visuellement à une case à cocher.
        
        Args:
            region: Image de la région à analyser
            
        Returns:
            Score entre 0 et 1 indiquant la probabilité que ce soit une case à cocher
        """
        try:
            if region is None or region.size == 0:
                return 0.0
                
            # Si l'image est trop petite, pas assez d'info pour analyser
            h, w = region.shape[:2]
            if h < 5 or w < 5:
                return 0.5  # Score neutre
            
            # 1. Détecter les bords
            edges = cv2.Canny(region, 50, 150)
            
            # 2. Calculer le gradient pour détecter les changements d'intensité
            sobelx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # 3. Caractéristiques d'une case à cocher:
            
            # A. Présence de bords (ratio de pixels de bord)
            edge_ratio = np.sum(edges > 0) / edges.size
            
            # B. Gradient fort sur les bords, faible au centre
            # Définir les régions
            border_width = max(1, min(w, h) // 6)
            
            # Créer les masques
            border_mask = np.zeros_like(region)
            border_mask[:border_width, :] = 1  # Haut
            border_mask[-border_width:, :] = 1  # Bas
            border_mask[:, :border_width] = 1  # Gauche
            border_mask[:, -border_width:] = 1  # Droite
            
            center_mask = 1 - border_mask
            
            # Calculer les gradients moyens
            border_gradient = np.mean(gradient_magnitude * border_mask)
            center_gradient = np.mean(gradient_magnitude * center_mask)
            
            # Rapport gradient bord/centre (élevé pour une case à cocher vide)
            gradient_ratio = border_gradient / center_gradient if center_gradient > 0 else 10
            
            # C. Histogramme bimodal (pixels sombres pour les bords, clairs pour l'intérieur)
            hist = cv2.calcHist([region], [0], None, [32], [0, 256])
            hist_normalized = hist / np.sum(hist)
            
            # Calcul de la bimodalité (élevée si l'histogramme a deux pics distincts)
            peaks = [i for i in range(1, len(hist_normalized)-1) 
                    if hist_normalized[i] > hist_normalized[i-1] and hist_normalized[i] > hist_normalized[i+1]]
            
            histogram_bimodality = 0.0
            if len(peaks) >= 2:
                # Distance entre les deux pics les plus élevés
                peaks_sorted = sorted(peaks, key=lambda i: hist_normalized[i], reverse=True)
                if len(peaks_sorted) >= 2:
                    peak_distance = abs(peaks_sorted[0] - peaks_sorted[1]) / 32
                    histogram_bimodality = peak_distance * (hist_normalized[peaks_sorted[0]] + hist_normalized[peaks_sorted[1]]) / 2
            
            # 4. Combiner les scores
            # Les cases à cocher typiques ont:
            # - Un ratio de bords modéré
            # - Un fort ratio de gradient bord/centre
            # - Une bimodalité d'histogramme modérée
            
            edge_score = min(1.0, edge_ratio * 5) if edge_ratio < 0.4 else max(0.0, 2.0 - edge_ratio * 5)
            gradient_score = min(1.0, gradient_ratio / 10)
            bimodality_score = min(1.0, histogram_bimodality * 3)
            
            # Score combiné
            combined_score = (edge_score * 0.5 + gradient_score * 0.3 + bimodality_score * 0.2)
            
            return combined_score
            
        except Exception as e:
            logger.debug(f"Erreur analyse visuelle case: {e}")
            return 0.5  # Score neutre en cas d'erreur

    def _detect_lines(self, binary_img: np.ndarray, is_horizontal: bool, enhanced: bool) -> np.ndarray:
        """
        Version améliorée pour détecter les lignes horizontales ou verticales.
        
        Args:
            binary_img: Image binaire
            is_horizontal: True pour lignes horizontales, False pour verticales
            enhanced: Activer les améliorations supplémentaires
            
        Returns:
            Image binaire avec les lignes détectées
        """
        # Paramètres améliorés pour la détection des lignes
        if is_horizontal:
            # Pour les lignes horizontales
            if enhanced:
                # Utiliser une longueur adaptative plus fine
                kernel_length = max(5, binary_img.shape[1] // 25)
            else:
                kernel_length = binary_img.shape[1] // 30
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        else:
            # Pour les lignes verticales
            if enhanced:
                # Utiliser une longueur adaptative plus fine
                kernel_length = max(5, binary_img.shape[0] // 25)
            else:
                kernel_length = binary_img.shape[0] // 30
                
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        
        # Amélioration: prétraitement de l'image
        # Appliquer un seuil adaptatif pour améliorer la détection des lignes fines
        if enhanced:
            # Réduction du bruit
            denoised = cv2.GaussianBlur(binary_img, (3, 3), 0)
            
            # Seuil adaptatif pour mieux extraire les structures
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            # Version simplifiée
            _, thresh = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Appliquer l'opération morphologique d'ouverture
        detected = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Dilatation légère pour améliorer la connexion
        if enhanced:
            detected = cv2.dilate(detected, np.ones((2, 2), np.uint8), iterations=1)
        
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
    
    def _is_checkbox_checked(self, checkbox_img: np.ndarray) -> Tuple[bool, float]:
        """
        Version améliorée pour déterminer si une case est cochée,
        avec retour de la confiance de la détection.
        
        Args:
            checkbox_img: Image de la case à cocher
            
        Returns:
            Tuple (is_checked, confidence_score)
        """
        try:
            # S'assurer que l'image n'est pas vide
            if checkbox_img is None or checkbox_img.size == 0:
                return False, 0.0
            
            # Si l'image est en couleur, la convertir en niveaux de gris
            if len(checkbox_img.shape) > 2:
                gray = cv2.cvtColor(checkbox_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = checkbox_img
            
            # Amélioration 1: Prétraitement avec réduction de bruit et amélioration de contraste
            # Appliquer une égalisation d'histogramme locale pour améliorer le contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Réduction de bruit avec filtre bilatéral (préserve les bords)
            denoised = cv2.bilateralFilter(enhanced, 5, 75, 75)
            
            # Amélioration 2: Utiliser plusieurs méthodes de binarisation et les combiner
            # Binarisation avec Otsu (inversé)
            _, binary_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Binarisation adaptative, plus sensible aux marques locales
            binary_adaptive = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Amélioration 3: Combiner les résultats des deux méthodes de binarisation
            # Intersection des deux méthodes pour les marques solides
            binary_solid = cv2.bitwise_and(binary_otsu, binary_adaptive)
            
            # Union pour capturer les marques plus faibles mais consistantes
            binary_all = cv2.bitwise_or(binary_otsu, binary_adaptive)
            
            # Paramètres pour l'analyse
            height, width = binary_solid.shape
            total_pixels = height * width
            
            # Amélioration 4: Analyse multi-échelle de la densité des marques
            
            # Analyse globale: ratio de pixels noirs
            solid_fill_ratio = np.sum(binary_solid > 0) / total_pixels if total_pixels > 0 else 0
            all_fill_ratio = np.sum(binary_all > 0) / total_pixels if total_pixels > 0 else 0
            
            # Analyse du centre: définir une région centrale
            center_margin = max(2, min(width, height) // 4)
            
            # Si l'image est assez grande pour l'analyse par régions
            if width > 2*center_margin and height > 2*center_margin:
                # Analyser le centre
                center_region = binary_solid[
                    center_margin:height-center_margin, 
                    center_margin:width-center_margin
                ]
                center_area = (height - 2*center_margin) * (width - 2*center_margin)
                center_fill_ratio = np.sum(center_region > 0) / center_area if center_area > 0 else 0
                
                # Calculer la concentration au centre
                center_concentration = center_fill_ratio / solid_fill_ratio if solid_fill_ratio > 0 else 0
            else:
                center_concentration = 1.0
                center_fill_ratio = solid_fill_ratio
            
            # Amélioration 5: Analyse morphologique des formes
            # Détecter les contours importants
            contours, _ = cv2.findContours(binary_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrer les contours insignifiants
            min_contour_area = total_pixels * 0.01  # 1% de la surface totale
            significant_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
            
            # Analyser la forme des contours
            shape_score = 0.0
            if significant_contours:
                for contour in significant_contours:
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    # Calculer la circularité (1 pour un cercle parfait, valeurs plus grandes pour formes complexes)
                    circularity = (perimeter**2) / (4 * np.pi * area) if area > 0 else 0
                    
                    # Les croix et les coches ont une circularité élevée (forme complexe)
                    if circularity > 2.0:
                        shape_score += min(1.0, circularity / 10)
            
            # Normaliser le score de forme
            shape_score = min(1.0, shape_score)
            
            # Amélioration 6: Calcul de score composite
            # Pondérer les différents indicateurs
            checked_score = (
                solid_fill_ratio * 0.3 +           # Remplissage solide
                all_fill_ratio * 0.1 +             # Remplissage total
                center_fill_ratio * 0.2 +          # Densité au centre
                center_concentration * 0.2 +       # Concentration au centre
                shape_score * 0.2                  # Complexité des formes
            )
            
            # Amélioration 7: Détection adaptative basée sur l'intensité globale
            # Calculer l'intensité moyenne pour ajuster le seuil
            mean_intensity = np.mean(gray) / 255.0
            
            # Ajuster les seuils selon la clarté générale de l'image
            # Cas clairs: besoin de moins de marques pour être considéré comme coché
            # Cas sombres: besoin de plus de marques
            intensity_factor = 1.0 + (0.5 - mean_intensity) * 0.5
            
            # Définition dynamique du seuil selon la clarté
            threshold_base = 0.25
            threshold = threshold_base * intensity_factor
            
            # Ajustement final de la confiance
            confidence = min(1.0, checked_score * 1.5)
            
            # Amélioration 8: Détection binaire avec décision optimisée
            is_checked = checked_score >= threshold
            
            # Cas spéciaux: fortes indications malgré un score global sous le seuil
            if not is_checked:
                # Forte complexité de forme + remplissage central
                if shape_score > 0.5 and center_fill_ratio > 0.15:
                    is_checked = True
                    confidence = max(confidence, 0.65)
                
                # Ou remplissage central très fort
                elif center_fill_ratio > 0.3:
                    is_checked = True
                    confidence = max(confidence, 0.7)
            
            return is_checked, confidence
            
        except Exception as e:
            logger.debug(f"Erreur analyse case cochée: {e}")
            return False, 0.0
        
    def _deduplicate_checkboxes(self, checkboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applique une déduplication stricte pour éliminer les détections multiples.
        
        Args:
            checkboxes: Liste des cases à cocher détectées
            
        Returns:
            Liste dédupliquée des cases à cocher
        """
        if not checkboxes:
            return []
        
        # Trier par confiance décroissante
        sorted_boxes = sorted(checkboxes, key=lambda cb: cb.get("confidence", 0), reverse=True)
        
        deduplicated = []
        used_positions = []
        
        for checkbox in sorted_boxes:
            bbox = checkbox.get("bbox", [0, 0, 0, 0])
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Vérifier s'il y a une case similaire déjà acceptée
            is_duplicate = False
            
            for used_pos in used_positions:
                used_x, used_y = used_pos
                
                # Calculer la distance euclidienne
                distance = ((center_x - used_x) ** 2 + (center_y - used_y) ** 2) ** 0.5
                
                # Si les centres sont proches (moins de 25 pixels), considérer comme doublon
                if distance < 25:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(checkbox)
                used_positions.append((center_x, center_y))
        
        return deduplicated
    
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
        Version avancée pour trouver le texte pertinent associé à une case à cocher.
        Filtre les étiquettes inappropriées et privilégie les textes significatifs.
        
        Args:
            page_text: Texte structuré de la page
            bbox: Rectangle englobant de la case [x1, y1, x2, y2]
            
        Returns:
            Texte le plus pertinent à associer à la case
        """
        if not page_text or "blocks" not in page_text:
            return ""
        
        # Coordonnées du centre de la case
        x1, y1, x2, y2 = bbox
        checkbox_center_x = (x1 + x2) / 2
        checkbox_center_y = (y1 + y2) / 2
        
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
                
                # Distance horizontale et verticale
                dx = line_center_x - checkbox_center_x
                dy = line_center_y - checkbox_center_y
                
                # Distance euclidienne
                distance = (dx**2 + dy**2)**0.5
                
                # Ne considérer que les textes à une distance raisonnable
                if distance < 150:
                    line_text = " ".join([span["text"] for span in line["spans"]])
                    
                    # Nettoyer le texte
                    cleaned_text = line_text
                    for symbol in ["☐", "☑", "☒", "□", "■", "▢", "▣", "▪", "▫"]:
                        cleaned_text = cleaned_text.replace(symbol, "").strip()
                    
                    if cleaned_text:
                        # Ajouter aux candidats
                        candidates.append({
                            "text": cleaned_text,
                            "distance": distance,
                            "dx": dx,
                            "dy": dy,
                            "bbox": line_bbox
                        })
        
        # Si aucun candidat, retourner chaîne vide
        if not candidates:
            return ""
        
        # Filtrage initial des candidats inappropriés
        filtered_candidates = []
        for candidate in candidates:
            text = candidate["text"].strip()
            
            # Rejeter les textes qui ressemblent à des numéros de page, heures, dates
            if re.match(r'^\d+\s*\/\s*\d+$', text):  # Format "4 / 5" (numéro de page)
                continue
            if re.match(r'^\d{1,2}:\d{2}$', text):  # Format heure "15:18"
                continue
            if re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', text):  # Format date
                continue
            
            # Garder les textes significatifs
            filtered_candidates.append(candidate)
        
        # Si après filtrage il ne reste rien, essayer les textes Oui/Non
        if not filtered_candidates:
            yes_no_candidates = [c for c in candidates if c["text"].lower() in ["oui", "non", "yes", "no"]]
            if yes_no_candidates:
                filtered_candidates = yes_no_candidates
        
        # S'il ne reste vraiment aucun candidat
        if not filtered_candidates:
            return ""
        
        # Évaluer les candidats en tenant compte de facteurs multiples
        best_candidate = None
        best_score = float('-inf')
        
        for candidate in filtered_candidates:
            # Facteurs pour le score
            dx = candidate["dx"]
            dy = candidate["dy"]
            distance = candidate["distance"]
            text = candidate["text"]
            
            # Bonus/malus selon la position relative
            position_score = 0
            
            # Position horizontale : préférence à droite (étiquette typique)
            if dx > 0 and dx < 100:  # À droite, pas trop loin
                position_score += 50
            elif dx < 0 and dx > -100:  # À gauche, pas trop loin
                position_score += 30
            
            # Position verticale : préférence même ligne
            if abs(dy) < 15:  # Même ligne approximativement
                position_score += 40
            
            # Bonus pour textes significatifs
            content_score = 0
            
            # Bonus pour Oui/Non (options classiques)
            if text.lower() in ["oui", "non", "yes", "no"]:
                content_score += 25
            
            # Bonus pour les textes qui ressemblent à des questions
            if text.endswith("?"):
                content_score += 30
            elif len(text) > 10 and len(text) < 100:  # Texte de longueur moyenne (probable question)
                content_score += 15
            
            # Pénalité pour textes très courts (sauf Oui/Non)
            if len(text) < 3 and text.lower() not in ["oui", "non", "yes", "no"]:
                content_score -= 20
            
            # Pénalité pour textes très longs
            if len(text) > 100:
                content_score -= 15
            
            # Score final combinant distance et autres facteurs
            # Distance inversée (plus c'est proche, mieux c'est)
            distance_score = 100 - min(100, distance)
            
            total_score = distance_score + position_score + content_score
            
            if total_score > best_score:
                best_score = total_score
                best_candidate = candidate
        
        # Retourner le meilleur texte
        return best_candidate["text"] if best_candidate else ""
    
    def _extract_question_for_yes_no_pair(self, yes_checkbox: Dict, no_checkbox: Dict, page_text: Dict) -> str:
        """
        Extrait la question associée à une paire de cases Oui/Non.
        
        Args:
            yes_checkbox: Case à cocher "Oui"
            no_checkbox: Case à cocher "Non"
            page_text: Texte structuré de la page
            
        Returns:
            Texte de la question, si identifié
        """
        if not page_text or "blocks" not in page_text:
            return ""
        
        # Position moyenne de la paire Oui/Non
        yes_bbox = yes_checkbox.get("bbox", [0, 0, 0, 0])
        no_bbox = no_checkbox.get("bbox", [0, 0, 0, 0])
        
        avg_x = (yes_bbox[0] + yes_bbox[2] + no_bbox[0] + no_bbox[2]) / 4
        avg_y = (yes_bbox[1] + yes_bbox[3] + no_bbox[1] + no_bbox[3]) / 4
        
        # Chercher le texte le plus pertinent au-dessus ou à gauche de la paire
        best_question = ""
        best_score = float('-inf')
        
        for block in page_text["blocks"]:
            if block["type"] != 0:
                continue
                
            for line in block["lines"]:
                line_bbox = line["bbox"]
                line_center_x = (line_bbox[0] + line_bbox[2]) / 2
                line_center_y = (line_bbox[1] + line_bbox[3]) / 2
                
                line_text = " ".join([span["text"] for span in line["spans"]])
                if not line_text or line_text.lower() in ["oui", "non", "yes", "no"]:
                    continue
                
                # Calculer un score pour ce texte comme question potentielle
                score = 0
                
                # Position: préférer les textes au-dessus ou à gauche
                dx = line_center_x - avg_x
                dy = line_center_y - avg_y
                
                # Au-dessus (plus haut sur la page = y plus petit)
                if dy < 0 and abs(dy) < 100:
                    score += 40 - abs(dy) * 0.3  # Plus proche, meilleur score
                
                # À gauche
                if dx < 0 and abs(dx) < 150:
                    score += 30 - abs(dx) * 0.15
                
                # Contenu: favoriser les textes qui ressemblent à des questions
                if line_text.endswith("?"):
                    score += 50
                elif "?" in line_text:
                    score += 30
                elif len(line_text) > 10:
                    score += 10
                
                # Éviter les textes trop courts ou trop longs
                if len(line_text) < 5:
                    score -= 20
                if len(line_text) > 150:
                    score -= 15
                
                if score > best_score:
                    best_score = score
                    best_question = line_text
        
        # Si aucune question pertinente trouvée
        if best_score < 10:
            return ""
        
        return best_question

    def _organize_checkbox_groups(self, results: Dict[str, Any]) -> None:
        """
        Organise les cases à cocher en groupes logiques par questions.
        
        Args:
            results: Résultats d'extraction à structurer in-place
        """
        checkboxes = results.get("checkboxes", [])
        if not checkboxes:
            return
        
        # 1. Nettoyer et normaliser les cases
        for checkbox in checkboxes:
            # Normaliser les libellés Oui/Non
            label = checkbox.get("label", "").strip()
            if re.match(r'^oui$|^yes$', label, re.IGNORECASE):
                checkbox["label"] = "Oui"
                checkbox["value"] = "Oui"
            elif re.match(r'^non$|^no$', label, re.IGNORECASE):
                checkbox["label"] = "Non"
                checkbox["value"] = "Non"
        
        # 2. Regrouper par proximité spatiale
        # Trier par page puis par position y
        sorted_boxes = sorted(checkboxes, key=lambda cb: (cb.get("page", 0), cb.get("bbox", [0, 0, 0, 0])[1]))
        
        # Identifier les groupes
        groups = []
        current_group = []
        current_page = None
        
        for checkbox in sorted_boxes:
            page = checkbox.get("page", 0)
            bbox = checkbox.get("bbox", [0, 0, 0, 0])
            y = bbox[1]
            
            # Nouvelle page ou trop éloigné = nouveau groupe
            if page != current_page or (current_group and abs(y - current_group[-1].get("bbox", [0, 0, 0, 0])[1]) > 50):
                if current_group:
                    groups.append(current_group)
                current_group = [checkbox]
                current_page = page
            else:
                current_group.append(checkbox)
        
        # Ajouter le dernier groupe
        if current_group:
            groups.append(current_group)
        
        # 3. Analyser chaque groupe pour identifier les questions/réponses
        structured_results = {
            "questions": [],
            "individual_checkboxes": []
        }
        
        for group in groups:
            # Identifier les paires Oui/Non
            oui_boxes = [cb for cb in group if cb.get("label", "").lower() == "oui"]
            non_boxes = [cb for cb in group if cb.get("label", "").lower() == "non"]
            
            # Si on a une paire Oui/Non
            if len(oui_boxes) == 1 and len(non_boxes) == 1:
                oui_box = oui_boxes[0]
                non_box = non_boxes[0]
                page_num = oui_box.get("page", 0)
                
                # Rechercher la question associée
                page_texts = {}  # Cache pour éviter de recalculer
                
                # Trouver le texte de la page si on ne l'a pas déjà
                if page_num not in page_texts and hasattr(self, "_current_pdf_doc"):
                    try:
                        page = self._current_pdf_doc[page_num - 1]  # 0-based index
                        page_texts[page_num] = page.get_text("dict")
                    except:
                        page_texts[page_num] = {}
                
                # Extraire la question
                question_text = self._extract_question_for_yes_no_pair(
                    oui_box, non_box, 
                    page_texts.get(page_num, {})
                )
                
                # Si pas de question trouvée, utiliser un ID par défaut
                if not question_text:
                    question_text = f"Question {len(structured_results['questions']) + 1}"
                
                # Déterminer la réponse (Oui, Non, ou aucune)
                answer = None
                if oui_box.get("checked", False) and not non_box.get("checked", False):
                    answer = "Oui"
                elif non_box.get("checked", False) and not oui_box.get("checked", False):
                    answer = "Non"
                
                # Ajouter à la structure des questions
                structured_results["questions"].append({
                    "text": question_text,
                    "answer": answer,
                    "page": page_num,
                    "checkboxes": [oui_box, non_box]
                })
            else:
                # Ajouter les cases individuelles
                for checkbox in group:
                    structured_results["individual_checkboxes"].append(checkbox)
        
        # 4. Ajouter la structure aux résultats
        results["structured_results"] = structured_results
        
        # 5. Créer un format simplifié des valeurs pour une utilisation facile
        form_values = {}
        
        # Ajouter les réponses aux questions
        for question in structured_results["questions"]:
            if question["answer"]:
                form_values[question["text"]] = question["answer"]
        
        # Ajouter les cases individuelles cochées
        for checkbox in structured_results["individual_checkboxes"]:
            if checkbox.get("checked", False) and checkbox.get("label"):
                form_values[checkbox["label"]] = "Oui"
        
        results["form_values"] = form_values
    
    def _post_process_checkboxes(self, checkboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Effectue un post-traitement pour améliorer la qualité des cases détectées.
        
        Args:
            checkboxes: Liste des cases à cocher détectées
            
        Returns:
            Liste des cases à cocher améliorée
        """
        if not checkboxes:
            return []
        
        # 1. Filtrer les cases sans étiquette cohérente
        filtered_boxes = []
        
        for checkbox in checkboxes:
            # Conserver les cases avec des étiquettes significatives
            label = checkbox.get("label", "").strip()
            
            if label:
                # Filtrer les étiquettes numériques ou formats spéciaux
                if re.match(r'^\d+\s*\/\s*\d+$', label):  # Format "4 / 5"
                    continue
                if re.match(r'^\d{1,2}:\d{2}$', label):  # Format heure
                    continue
                if re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', label):  # Format date
                    continue
                
                # Conserver la case
                filtered_boxes.append(checkbox)
            elif checkbox.get("checked", False) and checkbox.get("confidence", 0) > 0.75:
                # Conserver les cases cochées avec haute confiance même sans étiquette
                filtered_boxes.append(checkbox)
            elif len(filtered_boxes) < 5:
                # Pour les premières détections, accepter même sans étiquette
                # (pour éviter de tout filtrer si la détection d'étiquette échoue)
                filtered_boxes.append(checkbox)
        
        # 2. Fusionner les cases très proches (probables doublons avec libellés différents)
        merged_boxes = []
        used_indices = set()
        
        for i, checkbox1 in enumerate(filtered_boxes):
            if i in used_indices:
                continue
            
            bbox1 = checkbox1.get("bbox", [0, 0, 0, 0])
            center1_x = (bbox1[0] + bbox1[2]) / 2
            center1_y = (bbox1[1] + bbox1[3]) / 2
            
            # Chercher les cases similaires
            duplicates = [i]
            
            for j, checkbox2 in enumerate(filtered_boxes[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                bbox2 = checkbox2.get("bbox", [0, 0, 0, 0])
                center2_x = (bbox2[0] + bbox2[2]) / 2
                center2_y = (bbox2[1] + bbox2[3]) / 2
                
                # Distance entre les centres
                distance = ((center1_x - center2_x)**2 + (center1_y - center2_y)**2)**0.5
                
                # Si très proches, considérer comme doublons
                if distance < 20 and checkbox1.get("page") == checkbox2.get("page"):
                    duplicates.append(j)
            
            # Choisir la meilleure case parmi les doublons
            if len(duplicates) > 1:
                # Privilégier les cases avec étiquette
                labeled_boxes = [filtered_boxes[idx] for idx in duplicates if filtered_boxes[idx].get("label")]
                
                if labeled_boxes:
                    # Choisir celle avec l'étiquette la plus pertinente
                    best_box = max(labeled_boxes, key=lambda box: len(box.get("label", "")))
                else:
                    # Sans étiquette, prendre celle avec la plus haute confiance
                    best_box = max([filtered_boxes[idx] for idx in duplicates], 
                                key=lambda box: box.get("confidence", 0))
                
                merged_boxes.append(best_box)
            else:
                # Pas de doublon, ajouter directement
                merged_boxes.append(checkbox1)
            
            # Marquer tous les indices utilisés
            used_indices.update(duplicates)
        
        return merged_boxes

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

    # Filtrage avancé des étiquettes inappropriées

    def _clean_and_validate_label(self, text: str) -> str:
        """
        Nettoie et valide une étiquette candidate.
        Filtre les étiquettes inappropriées ou sans valeur informative.
        
        Args:
            text: Texte candidat pour une étiquette
            
        Returns:
            Étiquette nettoyée ou chaîne vide si inappropriée
        """
        if not text:
            return ""
        
        # Nettoyer les espaces et caractères spéciaux
        cleaned = text.strip()
        
        # Filtres basés sur des motifs réguliers
        
        # 1. Rejeter les prépositions isolées
        if re.match(r'^[àaádeèéêëdulesàauauxpourparsur]{1,3}$', cleaned.lower().replace(' ', '')):
            return ""
        
        # 2. Rejeter les numéros de page, dates et heures
        if re.match(r'^\d+\s*\/\s*\d+$', cleaned):  # Format "4 / 5" (page)
            return ""
        if re.match(r'^\d{1,2}:\d{2}$', cleaned):  # Format heure "15:18"
            return ""
        if re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', cleaned):  # Format date
            return ""
        
        # 3. Rejeter les codes et références
        if re.match(r'^[A-Z0-9]{3,}\d{2}/\d{3}[A-Z]?$', cleaned):  # Format "FIC03/006A"
            return ""
        
        # 4. Rejeter les coordonnées/contacts
        if '/' in cleaned and ('@' in cleaned or re.search(r'\d{2}\s*\d{2}\s*\d{2}\s*\d{2}\s*\d{2}', cleaned)):
            return ""
        
        # 5. Rejeter les textes trop courts sauf Oui/Non
        if len(cleaned) < 2 and cleaned.lower() not in ["oui", "non", "yes", "no"]:
            return ""
        
        # 6. Normaliser Oui/Non
        if cleaned.lower() in ["oui", "yes"]:
            return "Oui"
        if cleaned.lower() in ["non", "no"]:
            return "Non"
        
        # 7. Limiter la longueur des étiquettes très longues
        if len(cleaned) > 100:
            # Tronquer en cherchant un point ou une virgule pour une coupure propre
            for i in range(80, min(100, len(cleaned))):
                if cleaned[i] in ['.', ',', ';', ':', '!', '?']:
                    return cleaned[:i+1] + "..."
            # Si pas de ponctuation trouvée
            return cleaned[:80] + "..."
        
        return cleaned


    def _find_closest_text_improved(self, page_text: Dict, bbox: List[int]) -> str:
        """
        Version améliorée pour l'association d'étiquettes, avec meilleur filtrage.
        
        Args:
            page_text: Texte structuré de la page
            bbox: Rectangle englobant de la case [x1, y1, x2, y2]
            
        Returns:
            Texte le plus pertinent à associer à la case
        """
        if not page_text or "blocks" not in page_text:
            return ""
        
        # Coordonnées du centre de la case
        x1, y1, x2, y2 = bbox
        checkbox_center_x = (x1 + x2) / 2
        checkbox_center_y = (y1 + y2) / 2
        
        # Pour stocker tous les textes candidats
        candidates = []
        
        # 1. PHASE DE COLLECTE DES CANDIDATS
        
        # Distance maximale à considérer
        max_distance = 150
        
        # Chercher dans les blocs de texte
        for block in page_text["blocks"]:
            if block["type"] != 0:  # Ignorer les blocs non-textuels
                continue
                
            for line in block["lines"]:
                line_bbox = line["bbox"]
                line_center_x = (line_bbox[0] + line_bbox[2]) / 2
                line_center_y = (line_bbox[1] + line_bbox[3]) / 2
                
                # Distance horizontale et verticale
                dx = line_center_x - checkbox_center_x
                dy = line_center_y - checkbox_center_y
                
                # Distance euclidienne
                distance = (dx**2 + dy**2)**0.5
                
                if distance < max_distance:
                    # Récupérer le texte complet
                    line_text = " ".join([span["text"] for span in line["spans"]])
                    
                    # Nettoyer les symboles de case à cocher
                    cleaned_text = line_text
                    for symbol in ["☐", "☑", "☒", "□", "■", "▢", "▣", "▪", "▫"]:
                        cleaned_text = cleaned_text.replace(symbol, "").strip()
                    
                    if cleaned_text:
                        # Ajouter aux candidats
                        candidates.append({
                            "text": cleaned_text,
                            "distance": distance,
                            "dx": dx,
                            "dy": dy,
                            "bbox": line_bbox
                        })
        
        # Si aucun candidat, retourner chaîne vide
        if not candidates:
            return ""
        
        # 2. PHASE D'ANALYSE ET FILTRAGE
        
        # A. Première passe pour capter les paires Oui/Non
        oui_non_candidates = []
        for candidate in candidates:
            text = candidate["text"].lower().strip()
            if text in ["oui", "non", "yes", "no"]:
                oui_non_candidates.append(candidate)
        
        # Si au moins deux options trouvées et proches en Y, privilégier Oui/Non
        if len(oui_non_candidates) >= 2:
            # Trier par position Y
            oui_non_candidates.sort(key=lambda x: x["bbox"][1])
            
            # Vérifier si elles sont proches verticalement
            for i in range(len(oui_non_candidates) - 1):
                y1 = oui_non_candidates[i]["bbox"][1]
                y2 = oui_non_candidates[i+1]["bbox"][1]
                
                if abs(y1 - y2) < 30:  # Considérer comme une paire sur la même ligne
                    # Chercher celle la plus proche horizontalement de la case
                    closest = min(oui_non_candidates[i:i+2], key=lambda x: abs(x["dx"]))
                    return self._clean_and_validate_label(closest["text"])
        
        # B. Recherche de texte contextuel pertinent
        
        # Filtrage avancé: calculer un score pour chaque candidat
        scored_candidates = []
        
        for candidate in candidates:
            text = candidate["text"]
            distance = candidate["distance"]
            dx = candidate["dx"]
            dy = candidate["dy"]
            
            # a) Position: préférence à droite puis à gauche
            position_score = 0
            
            # Droite (étiquette typique)
            if 5 < dx < 100:  # Pas trop proche, pas trop loin
                position_score = 50 - abs(dx) * 0.2
            # Gauche
            elif -100 < dx < -5:
                position_score = 30 - abs(dx) * 0.15
            
            # b) Alignement vertical: préférence même ligne
            if abs(dy) < 15:
                position_score += 40
            elif abs(dy) < 30:
                position_score += 20
            
            # c) Contenu: favoriser textes informatifs
            content_score = 0
            
            # Bonus fort pour Oui/Non et réponses claires
            if text.lower() in ["oui", "non", "yes", "no"]:
                content_score += 50
            
            # Bonus pour texte formaté comme question
            if text.strip().endswith("?"):
                content_score += 40
            elif "?" in text:
                content_score += 25
            
            # Pénalité pour longueur inappropriée
            if len(text) < 2 and text.lower() not in ["oui", "non", "yes", "no"]:
                content_score -= 50  # Fortement pénaliser
            elif len(text) > 150:
                content_score -= 30  # Pénalité pour textes trop longs
            
            # d) Distance (inversée: plus proche = meilleur score)
            distance_score = 100 - min(100, distance * 0.8)
            
            # Score total pondéré
            total_score = (
                distance_score * 0.3 +
                position_score * 0.5 +  # Position a plus d'importance
                content_score * 0.2
            )
            
            # Vérification rapide pour filtrer les pires candidats
            if content_score > -30:  # Pas complètement disqualifié par le contenu
                scored_candidates.append({
                    "text": text,
                    "score": total_score
                })
        
        # Trier par score décroissant
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Prendre le meilleur candidat si score suffisant
        if scored_candidates and scored_candidates[0]["score"] > 20:
            return self._clean_and_validate_label(scored_candidates[0]["text"])
        
        # Aucun candidat adéquat
        return ""


    def _detect_and_group_yes_no_pairs(self, checkboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Détecte et regroupe les paires Oui/Non, en améliorant leurs étiquettes.
        
        Args:
            checkboxes: Liste des cases à cocher
            
        Returns:
            Liste des cases à cocher avec paires Oui/Non corrigées
        """
        if len(checkboxes) < 2:
            return checkboxes
        
        # Trier par page et position Y
        sorted_boxes = sorted(checkboxes, key=lambda cb: (cb.get("page", 0), cb.get("bbox", [0, 0, 0, 0])[1]))
        
        # Identifier les paires Oui/Non potentielles
        result = []
        i = 0
        
        while i < len(sorted_boxes):
            current = sorted_boxes[i]
            
            # Chercher une paire potentielle (case suivante proche verticalement)
            if i + 1 < len(sorted_boxes):
                next_box = sorted_boxes[i + 1]
                
                # Même page et proche verticalement
                same_page = current.get("page") == next_box.get("page")
                y_current = current.get("bbox", [0, 0, 0, 0])[1]
                y_next = next_box.get("bbox", [0, 0, 0, 0])[1]
                close_y = abs(y_current - y_next) < 30
                
                # Si les cases sont proches et ont des étiquettes Oui/Non ou pas d'étiquettes
                if same_page and close_y:
                    # Cas 1: Les deux ont déjà des étiquettes correctes Oui/Non
                    current_label = current.get("label", "").lower()
                    next_label = next_box.get("label", "").lower()
                    
                    if (current_label in ["oui", "yes"] and next_label in ["non", "no"]) or \
                    (current_label in ["non", "no"] and next_label in ["oui", "yes"]):
                        # Normaliser les étiquettes
                        if current_label in ["oui", "yes"]:
                            current["label"] = "Oui"
                            current["value"] = "Oui"
                            next_box["label"] = "Non"
                            next_box["value"] = "Non"
                        else:
                            current["label"] = "Non"
                            current["value"] = "Non"
                            next_box["label"] = "Oui"
                            next_box["value"] = "Oui"
                        
                        # Marquer comme faisant partie d'une paire
                        pair_id = f"pair_{current.get('page')}_{y_current}"
                        current["pair_id"] = pair_id
                        next_box["pair_id"] = pair_id
                        
                        # Ajouter les deux et avancer
                        result.append(current)
                        result.append(next_box)
                        i += 2
                        continue
                    
                    # Cas 2: Aucune n'a d'étiquette ou étiquettes incomplètes
                    if (not current_label or not next_label or 
                        (current_label in ["oui", "yes", "non", "no"] and not next_label) or
                        (not current_label and next_label in ["oui", "yes", "non", "no"])):
                        
                        # Analyser la position relative pour déterminer Oui/Non
                        x_current = current.get("bbox", [0, 0, 0, 0])[0]
                        x_next = next_box.get("bbox", [0, 0, 0, 0])[0]
                        
                        # En français, typiquement Oui est à gauche de Non
                        if x_current < x_next:
                            current["label"] = "Oui"
                            current["value"] = "Oui"
                            next_box["label"] = "Non"
                            next_box["value"] = "Non"
                        else:
                            current["label"] = "Non"
                            current["value"] = "Non"
                            next_box["label"] = "Oui"
                            next_box["value"] = "Oui"
                        
                        # Marquer comme faisant partie d'une paire
                        pair_id = f"pair_{current.get('page')}_{y_current}"
                        current["pair_id"] = pair_id
                        next_box["pair_id"] = pair_id
                        
                        # Ajouter les deux et avancer
                        result.append(current)
                        result.append(next_box)
                        i += 2
                        continue
                
            # Si pas de paire, ajouter normalement
            result.append(current)
            i += 1
        
        return result


    def _filter_redundant_checkboxes(self, checkboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filtre les cases redondantes et inutiles, comme des fausses détections.
        
        Args:
            checkboxes: Liste des cases à cocher
            
        Returns:
            Liste des cases à cocher filtrée
        """
        if not checkboxes:
            return []
        
        # 1. Filtrer les cases avec des étiquettes invalides
        filtered = []
        
        for checkbox in checkboxes:
            # Nettoyer et valider l'étiquette
            original_label = checkbox.get("label", "")
            cleaned_label = self._clean_and_validate_label(original_label)
            
            # Mettre à jour l'étiquette
            if cleaned_label != original_label:
                checkbox["label"] = cleaned_label
                if cleaned_label in ["Oui", "Non"]:
                    checkbox["value"] = cleaned_label
            
            # Filtrer les étiquettes problématiques
            if not cleaned_label and "pair_id" not in checkbox:
                # Garder uniquement les cases cochées sans étiquette
                if checkbox.get("checked", False) and checkbox.get("confidence", 0) > 0.75:
                    filtered.append(checkbox)
            else:
                filtered.append(checkbox)
        
        # 2. Éliminer les doublons spatiaux
        result = []
        used_positions = set()
        
        for checkbox in filtered:
            page = checkbox.get("page", 0)
            bbox = checkbox.get("bbox", [0, 0, 0, 0])
            
            # Calculer une position unique (arrondie pour tolérance)
            pos_key = f"{page}_{int(bbox[0]/5)}_{int(bbox[1]/5)}"
            
            if pos_key not in used_positions:
                used_positions.add(pos_key)
                result.append(checkbox)
        
        return result