# core/document_processing/invoice_processor.py
import re
import pandas as pd
import numpy as np
from datetime import datetime
import dateutil.parser
import json
import os
import io
import tempfile
import fitz  # PyMuPDF
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio


logger = logging.getLogger("invoice_processor")

class InvoiceProcessor:
    """
    Processeur spécialisé pour améliorer l'extraction des factures.
    Post-traite les tableaux extraits pour mieux structurer les données.
    """
    
    def __init__(self):
        """Initialise le processeur de factures."""
        # Patterns réguliers pour l'identification des champs
        self.patterns = {
            'invoice_number': r'(?i)facture\s*:?\s*([A-Z0-9\/-]+)',
            'date': r'(?i)date\s*:?\s*(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4})',
            'total': r'(?i)total.*?(\d+[,\.\s]\d+)',
            'vat_number': r'(?i)TVA\s*n[o°]?\.*\s*:?\s*([A-Z0-9]+)',
            'email': r'(?i)e-?mail\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        }
    
    def process(self, extracted_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Traite les tableaux extraits pour en faire une structure de facture.
        
        Args:
            extracted_tables: Liste des tableaux extraits
            
        Returns:
            Structure de facture enrichie
        """
        if not extracted_tables:
            return {"error": "Aucun tableau trouvé"}
        
        # Structure de résultat
        result = {
            "invoice_info": {},
            "customer_info": {},
            "line_items": [],
            "totals": {},
            "metadata": {
                "processor_version": "1.0",
                "processing_date": datetime.utcnow().isoformat(),
                "confidence": 0.0
            }
        }
        
        try:
            # Traitement du premier tableau (souvent le principal dans une facture)
            first_table = extracted_tables[0]
            raw_data = first_table.get("data", [])
            
            # Conversion du format de données si nécessaire
            data = []
            if isinstance(raw_data, pd.DataFrame):
                # Convertir DataFrame en liste de dictionnaires
                data = raw_data.to_dict(orient='records')
            elif isinstance(raw_data, str):
                # Tenter de parser JSON ou CSV
                try:
                    import json
                    data = json.loads(raw_data)
                    if not isinstance(data, list):
                        data = [{"content": raw_data}]
                except json.JSONDecodeError:
                    # Si ce n'est pas du JSON, essayer CSV
                    try:
                        import csv
                        import io
                        reader = csv.DictReader(io.StringIO(raw_data))
                        data = list(reader)
                    except:
                        # Fallback si aucun format n'est reconnu
                        data = [{"content": raw_data}]
            elif isinstance(raw_data, list):
                # Vérifier que les éléments sont des dictionnaires
                data = []
                for item in raw_data:
                    if isinstance(item, dict):
                        data.append(item)
                    else:
                        data.append({"value": str(item)})
            else:
                # Fallback pour tout autre type
                data = [{"content": str(raw_data)}]
            
            # Extraction des informations de base
            self._extract_basic_info(data, result)
            
            # Extraction des lignes de facture
            self._extract_line_items(data, result)
            
            # Extraction des totaux
            self._extract_totals(data, result)
            
            # Nettoyage et formatage des valeurs
            self._cleanup_values(result)
            
            # Calculs de validation (vérifier que le total correspond à la somme des lignes, etc.)
            confidence = self._validate_and_calculate_confidence(result)
            result["metadata"]["confidence"] = confidence
            
            # Identification du type de facture
            result["metadata"]["document_subtype"] = self._determine_document_subtype(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la facture: {e}")
            return {
                "error": f"Erreur de traitement: {str(e)}",
                "invoice_info": result.get("invoice_info", {}),
                "metadata": {
                    "processor_version": "1.0",
                    "processing_date": datetime.utcnow().isoformat(),
                    "confidence": 0.0,
                    "error": True
                }
            }
        
    def process_form_data(self, extracted_tables: List[Dict[str, Any]], checkbox_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Traite les données de formulaire pour créer une structure plus claire et exploitable.
        
        Args:
            extracted_tables: Liste des tableaux extraits
            checkbox_data: Données des cases à cocher extraites (optionnel)
            
        Returns:
            Structure de données de formulaire enrichie
        """
        # Structure de résultat
        result = {
            "type_document": "formulaire",
            "metadata": {
                "reference": None,
                "date": None,
                "titre": None,
                "processor_version": "1.0",
                "processing_date": datetime.utcnow().isoformat()
            },
            "sections": {},
            "form_fields": {}
        }
        
        try:
            # Extraire les métadonnées de base
            all_text = self._get_combined_text(extracted_tables)
            
            # Recherche de métadonnées communes dans les formulaires
            patterns = {
                'reference': r'(?i)(?:Réf|N°)[\.:\s]*([A-Z0-9\/-]+)',
                'date': r'(?i)(?:Le|Date|Edité le)[\s:]*(\d{1,2}[\s/.-]+\w+[\s/.-]+\d{4}|\d{1,2}[\s/.-]\d{1,2}[\s/.-]\d{2,4})',
                'client': r'(?i)(?:Client|Adress)[\.:\s]*([^\n.]+)',
                'titre': r'(?i)(?:Fiche|Formulaire|Affaire)[\.:\s]*([^\n.]+)',
            }
            
            # Extraction des métadonnées avec les patterns spécifiques
            for field, pattern in patterns.items():
                match = re.search(pattern, all_text)
                if match:
                    value = match.group(1).strip()
                    result["metadata"][field] = value
            
            # Traitement des tableaux principaux
            for i, table in enumerate(extracted_tables):
                table_data = table.get("data", None)
                
                # Ignorer les tableaux vides
                if not table_data or not isinstance(table_data, pd.DataFrame) or table_data.empty:
                    continue
                
                # Identifier les sections dans le tableau
                section_name = f"Section_{i+1}"
                
                # Tentative de détection du nom de section
                if "section" in table:
                    section_name = table["section"]
                else:
                    # Essayer de détecter le nom de section dans le tableau
                    for col_name in table_data.columns:
                        if "section" in str(col_name).lower() or "zone" in str(col_name).lower():
                            section_values = table_data[col_name].unique()
                            if len(section_values) > 0 and isinstance(section_values[0], str):
                                section_name = section_values[0]
                                break
                
                # Convertir le DataFrame en dictionnaire pour avoir une structure JSON exploitable
                fields = {}
                
                # Parcourir les lignes et rechercher des champs spécifiques
                for _, row in table_data.iterrows():
                    # Chercher des paires label/valeur
                    for i, col in enumerate(table_data.columns):
                        cell_value = row[col]
                        if pd.isna(cell_value) or cell_value == "":
                            continue
                        
                        cell_str = str(cell_value).strip()
                        
                        # Si c'est un label potentiel (se termine par :)
                        if cell_str.endswith(':'):
                            label = cell_str.rstrip(':')
                            
                            # Chercher la valeur dans les colonnes suivantes
                            for next_col in table_data.columns[i+1:]:
                                next_value = row[next_col]
                                if not pd.isna(next_value) and next_value != "":
                                    fields[label] = str(next_value).strip()
                                    break
                        # Sinon, si c'est une colonne clé/valeur typique de formulaire
                        elif i < len(table_data.columns) - 1:
                            next_col = table_data.columns[i+1]
                            next_value = row[next_col]
                            
                            if not pd.isna(next_value) and next_value != "":
                                fields[cell_str] = str(next_value).strip()
                                
                # Ajouter à la section appropriée
                if section_name not in result["sections"]:
                    result["sections"][section_name] = {}
                    
                result["sections"][section_name].update(fields)
                
                # Ajouter également au dictionnaire plat pour accès facile
                result["form_fields"].update(fields)
            
            # Intégrer les données de cases à cocher si disponibles
            if checkbox_data:
                checkbox_values = {}
                
                # Format par section
                if "sections" in checkbox_data:
                    for section, checkboxes in checkbox_data["sections"].items():
                        if section not in result["sections"]:
                            result["sections"][section] = {}
                        
                        for checkbox in checkboxes:
                            label = checkbox.get("label", "")
                            value = checkbox.get("value", "")
                            
                            if label:
                                result["sections"][section][label] = value
                                checkbox_values[label] = value
                
                # Format plat (liste complète)
                if "checkboxes" in checkbox_data:
                    for checkbox in checkbox_data["checkboxes"]:
                        label = checkbox.get("label", "")
                        value = checkbox.get("value", "")
                        
                        if label:
                            checkbox_values[label] = value
                
                # Format simplifié déjà prêt
                if "form_values" in checkbox_data:
                    checkbox_values.update(checkbox_data["form_values"])
                
                # Ajouter au dictionnaire plat
                result["form_fields"].update(checkbox_values)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données de formulaire: {e}")
            result["error"] = str(e)
            return result
    
    def _extract_basic_info(self, data: List[Dict[str, Any]], result: Dict[str, Any]):
        """Extrait les informations de base de la facture."""
        # Fusion de toutes les valeurs textuelles pour chercher les patterns
        all_text = " ".join([
            " ".join(str(value) for value in row.values() if value) 
            for row in data
        ])
        
        # Extraction avec les patterns réguliers
        for field, pattern in self.patterns.items():
            match = re.search(pattern, all_text)
            if match:
                value = match.group(1).strip()
                if field == 'invoice_number':
                    result["invoice_info"]["number"] = value
                elif field == 'date':
                    try:
                        # Tentative de parsing et formatage de la date
                        parsed_date = dateutil.parser.parse(value, dayfirst=True)
                        result["invoice_info"]["date"] = parsed_date.strftime("%Y-%m-%d")
                    except:
                        result["invoice_info"]["date"] = value
                elif field == 'vat_number':
                    result["invoice_info"]["vat_number"] = value
                elif field == 'email':
                    result["customer_info"]["email"] = value
        
        # Recherche du vendeur et client dans les premières lignes
        vendor_detected = False
        address_lines = []
        
        for i, row in enumerate(data[:6]):  # Les premières lignes contiennent souvent ces infos
            # Parcourir chaque colonne
            for col_name, value in row.items():
                if not value or not isinstance(value, str):
                    continue
                    
                value = value.strip()
                
                # Détection du nom de l'entreprise (Volotea, Air France, etc.)
                if not vendor_detected and re.search(r'(?i)(VOLOTEA|AIR\s+FRANCE|SNCF|ENGIE|EDF)', value):
                    vendor_name = re.search(r'(?i)(VOLOTEA|AIR\s+FRANCE|SNCF|ENGIE|EDF)', value).group(1).upper()
                    result["invoice_info"]["vendor"] = vendor_name
                    vendor_detected = True
                
                # Détection d'adresse
                if any(x in value.lower() for x in ["aéroport", "rue", "avenue", "boulevard", "cedex"]):
                    address_lines.append(value)
                
                # Détection de TVA intracommunautaire
                vat_match = re.search(r'(?i)TVA\s*(?:intra)?.*:\s*([A-Z0-9]+)', value)
                if vat_match:
                    result["invoice_info"]["vat_number"] = vat_match.group(1)
                
                # Détection de SIRET ou SIREN
                siret_match = re.search(r'(?i)SIRET\s*:?\s*(\d[\s\d]{13,})', value)
                if siret_match:
                    result["invoice_info"]["siret"] = siret_match.group(1).replace(" ", "")
                
                # Détection d'e-mail
                email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', value)
                if email_match:
                    result["customer_info"]["email"] = email_match.group(0)
        
        # Ajouter l'adresse si trouvée
        if address_lines:
            result["invoice_info"]["address"] = address_lines
    
    def _extract_line_items(self, data: List[Dict[str, Any]], result: Dict[str, Any]):
        """Extrait les lignes de facture."""
        line_items = []
        in_items_section = False
        
        for i, row in enumerate(data):
            # Reconstruire le texte complet de la ligne en joignant toutes les colonnes
            line_text = " ".join(str(value) for value in row.values() if value)
            
            # Détection des lignes de détail de vol
            if re.search(r'\d{2}/\d{2}/\d{4}', line_text) and ("-" in line_text or any(city in line_text for city in ["Montpellier", "Nantes", "Paris", "Lyon", "Bordeaux"])):
                in_items_section = True
                
                # Extraction des parties
                date_match = re.search(r'(\d{2}/\d{2}/\d{4})', line_text)
                date = date_match.group(1) if date_match else ""
                
                # Extraction du trajet
                route_match = re.search(r'([A-Za-zÀ-ÖØ-öø-ÿ]+\s*-\s*[A-Za-zÀ-ÖØ-öø-ÿ]+)', line_text)
                route = route_match.group(1) if route_match else ""
                
                # Extraction du passager
                passenger_match = re.search(r'([A-Za-zÀ-ÖØ-öø-ÿ]+\s+[A-Za-zÀ-ÖØ-öø-ÿ]+)(?=\s+\d)', line_text)
                passenger = passenger_match.group(1) if passenger_match else ""
                
                # Extraction du montant (en recherchant des chiffres séparés par des espaces à la fin)
                amount_match = re.search(r'(\d+\s+\d+(?:\s+\d+)?)$', line_text)
                amount = amount_match.group(1).replace(" ", ".") if amount_match else ""
                
                line_items.append({
                    "date": date,
                    "route": route,
                    "passenger": passenger,
                    "amount": self._clean_numeric(amount),
                    "type": "flight"
                })
            
            # Détection des réductions
            elif ("Réductions" in line_text or "réduction" in line_text.lower()) and in_items_section:
                amount_match = re.search(r'(-\d+[\s,\.]\d+)', line_text)
                amount = amount_match.group(1).replace(" ", ".") if amount_match else ""
                
                line_items.append({
                    "type": "discount",
                    "description": "Réduction",
                    "amount": self._clean_numeric(amount)
                })
                
            # Détection d'autres types de services (assurance, bagages, etc.)
            elif any(service in line_text.lower() for service in ["assurance", "bagages", "service", "supplément"]) and in_items_section:
                # Extraction du type de service
                service_type = "other"
                for s_type in ["assurance", "bagages", "supplément"]:
                    if s_type in line_text.lower():
                        service_type = s_type
                        break
                
                # Extraction du montant
                amount_match = re.search(r'(\d+[\s,\.]\d+)(?:\s+€)?$', line_text)
                amount = amount_match.group(1).replace(" ", ".") if amount_match else ""
                
                line_items.append({
                    "type": service_type,
                    "description": line_text.strip(),
                    "amount": self._clean_numeric(amount),
                })
        
        result["line_items"] = line_items
    
    def _extract_totals(self, data: List[Dict[str, Any]], result: Dict[str, Any]):
        """Extrait les totaux de la facture."""
        totals = {}
        
        for row in data:
            # Reconstruire le texte complet de la ligne
            line_text = " ".join(str(value) for value in row.values() if value)
            
            # Subtotal
            if "Subtotal" in line_text or "sous-total" in line_text.lower():
                amount_match = re.search(r'(\d+[\s,\.]\d+)', line_text)
                if amount_match:
                    totals["subtotal"] = self._clean_numeric(amount_match.group(1))
            
            # Base d'imposition
            elif "Base d'imposition" in line_text:
                if "assujettie" in line_text:
                    amount_match = re.search(r'(\d+[\s,\.]\d+)', line_text)
                    if amount_match:
                        totals["taxable_base"] = self._clean_numeric(amount_match.group(1))
                elif "non assujettie" in line_text:
                    amount_match = re.search(r'(\d+[\s,\.]\d+)', line_text)
                    if amount_match:
                        totals["non_taxable_base"] = self._clean_numeric(amount_match.group(1))
            
            # TVA
            elif "TVA" in line_text:
                rate_match = re.search(r'TVA\s*\((\d+)\s*%\)', line_text)
                amount_match = re.search(r'(\d+[\s,\.]\d+)', line_text)
                
                if amount_match:
                    amount = self._clean_numeric(amount_match.group(1))
                    
                    if rate_match:
                        rate = rate_match.group(1)
                        if "vat" not in totals:
                            totals["vat"] = []
                        
                        totals["vat"].append({
                            "rate": f"{rate}%",
                            "amount": amount
                        })
                    else:
                        # Si pas de taux spécifié, on met dans le total TVA
                        totals["vat_total"] = amount
            
            # Total
            elif "TOTAL" in line_text.upper():
                currency_match = re.search(r'\((.*?)\)', line_text)
                amount_match = re.search(r'(?:TOTAL(?:\s+\(.*?\))?\s*|^)(\d+[\s,\.]\d+)(?:\s*€)?$', line_text)
                
                if amount_match:
                    totals["total"] = self._clean_numeric(amount_match.group(1))
                    if currency_match:
                        totals["currency"] = currency_match.group(1)
                    else:
                        # Recherche de symbole de devise
                        currency_symbol = re.search(r'(\$|€|£|USD|EUR|GBP)', line_text)
                        totals["currency"] = currency_symbol.group(1) if currency_symbol else "EUR"
        
        result["totals"] = totals
    
    def _clean_numeric(self, value: str) -> float:
        """Nettoie et convertit une valeur numérique."""
        if not value:
            return 0.0
            
        try:
            # Supprimer les caractères non numériques sauf . et ,
            value = re.sub(r'[^\d,\.-]', '', value)
            
            # Remplacer la virgule par un point pour le parsing
            value = value.replace(',', '.')
            
            return float(value)
        except Exception as e:
            logger.debug(f"Erreur conversion numérique '{value}': {e}")
            return 0.0
    
    def _cleanup_values(self, result: Dict[str, Any]):
        """Nettoie et formate toutes les valeurs de la structure."""
        # Formatage des montants avec 2 décimales
        if "totals" in result:
            for key, value in result["totals"].items():
                if isinstance(value, (int, float)):
                    result["totals"][key] = round(value, 2)
                elif isinstance(value, list) and key == "vat":
                    for vat_item in value:
                        if "amount" in vat_item:
                            vat_item["amount"] = round(vat_item["amount"], 2)
        
        # Formatage des montants des lignes
        if "line_items" in result:
            for item in result["line_items"]:
                if "amount" in item and isinstance(item["amount"], (int, float)):
                    item["amount"] = round(item["amount"], 2)
        
        # Formatage des dates
        if "invoice_info" in result and "date" in result["invoice_info"]:
            date_str = result["invoice_info"]["date"]
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                try:
                    parsed_date = dateutil.parser.parse(date_str, dayfirst=True)
                    result["invoice_info"]["date"] = parsed_date.strftime("%Y-%m-%d")
                except Exception as e:
                    logger.debug(f"Erreur parsing date '{date_str}': {e}")
    
    def _validate_and_calculate_confidence(self, result: Dict[str, Any]) -> float:
        """
        Valide les données et calcule un score de confiance.
        
        Returns:
            Score de confiance (0-1)
        """
        score = 0.0
        checks = 0
        
        # Vérification de la présence des informations essentielles
        if result.get("invoice_info", {}).get("number"):
            score += 1
            checks += 1
        
        if result.get("invoice_info", {}).get("date"):
            score += 1
            checks += 1
        
        if result.get("invoice_info", {}).get("vendor"):
            score += 1
            checks += 1
        
        # Vérification du nombre de lignes
        line_items = result.get("line_items", [])
        if line_items:
            if len(line_items) >= 1:
                score += 1
            checks += 1
        
        # Vérification des totaux
        totals = result.get("totals", {})
        if "total" in totals:
            score += 1
            checks += 1
            
            # Vérifier la cohérence entre les lignes et le total
            if line_items:
                try:
                    items_total = sum(item.get("amount", 0) for item in line_items)
                    total_diff = abs(items_total - totals["total"])
                    
                    # Si la différence est inférieure à 1€ ou 5%, c'est plutôt bon
                    if total_diff < 1 or (totals["total"] > 0 and total_diff / totals["total"] < 0.05):
                        score += 2
                    checks += 2
                except Exception:
                    pass
        
        # Vérification de la TVA
        if "vat" in totals or "vat_total" in totals:
            score += 1
            checks += 1
        
        # Calcul du score final
        return round(score / max(1, checks), 2) if checks > 0 else 0.0
    
    def _determine_document_subtype(self, result: Dict[str, Any]) -> str:
        """
        Identifie le sous-type de document basé sur son contenu.
        
        Returns:
            Sous-type de document (airline_ticket, hotel_invoice, etc.)
        """
        # Vérifier s'il s'agit d'un billet d'avion
        if result.get("invoice_info", {}).get("vendor") in ["VOLOTEA", "AIR FRANCE", "LUFTHANSA", "EASYJET"]:
            for item in result.get("line_items", []):
                if item.get("type") == "flight" or item.get("route"):
                    return "airline_ticket"
        
        # Vérifier s'il s'agit d'un billet de train
        if result.get("invoice_info", {}).get("vendor") in ["SNCF", "EUROSTAR", "THALYS"]:
            return "train_ticket"
        
        # Par défaut, c'est une facture générique
        return "generic_invoice"
    
    def process_technical_invoice(self, extracted_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Traite les tableaux extraits pour les devis techniques (construction, métallurgie, etc.)
        
        Args:
            extracted_tables: Liste des tableaux extraits
            
        Returns:
            Structure de devis technique enrichie
        """
        if not extracted_tables:
            return {"error": "Aucun tableau trouvé"}
        
        # Structure de résultat spécifique aux devis techniques
        result = {
            "type_document": "devis_technique",
            "metadata": {
                "reference": None,
                "date": None,
                "client": None,
                "contact": None,
                "processor_version": "1.0",
                "processing_date": datetime.utcnow().isoformat(),
                "confidence": 0.0
            },
            "sections": [],
            "totals": {
                "montant_ht": None,
                "tva": None,
                "montant_ttc": None,
                "currency": "EUR"
            },
            "conditions": {
                "delai": None,
                "paiement": None,
                "validite": None
            }
        }
        
        try:
            # Extraction des métadonnées
            self._extract_technical_metadata(extracted_tables, result)
            
            # Détection et extraction des sections techniques
            self._extract_technical_sections(extracted_tables, result)
            
            # Extraction des totaux
            self._extract_technical_totals(extracted_tables, result)
            
            # Extraction des conditions
            self._extract_technical_conditions(extracted_tables, result)
            
            # Nettoyage et formatage final
            self._cleanup_technical_values(result)
            
            # Calcul du score de confiance
            confidence = self._calculate_technical_confidence(result)
            result["metadata"]["confidence"] = confidence
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du devis technique: {e}")
            result["error"] = str(e)
            return result

    def _extract_technical_metadata(self, extracted_tables: List[Dict[str, Any]], result: Dict[str, Any]):
        """
        Extrait les métadonnées du devis technique.
        
        Args:
            extracted_tables: Tableaux extraits
            result: Structure de résultat à compléter
        """
        # Patterns réguliers spécifiques aux devis techniques
        devis_patterns = {
            'reference': r'(?i)(?:Devis|Réf)[\.:\s]*([A-Z0-9\/-]+)',
            'date': r'(?i)(?:Le|Date)[\s:]*(\d{1,2}[\s/.-]+\w+[\s/.-]+\d{4}|\d{1,2}[\s/.-]\d{1,2}[\s/.-]\d{2,4})',
            'client': r'(?i)(?:A l\'attention de|Client)[\.:\s]*([^\n.]+)',
            'delai': r'(?i)Délai[\s:]*([^\n.]+)',
            'payment': r'(?i)(?:paiement|règlement)[\s:]*([^\n.]+)',
        }
        
        # Fusion de tous les textes pour la recherche
        all_text = self._get_combined_text(extracted_tables)
        
        # Extraction des métadonnées avec les patterns spécifiques
        for field, pattern in devis_patterns.items():
            match = re.search(pattern, all_text)
            if match:
                value = match.group(1).strip()
                
                if field == 'reference':
                    result["metadata"]["reference"] = value
                elif field == 'date':
                    try:
                        # Parsing de la date avec gestion des formats français
                        value = value.replace("janvier", "January").replace("février", "February") \
                                    .replace("mars", "March").replace("avril", "April") \
                                    .replace("mai", "May").replace("juin", "June") \
                                    .replace("juillet", "July").replace("août", "August") \
                                    .replace("septembre", "September").replace("octobre", "October") \
                                    .replace("novembre", "November").replace("décembre", "December")
                        
                        parsed_date = dateutil.parser.parse(value, dayfirst=True)
                        result["metadata"]["date"] = parsed_date.strftime("%Y-%m-%d")
                    except Exception as e:
                        logger.warning(f"Erreur parsing date '{value}': {e}")
                        result["metadata"]["date"] = value
                elif field == 'client':
                    result["metadata"]["client"] = value
                elif field == 'delai':
                    result["conditions"]["delai"] = value
                elif field == 'payment':
                    result["conditions"]["paiement"] = value
        
        # Recherche d'informations de contact
        contact_match = re.search(r'(?i)(?:tél|téléphone)[\.:\s]*([+\d\s.()-]{8,})', all_text)
        if contact_match:
            result["metadata"]["contact"] = contact_match.group(1).strip()

    def _extract_technical_sections(self, extracted_tables: List[Dict[str, Any]], result: Dict[str, Any]):
        """
        Extrait les sections techniques du devis.
        
        Args:
            extracted_tables: Tableaux extraits
            result: Structure de résultat à compléter
        """
        # Keywords pour détecter les sections techniques
        section_keywords = {
            "empannage": ["empannage", "empan", "entrait", "poutre"],
            "lissage": ["lissage", "lisse", "bardage", "pannes"],
            "pignon": ["pignon", "faîtage", "faitage", "poteau"],
            "couverture": ["couverture", "toiture", "bac", "acier"],
            "fondation": ["fondation", "semelle", "ancrage", "béton", "beton"]
        }
        
        # Parcourir les tableaux
        for table_idx, table in enumerate(extracted_tables):
            table_data = table.get("data", None)
            if table_data is None or not isinstance(table_data, pd.DataFrame) or table_data.empty:
                continue
            
            # Détecter la section en fonction des mots-clés
            section_type = None
            section_confidence = 0.0
            
            # Convertir le DataFrame en texte pour la recherche
            table_text = " ".join(table_data.astype(str).values.flatten()).lower()
            
            for section_name, keywords in section_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in table_text:
                        section_type = section_name
                        section_confidence = 0.8
                        break
                if section_type:
                    break
            
            # Si aucune section détectée, utiliser un nom générique
            if not section_type:
                section_type = f"section_{table_idx+1}"
                section_confidence = 0.4
            
            # Extraction des éléments de la section
            elements = self._extract_technical_elements(table_data, section_type)
            
            # Extraction du prix de la section
            section_price = self._extract_section_price(table_data)
            
            # Création de la section
            section = {
                "type": section_type,
                "elements": elements,
                "prix_ht": section_price,
                "confidence": section_confidence
            }
            
            result["sections"].append(section)

    def _extract_technical_elements(self, df: pd.DataFrame, section_type: str) -> List[Dict[str, Any]]:
        """
        Extrait les éléments d'une section technique.
        
        Args:
            df: DataFrame du tableau
            section_type: Type de section
            
        Returns:
            Liste des éléments
        """
        elements = []
        
        # Recherche de mots-clés spécifiques aux éléments techniques
        element_keywords = ["multibeam", "cours", "profile", "ipn", "ipe", "hea", "heb", "poutre"]
        
        # Parcourir les lignes
        for idx, row in df.iterrows():
            # Chercher des mots-clés dans chaque cellule
            for col in df.columns:
                cell_value = str(row[col]).lower()
                
                # Vérifier si c'est une ligne d'élément
                if any(keyword in cell_value for keyword in element_keywords):
                    # Extraire les informations de l'élément
                    element = self._extract_element_info(row, df.columns)
                    
                    if element and element.get("description"):
                        elements.append(element)
                    break
        
        # Si aucun élément trouvé avec les mots-clés, essayer une approche plus générique
        if not elements:
            # Parcourir les lignes pour trouver des motifs numériques typiques des devis
            for idx, row in df.iterrows():
                # Recherche de quantités (ml, kg, m², pièces)
                quantity_pattern = re.compile(r'(\d+(?:[,.]\d+)?)\s*(?:ml|kg|m²|m2|pièces|pieces|u)')
                found_quantity = False
                
                for col in df.columns:
                    cell_value = str(row[col])
                    if quantity_pattern.search(cell_value):
                        found_quantity = True
                        break
                
                if found_quantity:
                    # Extraire les informations de l'élément
                    element = self._extract_element_info(row, df.columns)
                    
                    if element and element.get("description"):
                        elements.append(element)
                        
        return elements

    def _extract_element_info(self, row: pd.Series, columns: pd.Index) -> Dict[str, Any]:
        """
        Extrait les informations d'un élément technique à partir d'une ligne de tableau.
        
        Args:
            row: Ligne du DataFrame
            columns: Colonnes du DataFrame
            
        Returns:
            Dictionnaire des informations de l'élément
        """
        element = {
            "description": None,
            "quantite": None,
            "dimensions": None,
            "unite": None,
            "specifications": None,
            "prix_unitaire": None
        }
        
        # Parcourir les colonnes pour identifier les informations
        for col in columns:
            cell_value = str(row[col]).strip()
            if not cell_value or cell_value.lower() in ["nan", "none", "-"]:
                continue
                
            # Profil technique (Multibeam, IPN, etc.)
            profile_match = re.search(r'(?i)(multibeam|profile|ipn|ipe|hea|heb|poutre|plaque)[\s\w]+([a-z]\d{2,3})', cell_value)
            if profile_match:
                element["description"] = cell_value
                continue
                
            # Quantité (avec unité)
            qty_match = re.search(r'(\d+(?:[,.]\d+)?)\s*(ml|kg|m²|m2|pièces|pieces|u)', cell_value)
            if qty_match:
                element["quantite"] = self._clean_numeric(qty_match.group(1))
                element["unite"] = qty_match.group(2)
                continue
                
            # Dimensions
            dim_match = re.search(r'(?i)(?:écartement|ecartement|espacement|dimensions?)\s+(?:maxi|max|de)?\s*(?:\w+\s+)?(\d+(?:[,.]\d+)?)\s*(?:x\s*(\d+(?:[,.]\d+)?))?\s*(?:mm|cm|m)', cell_value)
            if dim_match:
                element["dimensions"] = cell_value
                continue
                
            # Spécifications (galvanisation, etc.)
            if any(spec in cell_value.lower() for spec in ["galvanis", "standard", "qualité", "qualite", "acier", "face", "gr/m²"]):
                element["specifications"] = cell_value
                continue
                
            # Prix unitaire
            price_match = re.search(r'(\d+(?:[,.]\d+)?)\s*(?:€|EUR|HT)?(?:/(?:ml|kg|m²|m2|pièce|u))?', cell_value)
            if price_match and not element["prix_unitaire"]:
                element["prix_unitaire"] = self._clean_numeric(price_match.group(1))
                continue
        
        # Si aucune description trouvée mais d'autres champs sont remplis,
        # construire une description générique
        if not element["description"] and (element["quantite"] or element["dimensions"]):
            desc_parts = []
            if element["quantite"] and element["unite"]:
                desc_parts.append(f"{element['quantite']} {element['unite']}")
            if element["dimensions"]:
                desc_parts.append(element["dimensions"])
                
            if desc_parts:
                element["description"] = "Élément: " + " - ".join(desc_parts)
        
        return element

    def _extract_section_price(self, df: pd.DataFrame) -> Optional[float]:
        """
        Extrait le prix total d'une section.
        
        Args:
            df: DataFrame du tableau
            
        Returns:
            Prix total ou None
        """
        # Recherche de prix total
        price_value = None
        
        # Convertir toutes les données en chaînes
        df_str = df.astype(str)
        
        # Recherche par mots-clés
        price_keywords = ["prix total", "total", "ht", "franco"]
        
        for idx, row in df_str.iterrows():
            # Vérifier chaque cellule pour les mots-clés de prix
            for col in df.columns:
                cell_value = row[col].lower()
                
                if any(keyword in cell_value for keyword in price_keywords):
                    # Rechercher un prix dans cette ligne
                    price_match = None
                    
                    # D'abord chercher dans la même cellule
                    price_match = re.search(r'(\d+(?:[\s,.]\d+)?)\s*(?:€|EUR|HT)?', cell_value)
                    
                    # Si pas trouvé, chercher dans les autres cellules de la ligne
                    if not price_match:
                        for other_col in df.columns:
                            if other_col != col:
                                other_value = str(row[other_col])
                                price_match = re.search(r'(\d+(?:[\s,.]\d+)?)\s*(?:€|EUR|HT)?', other_value)
                                if price_match:
                                    price_value = self._clean_numeric(price_match.group(1))
                                    break
                    else:
                        price_value = self._clean_numeric(price_match.group(1))
                        
                    if price_value:
                        return price_value
        
        # Si aucun prix trouvé, recherche plus générique
        for idx, row in df_str.iterrows():
            for col in df.columns:
                cell_value = str(row[col])
                
                # Rechercher un format typique de prix (nombre significatif)
                price_match = re.search(r'(\d{3,}(?:[\s,.]\d{2}))\s*(?:€|EUR|HT)?$', cell_value)
                if price_match:
                    return self._clean_numeric(price_match.group(1))
        
        return None

    def _extract_technical_totals(self, extracted_tables: List[Dict[str, Any]], result: Dict[str, Any]):
        """
        Extrait les totaux du devis.
        
        Args:
            extracted_tables: Tableaux extraits
            result: Structure de résultat à compléter
        """
        # Recherche de tableaux contenant les totaux (généralement à la fin)
        total_ht = None
        total_tva = None
        total_ttc = None
        currency = "EUR"
        
        # Parcourir tous les tableaux
        for table in extracted_tables:
            table_data = table.get("data", None)
            if table_data is None or not isinstance(table_data, pd.DataFrame) or table_data.empty:
                continue
            
            # Convertir en chaînes pour la recherche
            df_str = table_data.astype(str)
            
            # Recherche de mots-clés liés aux totaux
            for idx, row in df_str.iterrows():
                row_text = " ".join(row.values).lower()
                
                # Total HT
                if "total" in row_text and ("ht" in row_text or "h.t" in row_text):
                    price_match = re.search(r'(\d+(?:[\s,.]\d+)?)', row_text)
                    if price_match:
                        total_ht = self._clean_numeric(price_match.group(1))
                
                # TVA
                elif "tva" in row_text or "t.v.a" in row_text:
                    price_match = re.search(r'(\d+(?:[\s,.]\d+)?)', row_text)
                    if price_match:
                        total_tva = self._clean_numeric(price_match.group(1))
                        
                        # Recherche du taux de TVA
                        tva_rate_match = re.search(r'(\d+(?:[\s,.]\d+)?)[\s%]', row_text)
                        if tva_rate_match:
                            result["totals"]["tva_rate"] = self._clean_numeric(tva_rate_match.group(1))
                
                # Total TTC
                elif "ttc" in row_text or "t.t.c" in row_text or "toutes taxes" in row_text:
                    price_match = re.search(r'(\d+(?:[\s,.]\d+)?)', row_text)
                    if price_match:
                        total_ttc = self._clean_numeric(price_match.group(1))
                        
                # Devise
                currency_match = re.search(r'(€|EUR|euros?|HT)', row_text)
                if currency_match:
                    currency_found = currency_match.group(1).upper()
                    if currency_found == "€":
                        currency = "EUR"
                    elif currency_found == "EUROS" or currency_found == "EURO":
                        currency = "EUR"
                    elif currency_found != "HT":
                        currency = currency_found
        
        # Si aucun total trouvé, calculer à partir des sections
        if total_ht is None:
            # Somme des prix des sections
            section_prices = [section.get("prix_ht", 0) for section in result["sections"] if section.get("prix_ht")]
            if section_prices:
                total_ht = sum(section_prices)
        
        # Si total TTC non trouvé mais HT et TVA disponibles, calculer
        if total_ht is not None and total_tva is not None and total_ttc is None:
            total_ttc = total_ht + total_tva
        
        # Mise à jour des résultats
        result["totals"]["montant_ht"] = total_ht
        result["totals"]["tva"] = total_tva
        result["totals"]["montant_ttc"] = total_ttc
        result["totals"]["currency"] = currency

    def _extract_technical_conditions(self, extracted_tables: List[Dict[str, Any]], result: Dict[str, Any]):
        """
        Extrait les conditions du devis.
        
        Args:
            extracted_tables: Tableaux extraits
            result: Structure de résultat à compléter
        """
        # Recherche de tableaux contenant les conditions
        for table in extracted_tables:
            table_data = table.get("data", None)
            if table_data is None or not isinstance(table_data, pd.DataFrame) or table_data.empty:
                continue
            
            # Convertir en chaînes pour la recherche
            df_str = table_data.astype(str)
            
            # Recherche de mots-clés liés aux conditions
            for idx, row in df_str.iterrows():
                row_text = " ".join(row.values).lower()
                
                # Délai
                if "délai" in row_text or "livraison" in row_text:
                    delai_match = re.search(r'(?:délai|livraison)[:\s]*(.*?)(?:\.|$)', row_text)
                    if delai_match:
                        result["conditions"]["delai"] = delai_match.group(1).strip()
                
                # Paiement
                elif "paiement" in row_text or "règlement" in row_text:
                    payment_match = re.search(r'(?:paiement|règlement)[:\s]*(.*?)(?:\.|$)', row_text)
                    if payment_match:
                        result["conditions"]["paiement"] = payment_match.group(1).strip()
                
                # Validité
                elif "valid" in row_text or "offre" in row_text:
                    valid_match = re.search(r'(?:validité|offre\s+valable)[:\s]*(.*?)(?:\.|$)', row_text)
                    if valid_match:
                        result["conditions"]["validite"] = valid_match.group(1).strip()
        
        # Si conditions non trouvées dans les tableaux, chercher dans le texte global
        if not any(result["conditions"].values()):
            all_text = self._get_combined_text(extracted_tables).lower()
            
            # Délai
            if not result["conditions"]["delai"]:
                delai_match = re.search(r'(?:délai|livraison)[:\s]*(.*?)(?:\.|$)', all_text)
                if delai_match:
                    result["conditions"]["delai"] = delai_match.group(1).strip()
            
            # Paiement
            if not result["conditions"]["paiement"]:
                payment_match = re.search(r'(?:paiement|règlement)[:\s]*(.*?)(?:\.|$)', all_text)
                if payment_match:
                    result["conditions"]["paiement"] = payment_match.group(1).strip()
            
            # Validité
            if not result["conditions"]["validite"]:
                valid_match = re.search(r'(?:validité|offre\s+valable)[:\s]*(.*?)(?:\.|$)', all_text)
                if valid_match:
                    result["conditions"]["validite"] = valid_match.group(1).strip()

    def _get_combined_text(self, extracted_tables: List[Dict[str, Any]]) -> str:
        """
        Combine tous les textes des tableaux pour la recherche.
        
        Args:
            extracted_tables: Tableaux extraits
            
        Returns:
            Texte combiné
        """
        all_text = []
        
        for table in extracted_tables:
            table_data = table.get("data", None)
            if table_data is None:
                continue
                
            if isinstance(table_data, pd.DataFrame):
                # Convertir le DataFrame en liste de chaînes
                for idx, row in table_data.iterrows():
                    row_text = " ".join(str(val) for val in row.values if not pd.isna(val))
                    all_text.append(row_text)
            elif isinstance(table_data, list):
                # Si le tableau est déjà une liste, la parcourir
                for row in table_data:
                    if isinstance(row, dict):
                        row_text = " ".join(str(val) for val in row.values() if val)
                        all_text.append(row_text)
                    elif isinstance(row, list):
                        row_text = " ".join(str(val) for val in row if val)
                        all_text.append(row_text)
                    else:
                        all_text.append(str(row))
            else:
                # Fallback pour tout autre type
                all_text.append(str(table_data))
        
        return " ".join(all_text)

    def _cleanup_technical_values(self, result: Dict[str, Any]):
        """
        Nettoie et formate les valeurs du résultat.
        
        Args:
            result: Structure de résultat à nettoyer
        """
        # Formatage des montants
        if result["totals"]["montant_ht"] is not None:
            result["totals"]["montant_ht"] = round(result["totals"]["montant_ht"], 2)
        
        if result["totals"]["tva"] is not None:
            result["totals"]["tva"] = round(result["totals"]["tva"], 2)
        
        if result["totals"]["montant_ttc"] is not None:
            result["totals"]["montant_ttc"] = round(result["totals"]["montant_ttc"], 2)
        
        # Formatage des prix dans les sections
        for section in result["sections"]:
            if section["prix_ht"] is not None:
                section["prix_ht"] = round(section["prix_ht"], 2)
            
            # Formatage des prix des éléments
            for element in section["elements"]:
                if element["prix_unitaire"] is not None:
                    element["prix_unitaire"] = round(element["prix_unitaire"], 2)

    def _calculate_technical_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calcule un score de confiance pour l'extraction.
        
        Args:
            result: Résultat de l'extraction
            
        Returns:
            Score de confiance (0-1)
        """
        # Points de score
        score = 0
        total_points = 0
        
        # Métadonnées de base
        for field in ["reference", "date", "client"]:
            total_points += 1
            if result["metadata"].get(field):
                score += 1
        
        # Sections
        if result["sections"]:
            total_points += 1
            score += 1
            
            # Éléments dans les sections
            section_with_elements = 0
            for section in result["sections"]:
                if section["elements"]:
                    section_with_elements += 1
            
            if section_with_elements > 0:
                total_points += 1
                score += min(1.0, section_with_elements / len(result["sections"]))
            
            # Prix des sections
            section_with_price = 0
            for section in result["sections"]:
                if section["prix_ht"] is not None:
                    section_with_price += 1
            
            if section_with_price > 0:
                total_points += 1
                score += min(1.0, section_with_price / len(result["sections"]))
        
        # Totaux
        total_points += 1
        if result["totals"].get("montant_ht") is not None:
            score += 0.5
        if result["totals"].get("montant_ttc") is not None:
            score += 0.5
        
        # Conditions
        conditions_found = sum(1 for val in result["conditions"].values() if val)
        if conditions_found > 0:
            total_points += 1
            score += min(1.0, conditions_found / len(result["conditions"]))
        
        # Calcul final
        return round(score / max(1, total_points), 2)

    def detect_document_type(self, file_content: Union[str, bytes, io.BytesIO]) -> Dict[str, float]:
        """
        Détecte le type de document à partir de son contenu (version synchrone).
        
        Args:
            file_content: Contenu du fichier (chemin, bytes ou BytesIO)
            
        Returns:
            Dictionnaire des types de documents avec scores de confiance
        """
        try:
            # Préparation du contenu
            if isinstance(file_content, io.BytesIO):
                file_content.seek(0)
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_file.write(file_content.read())
                    temp_path = temp_file.name
                try:
                    file_path = temp_path
                    is_temp = True
                except Exception as e:
                    logger.error(f"Erreur création fichier temporaire: {e}")
                    return {"generic": 1.0}
            elif isinstance(file_content, bytes):
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_file.write(file_content)
                    temp_path = temp_file.name
                file_path = temp_path
                is_temp = True
            else:
                file_path = file_content
                is_temp = False
            
            # Lecture du PDF
            with fitz.open(file_path) as doc:
                # Limiter à max 5 pages pour l'analyse
                max_pages = min(5, len(doc))
                
                # Extraction du texte pour analyse
                all_text = ""
                for page_idx in range(max_pages):
                    all_text += doc[page_idx].get_text().lower()
                
                # Détection des types de documents
                types = {
                    "devis_technique": 0.0,
                    "facture": 0.0,
                    "bon_livraison": 0.0,
                    "document_standard": 0.0
                }
                
                # Mots-clés pour devis techniques
                devis_keywords = [
                    "devis", "offre", "prix", "proposition", "empannage", "lissage", 
                    "pignon", "multibeam", "poutre", "bardage", "couverture", 
                    "acier", "charpente", "construction", "bâtiment", "ht", "franco"
                ]
                
                # Mots-clés pour factures
                facture_keywords = [
                    "facture", "avoir", "acquitté", "règlement", "tva", "remise", 
                    "client", "compte", "date d'échéance", "date de facturation"
                ]
                
                # Mots-clés pour bons de livraison
                livraison_keywords = [
                    "bon de livraison", "livraison", "transporteur", "expédition", 
                    "bls", "colis", "palette", "enlèvement", "réceptionné"
                ]
                
                # Calcul des scores
                devis_score = sum(10 if kw in all_text else 0 for kw in devis_keywords[:5]) + \
                            sum(5 if kw in all_text else 0 for kw in devis_keywords[5:])
                
                facture_score = sum(10 if kw in all_text else 0 for kw in facture_keywords[:5]) + \
                                sum(5 if kw in all_text else 0 for kw in facture_keywords[5:])
                
                livraison_score = sum(10 if kw in all_text else 0 for kw in livraison_keywords[:5]) + \
                                sum(5 if kw in all_text else 0 for kw in livraison_keywords[5:])
                
                # Normalisation des scores
                total_score = devis_score + facture_score + livraison_score + 10  # +10 pour éviter division par zéro
                
                types["devis_technique"] = min(0.95, devis_score / total_score)
                types["facture"] = min(0.95, facture_score / total_score)
                types["bon_livraison"] = min(0.95, livraison_score / total_score)
                
                # Score pour document standard (fallback)
                types["document_standard"] = max(0.0, 1.0 - max(types["devis_technique"], types["facture"], types["bon_livraison"]))
                
                # Nettoyage si nécessaire
                if is_temp and os.path.exists(file_path):
                    os.unlink(file_path)
                    
                return types
                    
        except Exception as e:
            logger.error(f"Erreur détection type document: {e}")
            
            # Nettoyage en cas d'erreur
            if 'is_temp' in locals() and is_temp and 'file_path' in locals() and os.path.exists(file_path):
                os.unlink(file_path)
                
            return {"document_standard": 1.0}