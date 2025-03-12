# core/document_processing/invoice_processor.py
from typing import Dict, List, Any, Optional
import re
import logging
import pandas as pd
import dateutil.parser
from datetime import datetime

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