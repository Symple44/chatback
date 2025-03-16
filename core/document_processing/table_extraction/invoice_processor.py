from typing import List, Dict, Any, Optional, Union
import os
import numpy as np
import re
import pandas as pd
from datetime import datetime
import json
import uuid

from core.utils.logger import get_logger

logger = get_logger("invoice_processor")

class InvoiceProcessor:
    """
    Processeur qui analyse et structure les données extraites de factures et documents techniques.
    
    Cette classe intègre les capacités d'extraction de tableaux avec 
    les fonctionnalités de détection de cases à cocher pour une analyse
    de document complète.
    """
    
    def __init__(self):
        """Initialise le processeur de factures."""
        pass
    
    def process(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Traite les tableaux extraits d'une facture.
        
        Args:
            tables: Liste des tableaux extraits
            
        Returns:
            Données structurées de la facture
        """
        try:
            # Initialiser la structure de données
            result = {
                "id": str(uuid.uuid4()),
                "type": "invoice",
                "processing_date": datetime.now().isoformat(),
                "status": "processed",
                "metadata": {},
                "tables_count": len(tables),
                "header": {},
                "line_items": [],
                "totals": {},
                "payment_info": {},
                "additional_info": {}
            }
            
            if not tables:
                result["status"] = "no_tables_found"
                return result
            
            # Détecter les différents types de tableaux
            header_table = None
            line_items_table = None
            totals_table = None
            
            for table in tables:
                table_type = self._identify_table_type(table)
                
                if table_type == "header":
                    header_table = table
                elif table_type == "line_items":
                    line_items_table = table
                elif table_type == "totals":
                    totals_table = table
            
            # Traiter l'en-tête de la facture
            if header_table:
                result["header"] = self._process_header_table(header_table)
            
            # Traiter les éléments de ligne
            if line_items_table:
                result["line_items"] = self._process_line_items_table(line_items_table)
            
            # Traiter les totaux
            if totals_table:
                result["totals"] = self._process_totals_table(totals_table)
            
            # Calculer des statistiques
            if result["line_items"]:
                result["metadata"]["items_count"] = len(result["line_items"])
                result["metadata"]["total_amount"] = sum(item.get("total", 0) for item in result["line_items"])
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur traitement facture: {e}")
            return {
                "id": str(uuid.uuid4()),
                "type": "invoice",
                "processing_date": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    def process_technical_invoice(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Traite les tableaux extraits d'un devis technique.
        
        Args:
            tables: Liste des tableaux extraits
            
        Returns:
            Données structurées du devis technique
        """
        try:
            # Initialiser la structure de données
            result = {
                "id": str(uuid.uuid4()),
                "type": "technical_invoice",
                "processing_date": datetime.now().isoformat(),
                "status": "processed",
                "metadata": {},
                "reference": "",
                "client_info": {},
                "sections": [],
                "products": [],
                "services": [],
                "options": [],
                "totals": {},
                "terms_conditions": []
            }
            
            if not tables:
                result["status"] = "no_tables_found"
                return result
            
            # Traiter chaque tableau
            for table in tables:
                table_type = self._identify_technical_table_type(table)
                
                if table_type == "client_info":
                    result["client_info"] = self._process_client_info_table(table)
                elif table_type == "products":
                    products = self._process_products_table(table)
                    result["products"].extend(products)
                elif table_type == "services":
                    services = self._process_services_table(table)
                    result["services"].extend(services)
                elif table_type == "options":
                    options = self._process_options_table(table)
                    result["options"].extend(options)
                elif table_type == "totals":
                    result["totals"] = self._process_technical_totals_table(table)
                elif table_type == "terms":
                    terms = self._process_terms_table(table)
                    result["terms_conditions"].extend(terms)
            
            # Calculer des métadonnées
            result["metadata"]["products_count"] = len(result["products"])
            result["metadata"]["services_count"] = len(result["services"])
            result["metadata"]["options_count"] = len(result["options"])
            
            # Extraire la référence du document
            result["reference"] = self._extract_reference(tables)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur traitement devis technique: {e}")
            return {
                "id": str(uuid.uuid4()),
                "type": "technical_invoice",
                "processing_date": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    def process_form_data(self, tables: List[Dict[str, Any]], checkbox_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Traite les données de formulaire en combinant tables et cases à cocher.
        
        Args:
            tables: Liste des tableaux extraits
            checkbox_data: Données des cases à cocher extraites
            
        Returns:
            Données structurées du formulaire
        """
        try:
            # Initialiser la structure de données
            result = {
                "id": str(uuid.uuid4()),
                "type": "form",
                "processing_date": datetime.now().isoformat(),
                "status": "processed",
                "metadata": {},
                "form_sections": {},
                "table_data": [],
                "checkbox_data": {},
                "merged_data": {}
            }
            
            # Traiter les tableaux
            if tables:
                for table in tables:
                    table_data = self._process_form_table(table)
                    result["table_data"].append(table_data)
            
            # Traiter les données de cases à cocher
            if checkbox_data:
                # Organiser les cases à cocher par section
                sections = checkbox_data.get("form_sections", {})
                values = checkbox_data.get("form_values", {})
                checkboxes = checkbox_data.get("checkboxes", [])
                
                # Ajouter les sections de formulaire
                for section_name, checkbox_ids in sections.items():
                    if section_name not in result["form_sections"]:
                        result["form_sections"][section_name] = {
                            "type": "checkbox_group",
                            "items": []
                        }
                    
                    # Ajouter les cases à cocher de cette section
                    for checkbox_id in checkbox_ids:
                        # Trouver la case à cocher correspondante
                        checkbox = next((cb for cb in checkboxes if cb.get("id") == checkbox_id), None)
                        if checkbox:
                            result["form_sections"][section_name]["items"].append({
                                "id": checkbox.get("id"),
                                "label": checkbox.get("label", ""),
                                "value": checkbox.get("is_checked", False),
                                "field_name": checkbox.get("field_name", "")
                            })
                
                # Ajouter les valeurs de formulaire
                result["checkbox_data"] = values
            
            # Fusionner les données des tableaux et des cases à cocher
            merged_data = {}
            
            # Ajouter les données des tableaux
            if result["table_data"]:
                for table_entry in result["table_data"]:
                    if "data" in table_entry and isinstance(table_entry["data"], list):
                        for row in table_entry["data"]:
                            if isinstance(row, dict):
                                for key, value in row.items():
                                    # Nettoyer les noms de clés
                                    clean_key = self._normalize_key(key)
                                    if clean_key:
                                        merged_data[clean_key] = value
            
            # Ajouter les données des cases à cocher
            if result["checkbox_data"]:
                for key, value in result["checkbox_data"].items():
                    # Nettoyer les noms de clés
                    clean_key = self._normalize_key(key)
                    if clean_key:
                        merged_data[clean_key] = value
            
            result["merged_data"] = merged_data
            result["metadata"]["table_count"] = len(tables) if tables else 0
            result["metadata"]["checkbox_count"] = len(checkboxes) if checkbox_data and "checkboxes" in checkbox_data else 0
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur traitement formulaire: {e}")
            return {
                "id": str(uuid.uuid4()),
                "type": "form",
                "processing_date": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    def detect_document_type(self, file_obj: Union[str, bytes, object]) -> Dict[str, float]:
        """
        Détecte le type du document.
        
        Args:
            file_obj: Fichier à analyser
            
        Returns:
            Dictionnaire des probabilités par type de document
        """
        # Implémentation simple (à améliorer avec une IA dédiée)
        # Retourne un dictionnaire de probabilités
        return {
            "invoice": 0.25,
            "technical_invoice": 0.25,
            "form": 0.25,
            "generic": 0.25
        }
    
    def _identify_table_type(self, table: Dict[str, Any]) -> str:
        """
        Identifie le type de tableau dans une facture.
        
        Args:
            table: Tableau à identifier
            
        Returns:
            Type du tableau ("header", "line_items", "totals", "unknown")
        """
        if not table or "data" not in table:
            return "unknown"
            
        data = table["data"]
        
        # Convertir en DataFrame si nécessaire
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                return "unknown"
        
        # Vérifier les colonnes pour déterminer le type
        columns = [str(col).lower() for col in data.columns]
        
        # Vérifier si c'est un tableau d'en-tête
        header_keywords = ["facture", "invoice", "client", "date", "numéro", "number", "adresse", "address"]
        if any(keyword in ' '.join(columns) for keyword in header_keywords) and len(data) < 5:
            return "header"
        
        # Vérifier si c'est un tableau d'éléments de ligne
        line_item_keywords = ["description", "quantité", "quantity", "prix", "price", "montant", "amount", "total"]
        if sum(1 for keyword in line_item_keywords if any(keyword in col for col in columns)) >= 2:
            return "line_items"
        
        # Vérifier si c'est un tableau de totaux
        total_keywords = ["total", "sous-total", "subtotal", "tva", "vat", "tax", "ttc", "ht"]
        if any(keyword in ' '.join(columns) for keyword in total_keywords) and len(data) < 10:
            return "totals"
        
        return "unknown"
    
    def _identify_technical_table_type(self, table: Dict[str, Any]) -> str:
        """
        Identifie le type de tableau dans un devis technique.
        
        Args:
            table: Tableau à identifier
            
        Returns:
            Type du tableau
        """
        if not table or "data" not in table:
            return "unknown"
            
        data = table["data"]
        
        # Convertir en DataFrame si nécessaire
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                return "unknown"
        
        # Vérifier les colonnes pour déterminer le type
        columns = [str(col).lower() for col in data.columns]
        
        # Vérifier si c'est un tableau d'information client
        client_keywords = ["client", "nom", "name", "adresse", "address", "contact", "tel", "email"]
        if any(keyword in ' '.join(columns) for keyword in client_keywords) and len(data) < 5:
            return "client_info"
        
        # Vérifier si c'est un tableau de produits
        product_keywords = ["produit", "product", "référence", "reference", "modèle", "model"]
        if any(keyword in ' '.join(columns) for keyword in product_keywords):
            return "products"
        
        # Vérifier si c'est un tableau de services
        service_keywords = ["service", "prestation", "intervention", "maintenance", "support"]
        if any(keyword in ' '.join(columns) for keyword in service_keywords):
            return "services"
        
        # Vérifier si c'est un tableau d'options
        option_keywords = ["option", "accessoire", "accessory", "supplément", "add-on"]
        if any(keyword in ' '.join(columns) for keyword in option_keywords):
            return "options"
        
        # Vérifier si c'est un tableau de totaux
        total_keywords = ["total", "sous-total", "subtotal", "tva", "vat", "tax", "ttc", "ht"]
        if any(keyword in ' '.join(columns) for keyword in total_keywords) and len(data) < 10:
            return "totals"
        
        # Vérifier si c'est un tableau de conditions
        terms_keywords = ["condition", "term", "clause", "garantie", "warranty", "livraison", "delivery"]
        if any(keyword in ' '.join(columns) for keyword in terms_keywords):
            return "terms"
        
        return "unknown"
    
    def _process_header_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite un tableau d'en-tête de facture.
        
        Args:
            table: Tableau d'en-tête
            
        Returns:
            Informations d'en-tête structurées
        """
        header = {}
        
        if not table or "data" not in table:
            return header
            
        data = table["data"]
        
        # Convertir en DataFrame si nécessaire
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                return header
        
        # Extraire les informations clés
        # Pour un tableau d'en-tête, nous convertissons les colonnes en un dictionnaire plat
        for col in data.columns:
            col_lower = str(col).lower()
            
            # Identifier des champs spécifiques
            if "facture" in col_lower or "invoice" in col_lower:
                header["invoice_number"] = str(data[col].iloc[0]) if not data[col].empty else ""
            elif "date" in col_lower:
                header["invoice_date"] = str(data[col].iloc[0]) if not data[col].empty else ""
            elif "client" in col_lower or "customer" in col_lower:
                header["client_name"] = str(data[col].iloc[0]) if not data[col].empty else ""
            elif "adresse" in col_lower or "address" in col_lower:
                header["client_address"] = str(data[col].iloc[0]) if not data[col].empty else ""
            elif "échéance" in col_lower or "due" in col_lower:
                header["due_date"] = str(data[col].iloc[0]) if not data[col].empty else ""
            else:
                # Champs génériques
                header[col] = str(data[col].iloc[0]) if not data[col].empty else ""
        
        return header
    
    def _process_line_items_table(self, table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Traite un tableau d'éléments de ligne de facture.
        
        Args:
            table: Tableau d'éléments de ligne
            
        Returns:
            Liste des éléments de ligne structurés
        """
        line_items = []
        
        if not table or "data" not in table:
            return line_items
            
        data = table["data"]
        
        # Convertir en DataFrame si nécessaire
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                return line_items
        
        # Normaliser les noms de colonnes
        column_mapping = {}
        for col in data.columns:
            col_lower = str(col).lower()
            if "description" in col_lower or "produit" in col_lower or "service" in col_lower:
                column_mapping[col] = "description"
            elif "quantité" in col_lower or "quantity" in col_lower or "qty" in col_lower:
                column_mapping[col] = "quantity"
            elif "prix" in col_lower or "price" in col_lower or "unit" in col_lower:
                column_mapping[col] = "price"
            elif "total" in col_lower or "montant" in col_lower or "amount" in col_lower:
                column_mapping[col] = "total"
            elif "tva" in col_lower or "vat" in col_lower or "tax" in col_lower:
                column_mapping[col] = "tax"
            else:
                column_mapping[col] = col_lower
        
        # Renommer les colonnes
        data_renamed = data.rename(columns=column_mapping)
        
        # Convertir en liste de dictionnaires
        for _, row in data_renamed.iterrows():
            item = {}
            for col in data_renamed.columns:
                value = row[col]
                
                # Convertir les colonnes numériques
                if col in ["quantity", "price", "total", "tax"]:
                    try:
                        if pd.notna(value):
                            # Si c'est une chaîne, nettoyer et convertir
                            if isinstance(value, str):
                                # Supprimer les caractères non numériques sauf le point décimal
                                value = re.sub(r'[^\d.,]', '', value)
                                value = value.replace(',', '.')
                            item[col] = float(value)
                        else:
                            item[col] = 0.0
                    except (ValueError, TypeError):
                        item[col] = 0.0
                else:
                    item[col] = str(value) if pd.notna(value) else ""
            
            line_items.append(item)
        
        return line_items
    
    def _process_totals_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite un tableau de totaux de facture.
        
        Args:
            table: Tableau de totaux
            
        Returns:
            Informations de totaux structurées
        """
        totals = {}
        
        if not table or "data" not in table:
            return totals
            
        data = table["data"]
        
        # Convertir en DataFrame si nécessaire
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                return totals
        
        # Extraire les informations de totaux
        for _, row in data.iterrows():
            row_dict = row.to_dict()
            label = None
            value = None
            
            # Trouver la colonne de label et la colonne de valeur
            for col, val in row_dict.items():
                col_str = str(col).lower()
                val_str = str(val).lower() if pd.notna(val) else ""
                
                # Si la colonne ou la valeur contient un mot-clé de total
                if any(keyword in col_str for keyword in ["total", "sous", "sub", "tva", "vat", "tax", "ht", "ttc"]):
                    label = col
                elif any(keyword in val_str for keyword in ["total", "sous", "sub", "tva", "vat", "tax", "ht", "ttc"]):
                    label = val
                elif re.search(r'\d+[,.]\d+', str(val)):
                    # Si la valeur ressemble à un montant (chiffres avec virgule/point)
                    value = val
            
            if label and value:
                # Nettoyer le label
                label_clean = str(label).lower().strip()
                
                # Convertir la valeur en nombre
                try:
                    if isinstance(value, str):
                        value = re.sub(r'[^\d.,]', '', value)
                        value = value.replace(',', '.')
                    value_clean = float(value)
                except (ValueError, TypeError):
                    value_clean = 0.0
                
                # Mapper les labels courants
                if "total" in label_clean and "ht" in label_clean:
                    totals["total_ht"] = value_clean
                elif "total" in label_clean and "ttc" in label_clean:
                    totals["total_ttc"] = value_clean
                elif "tva" in label_clean or "vat" in label_clean or "tax" in label_clean:
                    totals["tax"] = value_clean
                elif "sous" in label_clean or "sub" in label_clean:
                    totals["subtotal"] = value_clean
                else:
                    totals[label_clean] = value_clean
        
        return totals
    
    def _process_client_info_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite un tableau d'informations client.
        
        Args:
            table: Tableau d'informations client
            
        Returns:
            Informations client structurées
        """
        client_info = {}
        
        if not table or "data" not in table:
            return client_info
            
        data = table["data"]
        
        # Convertir en DataFrame si nécessaire
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                return client_info
        
        # Parcourir les données pour extraire les informations client
        for col in data.columns:
            col_lower = str(col).lower()
            
            # Identifier des champs spécifiques
            if "nom" in col_lower or "name" in col_lower:
                client_info["name"] = str(data[col].iloc[0]) if not data[col].empty else ""
            elif "adresse" in col_lower or "address" in col_lower:
                client_info["address"] = str(data[col].iloc[0]) if not data[col].empty else ""
            elif "email" in col_lower or "courriel" in col_lower:
                client_info["email"] = str(data[col].iloc[0]) if not data[col].empty else ""
            elif "téléphone" in col_lower or "telephone" in col_lower or "phone" in col_lower:
                client_info["phone"] = str(data[col].iloc[0]) if not data[col].empty else ""
            elif "contact" in col_lower:
                client_info["contact"] = str(data[col].iloc[0]) if not data[col].empty else ""
            else:
                # Champs génériques
                client_info[col_lower] = str(data[col].iloc[0]) if not data[col].empty else ""
        
        return client_info
    
    def _process_products_table(self, table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Traite un tableau de produits.
        
        Args:
            table: Tableau de produits
            
        Returns:
            Liste des produits structurés
        """
        products = []
        
        if not table or "data" not in table:
            return products
            
        data = table["data"]
        
        # Convertir en DataFrame si nécessaire
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                return products
        
        # Normaliser les noms de colonnes
        column_mapping = {}
        for col in data.columns:
            col_lower = str(col).lower()
            if "référence" in col_lower or "reference" in col_lower or "ref" in col_lower:
                column_mapping[col] = "reference"
            elif "produit" in col_lower or "product" in col_lower or "description" in col_lower:
                column_mapping[col] = "description"
            elif "quantité" in col_lower or "quantity" in col_lower or "qty" in col_lower:
                column_mapping[col] = "quantity"
            elif "prix" in col_lower or "price" in col_lower or "unit" in col_lower:
                column_mapping[col] = "price"
            elif "total" in col_lower or "montant" in col_lower or "amount" in col_lower:
                column_mapping[col] = "total"
            else:
                column_mapping[col] = col_lower
        
        # Renommer les colonnes
        data_renamed = data.rename(columns=column_mapping)
        
        # Convertir en liste de dictionnaires
        for _, row in data_renamed.iterrows():
            item = {
                "type": "product",
                "id": str(uuid.uuid4())
            }
            
            for col in data_renamed.columns:
                value = row[col]
                
                # Convertir les colonnes numériques
                if col in ["quantity", "price", "total"]:
                    try:
                        if pd.notna(value):
                            # Si c'est une chaîne, nettoyer et convertir
                            if isinstance(value, str):
                                value = re.sub(r'[^\d.,]', '', value)
                                value = value.replace(',', '.')
                            item[col] = float(value)
                        else:
                            item[col] = 0.0
                    except (ValueError, TypeError):
                        item[col] = 0.0
                else:
                    item[col] = str(value) if pd.notna(value) else ""
            
            products.append(item)
        
        return products
    
    def _process_services_table(self, table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Traite un tableau de services.
        
        Args:
            table: Tableau de services
            
        Returns:
            Liste des services structurés
        """
        services = []
        
        if not table or "data" not in table:
            return services
            
        data = table["data"]
        
        # Convertir en DataFrame si nécessaire
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                return services
        
        # Normaliser les noms de colonnes
        column_mapping = {}
        for col in data.columns:
            col_lower = str(col).lower()
            if "service" in col_lower or "prestation" in col_lower or "description" in col_lower:
                column_mapping[col] = "description"
            elif "durée" in col_lower or "duration" in col_lower:
                column_mapping[col] = "duration"
            elif "quantité" in col_lower or "quantity" in col_lower or "qty" in col_lower:
                column_mapping[col] = "quantity"
            elif "prix" in col_lower or "price" in col_lower or "tarif" in col_lower:
                column_mapping[col] = "price"
            elif "total" in col_lower or "montant" in col_lower or "amount" in col_lower:
                column_mapping[col] = "total"
            else:
                column_mapping[col] = col_lower
        
        # Renommer les colonnes
        data_renamed = data.rename(columns=column_mapping)
        
        # Convertir en liste de dictionnaires
        for _, row in data_renamed.iterrows():
            item = {
                "type": "service",
                "id": str(uuid.uuid4())
            }
            
            for col in data_renamed.columns:
                value = row[col]
                
                # Convertir les colonnes numériques
                if col in ["quantity", "price", "total"]:
                    try:
                        if pd.notna(value):
                            # Si c'est une chaîne, nettoyer et convertir
                            if isinstance(value, str):
                                value = re.sub(r'[^\d.,]', '', value)
                                value = value.replace(',', '.')
                            item[col] = float(value)
                        else:
                            item[col] = 0.0
                    except (ValueError, TypeError):
                        item[col] = 0.0
                else:
                    item[col] = str(value) if pd.notna(value) else ""
            
            services.append(item)
        
        return services
    
    def _process_options_table(self, table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Traite un tableau d'options.
        
        Args:
            table: Tableau d'options
            
        Returns:
            Liste des options structurées
        """
        options = []
        
        if not table or "data" not in table:
            return options
            
        data = table["data"]
        
        # Convertir en DataFrame si nécessaire
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                return options
        
        # Normaliser les noms de colonnes
        column_mapping = {}
        for col in data.columns:
            col_lower = str(col).lower()
            if "option" in col_lower or "accessoire" in col_lower or "description" in col_lower:
                column_mapping[col] = "description"
            elif "quantité" in col_lower or "quantity" in col_lower or "qty" in col_lower:
                column_mapping[col] = "quantity"
            elif "prix" in col_lower or "price" in col_lower or "tarif" in col_lower:
                column_mapping[col] = "price"
            elif "total" in col_lower or "montant" in col_lower or "amount" in col_lower:
                column_mapping[col] = "total"
            elif "inclus" in col_lower or "included" in col_lower:
                column_mapping[col] = "included"
            else:
                column_mapping[col] = col_lower
        
        # Renommer les colonnes
        data_renamed = data.rename(columns=column_mapping)
        
        # Convertir en liste de dictionnaires
        for _, row in data_renamed.iterrows():
            item = {
                "type": "option",
                "id": str(uuid.uuid4())
            }
            
            for col in data_renamed.columns:
                value = row[col]
                
                # Vérifier si l'option est incluse
                if col == "included":
                    item[col] = self._parse_boolean_value(value)
                # Convertir les colonnes numériques
                elif col in ["quantity", "price", "total"]:
                    try:
                        if pd.notna(value):
                            # Si c'est une chaîne, nettoyer et convertir
                            if isinstance(value, str):
                                value = re.sub(r'[^\d.,]', '', value)
                                value = value.replace(',', '.')
                            item[col] = float(value)
                        else:
                            item[col] = 0.0
                    except (ValueError, TypeError):
                        item[col] = 0.0
                else:
                    item[col] = str(value) if pd.notna(value) else ""
            
            options.append(item)
        
        return options
    
    def _process_technical_totals_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite un tableau de totaux pour un devis technique.
        
        Args:
            table: Tableau de totaux
            
        Returns:
            Informations de totaux structurées
        """
        # Réutiliser la méthode pour les factures
        return self._process_totals_table(table)
    
    def _process_terms_table(self, table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Traite un tableau de conditions.
        
        Args:
            table: Tableau de conditions
            
        Returns:
            Liste des conditions structurées
        """
        terms = []
        
        if not table or "data" not in table:
            return terms
            
        data = table["data"]
        
        # Convertir en DataFrame si nécessaire
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                return terms
        
        # Convertir en liste de termes
        for _, row in data.iterrows():
            row_dict = row.to_dict()
            
            # Trouver la colonne contenant le texte des conditions
            for col, val in row_dict.items():
                if pd.notna(val) and isinstance(val, str) and len(val) > 10:
                    terms.append({
                        "id": str(uuid.uuid4()),
                        "text": val,
                        "category": self._identify_term_category(val)
                    })
        
        return terms
    
    def _process_form_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite un tableau de formulaire.
        
        Args:
            table: Tableau de formulaire
            
        Returns:
            Données structurées du formulaire
        """
        result = {
            "id": str(uuid.uuid4()),
            "page": table.get("page", 1),
            "table_id": table.get("table_id", 1),
            "data": [],
            "metadata": {}
        }
        
        if not table or "data" not in table:
            return result
            
        data = table["data"]
        
        # Convertir en DataFrame si nécessaire
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                return result
        
        # Traitement spécifique pour les formulaires
        # Pour un formulaire, nous voulons conserver la structure du tableau
        # tout en normalisant les valeurs
        
        # Convertir le DataFrame en liste de dictionnaires
        if hasattr(data, 'to_dict'):
            table_data = data.to_dict(orient='records')
            
            # Nettoyer les valeurs
            for row in table_data:
                cleaned_row = {}
                for key, value in row.items():
                    # Vérifier si la valeur ressemble à un booléen
                    if isinstance(value, str) and value.lower() in ['oui', 'non', 'yes', 'no', 'true', 'false']:
                        cleaned_row[key] = self._parse_boolean_value(value)
                    else:
                        cleaned_row[key] = value
                
                result["data"].append(cleaned_row)
        
        # Ajouter des métadonnées
        result["metadata"]["rows"] = len(result["data"])
        result["metadata"]["columns"] = len(data.columns) if hasattr(data, 'columns') else 0
        
        return result
    
    def _extract_reference(self, tables: List[Dict[str, Any]]) -> str:
        """
        Extrait la référence du document à partir des tableaux.
        
        Args:
            tables: Liste des tableaux
            
        Returns:
            Référence extraite
        """
        reference = ""
        
        # Parcourir tous les tableaux pour trouver des références
        for table in tables:
            if not table or "data" not in table:
                continue
                
            data = table["data"]
            
            # Convertir en DataFrame si nécessaire
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, list):
                    data = pd.DataFrame(data)
                else:
                    continue
            
            # Rechercher des motifs de référence dans les colonnes et les valeurs
            for col in data.columns:
                col_lower = str(col).lower()
                
                # Vérifier si la colonne est liée à une référence
                if any(keyword in col_lower for keyword in ["référence", "reference", "ref", "devis", "numéro", "number"]):
                    for val in data[col].dropna():
                        # Vérifier si la valeur ressemble à une référence
                        val_str = str(val)
                        if re.search(r'\b[A-Z0-9]{2,}[-/]?[A-Z0-9]{2,}\b', val_str, re.IGNORECASE):
                            reference = val_str
                            return reference
        
        return reference
    
    def _identify_term_category(self, text: str) -> str:
        """
        Identifie la catégorie d'une condition.
        
        Args:
            text: Texte de la condition
            
        Returns:
            Catégorie identifiée
        """
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ["garantie", "warranty", "guarantee"]):
            return "warranty"
        elif any(keyword in text_lower for keyword in ["livraison", "delivery", "shipping"]):
            return "delivery"
        elif any(keyword in text_lower for keyword in ["paiement", "payment", "règlement", "settlement"]):
            return "payment"
        elif any(keyword in text_lower for keyword in ["annulation", "cancellation", "résiliation", "termination"]):
            return "cancellation"
        elif any(keyword in text_lower for keyword in ["confidentialité", "confidentiality", "privacy"]):
            return "confidentiality"
        else:
            return "general"
    
    def _parse_boolean_value(self, value) -> bool:
        """
        Analyse une valeur pour déterminer si elle représente un booléen.
        
        Args:
            value: Valeur à analyser
            
        Returns:
            Valeur booléenne
        """
        if isinstance(value, bool):
            return value
            
        if pd.isna(value):
            return False
            
        if isinstance(value, str):
            value_lower = value.lower().strip()
            return value_lower in ["oui", "yes", "true", "vrai", "1", "x", "✓", "checked", "coché"]
        
        if isinstance(value, (int, float)):
            return value > 0
            
        return False
    
    def _normalize_key(self, key: str) -> str:
        """
        Normalise un nom de clé pour la fusion des données.
        
        Args:
            key: Nom de clé à normaliser
            
        Returns:
            Nom de clé normalisé
        """
        if pd.isna(key) or not key:
            return ""
            
        # Convertir en chaîne
        key_str = str(key).strip()
        
        # Supprimer les caractères spéciaux et convertir en snake_case
        key_clean = re.sub(r'[^\w\s]', '', key_str)
        key_clean = re.sub(r'\s+', '_', key_clean)
        key_clean = key_clean.lower()
        
        # Limiter la longueur
        key_clean = key_clean[:50]
        
        return key_clean