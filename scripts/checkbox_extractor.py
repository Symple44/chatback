#!/usr/bin/env python3
"""
Script de test pour l'extracteur de cases à cocher.
Version simplifiée sans dépendance matplotlib.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
import logging

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("checkbox_test")

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def test_checkbox_extractor(pdf_path, verbose=False, include_images=False, pages=None, output=None, debug=False):
    """Teste l'extraction des cases à cocher sur un fichier PDF."""
    try:
        # Importer CheckboxExtractor
        from core.document_processing.table_extraction.checkbox_extractor import CheckboxExtractor
        
        # Créer une instance
        extractor = CheckboxExtractor()
        
        # Configuration
        config = {
            "confidence_threshold": 0.5,  # Seuil plus bas pour le test
            "enhance_detection": True,
            "include_images": include_images
        }
        
        # Définir les pages à analyser
        if pages:
            page_range = []
            for part in pages.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    page_range.extend(range(start, end+1))
                else:
                    page_range.append(int(part))
        else:
            page_range = None
        
        # Démarrer le chronomètre
        start_time = time.time()
        
        # Effectuer l'extraction
        results = await extractor.extract_checkboxes_from_pdf(
            pdf_path,
            page_range=page_range,
            config=config
        )
        
        # Temps d'exécution
        execution_time = time.time() - start_time
        
        # Afficher les résultats
        print("\n===== RÉSULTATS D'EXTRACTION DES CASES À COCHER =====")
        print(f"Fichier: {pdf_path}")
        print(f"Temps d'exécution: {execution_time:.2f} secondes")
        print(f"Nombre de pages traitées: {results.get('metadata', {}).get('processed_pages', 0)}")
        print(f"Nombre de cases détectées: {len(results.get('checkboxes', []))}")
        
        # Afficher les valeurs de formulaire
        print("\n--- Valeurs de formulaire ---")
        form_values = results.get("form_values", {})
        if form_values:
            for label, value in form_values.items():
                print(f"- {label}: {value}")
        else:
            print("Aucune valeur de formulaire détectée")
        
        # En mode verbeux, afficher les détails de chaque case
        if verbose:
            print("\n--- Détails des cases à cocher ---")
            for i, checkbox in enumerate(results.get("checkboxes", [])):
                print(f"\nCase #{i+1}:")
                print(f"  Label: {checkbox.get('label', 'Aucun')}")
                print(f"  État: {'Cochée' if checkbox.get('checked', False) else 'Non cochée'}")
                print(f"  Page: {checkbox.get('page', '?')}")
                print(f"  Méthode: {checkbox.get('method', '?')}")
                print(f"  Confiance: {checkbox.get('confidence', 0):.2f}")
                
                if "value" in checkbox and checkbox["value"]:
                    print(f"  Valeur: {checkbox['value']}")
        
        # Enregistrer les résultats dans un fichier si demandé
        if output:
            # Supprimer les images pour simplifier la sortie, sauf si explicitement demandé
            if not include_images and "checkbox_images" in results:
                del results["checkbox_images"]
                
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nRésultats enregistrés dans {output}")
        
        # Mode debug simple (sans matplotlib)
        if debug:
            debug_dir = "debug_output"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Enregistrer les résultats détaillés en JSON
            debug_json = os.path.join(debug_dir, "debug_checkboxes.json")
            with open(debug_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\nDétails de débogage enregistrés dans {debug_json}")
            
            # Statistiques supplémentaires
            methods = {}
            checked_count = 0
            empty_labels = 0
            
            for checkbox in results.get("checkboxes", []):
                method = checkbox.get("method", "unknown")
                if method not in methods:
                    methods[method] = 0
                methods[method] += 1
                
                if checkbox.get("checked", False):
                    checked_count += 1
                
                if not checkbox.get("label", "").strip():
                    empty_labels += 1
            
            print("\n--- Statistiques de débogage ---")
            print(f"Cases cochées: {checked_count} sur {len(results.get('checkboxes', []))}")
            print(f"Cases sans étiquette: {empty_labels}")
            print("Méthodes de détection:")
            for method, count in methods.items():
                print(f"  - {method}: {count}")
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors du test: {e}")
        raise

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Test d'extraction de cases à cocher")
    parser.add_argument("pdf_path", help="Chemin vers le fichier PDF à analyser")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mode verbeux")
    parser.add_argument("-i", "--images", action="store_true", help="Inclure les images des cases à cocher")
    parser.add_argument("-p", "--pages", help="Pages à analyser (ex: 1,3,5-7)")
    parser.add_argument("-o", "--output", help="Enregistrer les résultats dans un fichier JSON")
    parser.add_argument("-d", "--debug", action="store_true", help="Mode débogage")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Erreur: Le fichier {args.pdf_path} n'existe pas.")
        return 1
    
    try:
        asyncio.run(test_checkbox_extractor(
            args.pdf_path,
            verbose=args.verbose,
            include_images=args.images,
            pages=args.pages,
            output=args.output,
            debug=args.debug
        ))
        return 0
    except Exception as e:
        print(f"Erreur: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())