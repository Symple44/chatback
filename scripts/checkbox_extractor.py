#!/usr/bin/env python3
"""
Script de test pour l'extracteur de cases à cocher.
Permet de valider le fonctionnement de la classe CheckboxExtractor.

Utilisation:
python test_checkbox_extractor.py path/to/pdf_with_checkboxes.pdf

Options:
-v, --verbose      Mode verbeux avec plus de détails
-i, --images       Inclure les images des cases à cocher
-p, --pages PAGE   Spécifier les pages à analyser (ex: 1,3,5-7)
-o, --output FILE  Enregistrer les résultats dans un fichier JSON
-d, --debug        Activer le mode de débogage avancé
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
        
        # Mode debug avancé
        if debug:
            import cv2
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            import io
            import fitz  # PyMuPDF
            
            print("\n===== MODE DEBUG =====")
            
            # Ouvrir le PDF
            doc = fitz.open(pdf_path)
            
            # Pour chaque page avec des cases à cocher
            checkboxes_by_page = {}
            for checkbox in results.get("checkboxes", []):
                page_num = checkbox.get("page", 1)
                if page_num not in checkboxes_by_page:
                    checkboxes_by_page[page_num] = []
                checkboxes_by_page[page_num].append(checkbox)
            
            # Visualiser les pages avec les cases détectées
            for page_num, page_checkboxes in checkboxes_by_page.items():
                if page_num <= len(doc):
                    page = doc[page_num-1]  # Convertir 1-indexed en 0-indexed
                    
                    # Convertir la page en image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4:  # RGBA
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    
                    # Afficher l'image
                    plt.figure(figsize=(12, 16))
                    plt.imshow(img)
                    
                    # Ajouter des rectangles pour les cases à cocher
                    ax = plt.gca()
                    for checkbox in page_checkboxes:
                        bbox = checkbox.get("bbox", [0, 0, 0, 0])
                        x, y, x2, y2 = bbox
                        width, height = x2 - x, y2 - y
                        
                        # Couleur selon l'état (vert = cochée, rouge = non cochée)
                        color = 'g' if checkbox.get("checked", False) else 'r'
                        
                        # Dessiner le rectangle
                        rect = Rectangle((x, y), width, height, 
                                         linewidth=2, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)
                        
                        # Afficher le numéro de la case
                        plt.text(x, y-10, f"#{page_checkboxes.index(checkbox)+1}", 
                                 color=color, fontsize=12, backgroundcolor='white')
                    
                    plt.title(f"Page {page_num} - {len(page_checkboxes)} cases détectées")
                    plt.axis('off')
                    plt.tight_layout()
                    
                    # Enregistrer l'image de débogage
                    debug_dir = "debug_output"
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_filename = os.path.join(debug_dir, f"debug_page_{page_num}.png")
                    plt.savefig(debug_filename, dpi=150)
                    print(f"Image de débogage enregistrée: {debug_filename}")
                    
                    # Fermer pour libérer la mémoire
                    plt.close()
            
            # Fermer le document
            doc.close()
        
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
    parser.add_argument("-d", "--debug", action="store_true", help="Mode débogage avancé")
    
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