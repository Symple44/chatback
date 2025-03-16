#!/usr/bin/env python3
"""
Script de test pour l'extracteur de cases à cocher.
Version améliorée avec une détection plus précise et meilleure gestion des cas limites.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
import logging
from tabulate import tabulate  # Pour une présentation améliorée des résultats

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("checkbox_test")

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def test_checkbox_extractor(pdf_path, verbose=False, include_images=False, pages=None, output=None, 
                                debug=False, strict_mode=True, custom_threshold=None, adaptive=True):
    """
    Teste l'extraction des cases à cocher sur un fichier PDF avec des paramètres améliorés.
    
    Args:
        pdf_path: Chemin vers le fichier PDF à analyser
        verbose: Afficher les détails de chaque case
        include_images: Inclure les images des cases à cocher
        pages: Pages à analyser (format: "1,3,5-7")
        output: Chemin du fichier JSON pour la sauvegarde des résultats
        debug: Mode débogage avec statistiques supplémentaires
        strict_mode: Activer le mode strict pour filtrer les faux positifs
        custom_threshold: Seuil de confiance personnalisé (0.0-1.0)
        adaptive: Activer le traitement adaptatif selon le type de document
    """
    try:
        # Importer CheckboxExtractor
        from core.document_processing.table_extraction.checkbox_extractor import CheckboxExtractor
        
        # Créer une instance
        extractor = CheckboxExtractor()
        
        # Configuration avec paramètres améliorés
        config = {
            "confidence_threshold": custom_threshold if custom_threshold is not None else 0.65,  # Seuil plus élevé par défaut
            "enhance_detection": True,
            "include_images": include_images,
            "strict_mode": strict_mode,
            "adaptive_processing": adaptive,
            "max_checkboxes_ratio": 0.3,  # Limite plus restrictive pour les cases cochées
            "post_process_enabled": True,  # Activer le post-traitement avancé
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
        
        checkboxes = results.get('checkboxes', [])
        print(f"Nombre de cases détectées: {len(checkboxes)}")
        
        # Calcul du ratio de cases cochées
        checked_count = sum(1 for cb in checkboxes if cb.get('checked', False))
        checked_ratio = checked_count / len(checkboxes) if checkboxes else 0
        print(f"Cases cochées: {checked_count}/{len(checkboxes)} ({checked_ratio:.1%})")
        
        # Afficher les avertissements et erreurs éventuels
        if "warnings" in results:
            print("\n--- Avertissements ---")
            for warning in results["warnings"]:
                print(f"- {warning}")
        
        # Afficher les valeurs de formulaire dans un format plus lisible
        print("\n--- Valeurs de formulaire ---")
        form_values = results.get("form_values", {})
        if form_values:
            # Préparation des données pour tabulate
            form_data = [[label, value] for label, value in form_values.items()]
            print(tabulate(form_data, headers=["Champ", "Valeur"], tablefmt="simple"))
        else:
            print("Aucune valeur de formulaire détectée")
        
        # Statistiques sur la qualité de détection
        quality = results.get("metadata", {}).get("quality", {})
        if quality:
            print("\n--- Qualité de détection ---")
            print(f"Confiance moyenne: {quality.get('avg_confidence', 0):.2f}")
            print(f"Cases sans étiquette: {quality.get('unlabeled_count', 0)}")
        
        # En mode verbeux, afficher les détails de chaque case
        if verbose:
            print("\n--- Détails des cases à cocher ---")
            for i, checkbox in enumerate(checkboxes):
                print(f"\nCase #{i+1}:")
                print(f"  Label: {checkbox.get('label', '')}")
                print(f"  État: {'Cochée' if checkbox.get('checked', False) else 'Non cochée'}")
                print(f"  Page: {checkbox.get('page', '?')}")
                print(f"  Méthode: {checkbox.get('method', '?')}")
                print(f"  Confiance: {checkbox.get('confidence', 0):.2f}")
                
                if "value" in checkbox and checkbox["value"]:
                    print(f"  Valeur: {checkbox['value']}")
                
                # Indiquer si la case a été corrigée automatiquement
                if checkbox.get('auto_corrected', False):
                    print(f"  [Auto-corrigée]")
        
        # Afficher les groupes Oui/Non identifiés
        if "structured_results" in results and not verbose:
            questions = results["structured_results"].get("questions", [])
            if questions:
                print("\n--- Questions Oui/Non identifiées ---")
                questions_data = []
                for q in questions:
                    questions_data.append([
                        q.get("text", ""),
                        q.get("answer", "Non spécifié")
                    ])
                print(tabulate(questions_data, headers=["Question", "Réponse"], tablefmt="simple"))
        
        # Enregistrer les résultats dans un fichier si demandé
        if output:
            # Supprimer les images pour simplifier la sortie, sauf si explicitement demandé
            if not include_images and "checkbox_images" in results:
                del results["checkbox_images"]
                
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nRésultats enregistrés dans {output}")
        
        # Mode debug amélioré
        if debug:
            debug_dir = "debug_output"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Enregistrer les résultats détaillés en JSON
            debug_json = os.path.join(debug_dir, f"debug_checkboxes_{Path(pdf_path).stem}.json")
            with open(debug_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\nDétails de débogage enregistrés dans {debug_json}")
            
            # Statistiques supplémentaires plus détaillées
            methods = {}
            confidence_levels = {"low": 0, "medium": 0, "high": 0}
            label_stats = {"with_label": 0, "without_label": 0}
            yes_no_count = 0
            corrections = 0
            
            for checkbox in checkboxes:
                # Statistiques par méthode de détection
                method = checkbox.get("method", "unknown")
                if method not in methods:
                    methods[method] = 0
                methods[method] += 1
                
                # Statistiques par niveau de confiance
                conf = checkbox.get("confidence", 0)
                if conf < 0.65:
                    confidence_levels["low"] += 1
                elif conf < 0.85:
                    confidence_levels["medium"] += 1
                else:
                    confidence_levels["high"] += 1
                
                # Statistiques sur les étiquettes
                if checkbox.get("label", "").strip():
                    label_stats["with_label"] += 1
                else:
                    label_stats["without_label"] += 1
                
                # Comptage des Oui/Non
                if checkbox.get("label", "") in ["Oui", "Non"]:
                    yes_no_count += 1
                
                # Comptage des corrections automatiques
                if checkbox.get("auto_corrected", False):
                    corrections += 1
            
            print("\n--- Statistiques de débogage ---")
            print(f"Cases cochées: {checked_count} sur {len(checkboxes)} ({checked_ratio:.1%})")
            print(f"Cases sans étiquette: {label_stats['without_label']} ({label_stats['without_label']/len(checkboxes):.1%})")
            print(f"Cases Oui/Non: {yes_no_count} ({yes_no_count/len(checkboxes):.1%})")
            print(f"Cases auto-corrigées: {corrections}")
            
            print("\nMéthodes de détection:")
            for method, count in methods.items():
                print(f"  - {method}: {count} ({count/len(checkboxes):.1%})")
            
            print("\nNiveaux de confiance:")
            for level, count in confidence_levels.items():
                print(f"  - {level}: {count} ({count/len(checkboxes):.1%})")
            
            # Informations sur les paramètres adaptatifs
            adaptive_params = results.get("metadata", {}).get("adaptive_params", {})
            if adaptive_params:
                print("\nParamètres adaptatifs:")
                for param, value in adaptive_params.items():
                    print(f"  - {param}: {value}")
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors du test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def main():
    """Fonction principale avec des options de ligne de commande améliorées."""
    parser = argparse.ArgumentParser(description="Test d'extraction de cases à cocher amélioré")
    parser.add_argument("pdf_path", help="Chemin vers le fichier PDF à analyser")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mode verbeux (détails de chaque case)")
    parser.add_argument("-i", "--images", action="store_true", help="Inclure les images des cases à cocher")
    parser.add_argument("-p", "--pages", help="Pages à analyser (ex: 1,3,5-7)")
    parser.add_argument("-o", "--output", help="Enregistrer les résultats dans un fichier JSON")
    parser.add_argument("-d", "--debug", action="store_true", help="Mode débogage avancé")
    parser.add_argument("-n", "--no-strict", action="store_true", help="Désactiver le mode strict")
    parser.add_argument("-t", "--threshold", type=float, help="Seuil de confiance personnalisé (0.0-1.0)")
    parser.add_argument("-a", "--no-adaptive", action="store_true", help="Désactiver le traitement adaptatif")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Erreur: Le fichier {args.pdf_path} n'existe pas.")
        return 1
    
    # Valider le seuil de confiance s'il est fourni
    if args.threshold is not None and not (0 <= args.threshold <= 1):
        print(f"Erreur: Le seuil de confiance doit être entre 0.0 et 1.0.")
        return 1
    
    try:
        asyncio.run(test_checkbox_extractor(
            args.pdf_path,
            verbose=args.verbose,
            include_images=args.images,
            pages=args.pages,
            output=args.output,
            debug=args.debug,
            strict_mode=not args.no_strict,
            custom_threshold=args.threshold,
            adaptive=not args.no_adaptive
        ))
        return 0
    except Exception as e:
        print(f"Erreur: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())