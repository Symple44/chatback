#!/usr/bin/env python3
# scripts/generate_api_key.py
"""
Script pour générer une clé API et la configurer dans le .env
"""
import os
import sys
import argparse
from pathlib import Path
import re
from datetime import datetime
import json

# Ajouter le répertoire parent au PYTHONPATH
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Importer après avoir configuré le PYTHONPATH
from core.utils.api_key import APIKeyGenerator

def update_env_file(env_path: Path, key: str, value: str) -> bool:
    """
    Met à jour une variable dans le fichier .env
    
    Args:
        env_path: Chemin vers le fichier .env
        key: Nom de la variable à mettre à jour
        value: Nouvelle valeur
        
    Returns:
        True si la mise à jour a réussi, False sinon
    """
    if not env_path.exists():
        print(f"Erreur: Le fichier {env_path} n'existe pas")
        return False
    
    # Lire le contenu actuel
    with open(env_path, "r") as f:
        content = f.read()
    
    # Vérifier si la variable existe déjà
    pattern = re.compile(f"^{re.escape(key)}=.*$", re.MULTILINE)
    if pattern.search(content):
        # Mettre à jour la variable existante
        new_content = pattern.sub(f"{key}={value}", content)
    else:
        # Ajouter la variable à la fin du fichier
        if not content.endswith("\n"):
            content += "\n"
        new_content = content + f"{key}={value}\n"
    
    # Écrire le nouveau contenu
    with open(env_path, "w") as f:
        f.write(new_content)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Génère une clé API et la configure dans le .env")
    parser.add_argument("--env", default=".env", help="Chemin vers le fichier .env")
    parser.add_argument("--length", type=int, default=32, help="Longueur de la clé API")
    parser.add_argument("--prefix", default="gustave", help="Préfixe de la clé API")
    parser.add_argument("--name", default="API Key", help="Nom de la clé API")
    parser.add_argument("--export", action="store_true", help="Exporter la clé dans un fichier JSON")
    parser.add_argument("--no-update", action="store_true", help="Ne pas mettre à jour le fichier .env")
    args = parser.parse_args()
    
    # Générer la clé API
    print("Génération d'une nouvelle clé API...")
    if args.name:
        key_data = APIKeyGenerator.create_key_with_metadata(
            name=args.name,
            scopes=["read", "write", "chat"],
            expiry_days=365  # 1 an par défaut
        )
        api_key = key_data["key"]
    else:
        api_key = APIKeyGenerator.generate_api_key(length=args.length, prefix=args.prefix)
        key_data = {
            "key": api_key,
            "created_at": datetime.utcnow().isoformat(),
            "id": "default"
        }
    
    # Afficher la clé
    print("\n" + "="*50)
    print(f"Nouvelle clé API générée: {api_key}")
    print("="*50 + "\n")
    
    # Mettre à jour le fichier .env si demandé
    if not args.no_update:
        env_path = Path(args.env)
        if update_env_file(env_path, "API_KEY", api_key):
            print(f"✓ Clé API mise à jour dans {env_path}")
        else:
            print(f"✗ Échec de la mise à jour de la clé API dans {env_path}")
    
    # Exporter la clé dans un fichier JSON si demandé
    if args.export:
        export_file = Path("api_keys.json")
        
        # Lire les clés existantes si le fichier existe
        existing_keys = []
        if export_file.exists():
            try:
                with open(export_file, "r") as f:
                    existing_keys = json.load(f)
            except json.JSONDecodeError:
                existing_keys = []
        
        # Ajouter la nouvelle clé
        existing_keys.append(key_data)
        
        # Écrire le fichier
        with open(export_file, "w") as f:
            json.dump(existing_keys, f, indent=2)
        
        print(f"✓ Clé API exportée dans {export_file}")
    
    print("\nUtilisation de la clé API:")
    print("  • Dans les headers HTTP: X-API-Key: " + api_key)
    print("  • Dans l'URL: ?api_key=" + api_key)
    print("\nRappel: Protégez cette clé, elle donne accès à votre API!")

if __name__ == "__main__":
    main()