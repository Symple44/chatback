#!/usr/bin/env python3
import json
import os
from typing import Tuple, List
import datetime

def validate_credentials(file_path: str) -> Tuple[bool, List[str]]:
    """
    Valide un fichier de credentials Google.
    
    Returns:
        Tuple[bool, List[str]]: (valide, liste des erreurs)
    """
    errors = []
    
    # 1. Vérification de l'existence du fichier
    if not os.path.exists(file_path):
        return False, [f"Le fichier {file_path} n'existe pas"]
    
    # 2. Vérification des permissions
    if not os.access(file_path, os.R_OK):
        return False, [f"Le fichier {file_path} n'est pas lisible"]
        
    try:
        # 3. Lecture et parse du JSON
        with open(file_path, 'r') as f:
            creds = json.load(f)
            
        # 4. Vérification des champs requis
        required_fields = {
            'type': str,
            'project_id': str,
            'private_key_id': str,
            'private_key': str,
            'client_email': str,
            'client_id': str,
            'auth_uri': str,
            'token_uri': str
        }
        
        for field, field_type in required_fields.items():
            if field not in creds:
                errors.append(f"Champ manquant: {field}")
            elif not isinstance(creds[field], field_type):
                errors.append(f"Type invalide pour {field}: attendu {field_type}, reçu {type(creds[field])}")
                
        # 5. Vérifications spécifiques
        if creds.get('type') != 'service_account':
            errors.append("Le type doit être 'service_account'")
            
        if 'private_key' in creds:
            pk = creds['private_key']
            if not (pk.startswith('-----BEGIN PRIVATE KEY-----') and 
                   pk.endswith('-----END PRIVATE KEY-----\n')):
                errors.append("Format de private_key invalide")
                
        if 'client_email' in creds:
            email = creds['client_email']
            if not (email.endswith('.gserviceaccount.com') and '@' in email):
                errors.append("Format de client_email invalide")
        
        # 6. Vérification de la date d'expiration
        try:
            created_timestamp = int(creds.get('auth_provider_x509_cert_url', '0'))
            created_date = datetime.datetime.fromtimestamp(created_timestamp)
            if created_date > datetime.datetime.now():
                errors.append("Date de création invalide (dans le futur)")
        except:
            pass  # La date n'est pas toujours présente
            
        return len(errors) == 0, errors
        
    except json.JSONDecodeError as e:
        return False, [f"JSON invalide: {str(e)}"]
    except Exception as e:
        return False, [f"Erreur inattendue: {str(e)}"]

def print_validation_result(file_path: str):
    """Affiche le résultat de la validation."""
    print(f"\nValidation du fichier: {file_path}")
    print("-" * 50)
    
    valid, errors = validate_credentials(file_path)
    
    if valid:
        print("✅ Le fichier de credentials est valide!")
    else:
        print("❌ Le fichier de credentials est invalide:")
        for error in errors:
            print(f"  - {error}")
            
    # Informations supplémentaires si le fichier existe
    if os.path.exists(file_path):
        print("\nInformations sur le fichier:")
        stat = os.stat(file_path)
        print(f"  - Taille: {stat.st_size} bytes")
        print(f"  - Permissions: {oct(stat.st_mode)[-3:]}")
        print(f"  - Dernière modification: {datetime.datetime.fromtimestamp(stat.st_mtime)}")
        
        try:
            with open(file_path, 'r') as f:
                creds = json.load(f)
                print("\nInformations du projet:")
                print(f"  - Project ID: {creds.get('project_id', 'N/A')}")
                print(f"  - Client Email: {creds.get('client_email', 'N/A')}")
        except:
            pass

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "/home/aiuser/Access_AIServ/google/google.json"
    
    print_validation_result(file_path)
