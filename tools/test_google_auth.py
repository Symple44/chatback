#!/usr/bin/env python3
import os
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import json
from datetime import datetime

def test_credentials(creds_path: str):
    """Test complet des credentials Google Drive."""
    print(f"\nTest des credentials: {creds_path}")
    print("-" * 50)
    
    try:
        # 1. Lecture du fichier
        with open(creds_path, 'r') as f:
            creds_data = json.load(f)
            print("✅ Fichier JSON lu avec succès")
            print(f"  - Project ID: {creds_data.get('project_id')}")
            print(f"  - Client Email: {creds_data.get('client_email')}")
        
        # 2. Création des credentials
        credentials = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        print("✅ Credentials créés avec succès")
        
        # 3. Test de refresh
        if not credentials.valid:
            print("ℹ️  Tentative de refresh des credentials...")
            credentials.refresh(Request())
            print("✅ Refresh réussi")
        
        # 4. Test de création du service
        service = build('drive', 'v3', credentials=credentials)
        print("✅ Service Drive créé avec succès")
        
        # 5. Test d'une requête simple
        files = service.files().list(pageSize=1).execute()
        print("✅ Requête API réussie")
        
        print("\n🎉 Tous les tests ont réussi!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        if "invalid_grant" in str(e):
            print("\nSolutions possibles:")
            print("1. Vérifiez que l'horloge du système est synchronisée")
            print("2. Générez de nouveaux credentials dans la Console Google Cloud")
            print("3. Vérifiez que le compte de service est actif")
        return False

if __name__ == "__main__":
    creds_path = "/home/aiuser/Access_AIServ/google/google.json"
    
    # Afficher l'heure système
    print(f"Heure système: {datetime.now()}")
    
    # Vérifier les permissions
    perms = oct(os.stat(creds_path).st_mode)[-3:]
    print(f"Permissions du fichier: {perms}")
    
    # Tester les credentials
    test_credentials(creds_path)
