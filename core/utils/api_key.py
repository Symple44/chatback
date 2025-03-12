# core/utils/api_key.py
import secrets
import string
import hashlib
import base64
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

class APIKeyGenerator:
    """Utilitaire pour générer et gérer les clés API."""
    
    @staticmethod
    def generate_api_key(length: int = 32, prefix: str = "gustave") -> str:
        """
        Génère une clé API sécurisée.
        
        Args:
            length: Longueur de la clé (hors préfixe)
            prefix: Préfixe de la clé (pour identification)
            
        Returns:
            Clé API au format: prefix_randomstring
        """
        charset = string.ascii_letters + string.digits
        random_part = ''.join(secrets.choice(charset) for _ in range(length))
        
        # Format: prefix_randomstring
        return f"{prefix}_{random_part}"
    
    @staticmethod
    def generate_hmac_key(data: str) -> str:
        """
        Génère une clé HMAC basée sur des données.
        Utile pour créer des signatures ou des clés dérivées.
        
        Args:
            data: Données à utiliser comme base
            
        Returns:
            Clé HMAC au format base64
        """
        salt = uuid.uuid4().hex
        key = hashlib.pbkdf2_hmac(
            'sha256', 
            data.encode('utf-8'), 
            salt.encode('utf-8'), 
            100000  # Nombre d'itérations
        )
        return base64.b64encode(key).decode('utf-8')
    
    @staticmethod
    def generate_key_pair() -> Tuple[str, str]:
        """
        Génère une paire de clés: clé publique et clé secrète.
        
        Returns:
            Tuple (clé_publique, clé_secrète)
        """
        public_key = APIKeyGenerator.generate_api_key(prefix="gustave_pub")
        secret_key = APIKeyGenerator.generate_api_key(length=48, prefix="gustave_sec")
        return public_key, secret_key
    
    @staticmethod
    def create_key_with_metadata(name: str, scopes: list = None, expiry_days: int = 365) -> Dict:
        """
        Crée une clé API avec des métadonnées.
        
        Args:
            name: Nom ou description de la clé
            scopes: Liste des autorisations accordées à cette clé
            expiry_days: Nombre de jours avant expiration
            
        Returns:
            Dictionnaire contenant la clé et ses métadonnées
        """
        expiry = datetime.utcnow() + timedelta(days=expiry_days)
        key = APIKeyGenerator.generate_api_key()
        
        return {
            "key": key,
            "name": name,
            "scopes": scopes or ["read"],
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expiry.isoformat(),
            "id": str(uuid.uuid4())
        }

# Exemple d'utilisation du générateur
if __name__ == "__main__":
    # Générer une simple clé API
    api_key = APIKeyGenerator.generate_api_key()
    print(f"Clé API simple: {api_key}")
    
    # Générer une paire de clés
    public_key, secret_key = APIKeyGenerator.generate_key_pair()
    print(f"Clé publique: {public_key}")
    print(f"Clé secrète: {secret_key}")
    
    # Créer une clé avec métadonnées
    key_data = APIKeyGenerator.create_key_with_metadata(
        name="Clé d'API pour l'intégration client", 
        scopes=["read", "chat"], 
        expiry_days=90
    )
    print(f"Clé avec métadonnées: {key_data}")