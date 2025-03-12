# core/auth/dependencies.py
from fastapi import Depends, HTTPException, status, Header, Query, Request
from typing import Optional, List, Callable
import ipaddress

from core.config.config import settings
from core.utils.logger import get_logger

logger = get_logger("auth_dependencies")

async def get_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None),
    api_key: Optional[str] = Query(None)
) -> str:
    """
    Dépendance pour vérifier la clé API.
    
    La clé peut être fournie de deux façons:
    1. Via le header X-API-Key
    2. Via le paramètre de requête api_key
    
    Args:
        request: Requête FastAPI
        x_api_key: Clé API dans le header
        api_key: Clé API en paramètre de requête
        
    Returns:
        Clé API vérifiée
        
    Raises:
        HTTPException: Si la clé est manquante ou invalide
    """
    # Vérifier si on est en mode développement
    if settings.DEBUG:
        # En développement, vérifier si l'IP est locale
        client_ip = request.client.host if request.client else "unknown"
        
        # Liste des IP et réseaux de confiance
        trusted_networks = [
            "127.0.0.1",       # localhost
            "::1",             # localhost IPv6
            "192.168.0.0/24",  # Réseau local 192.168.0.x
            "10.0.0.0/8"       # Réseau privé
        ]
        
        # Vérifier si l'IP est dans un réseau de confiance
        try:
            ip_obj = ipaddress.ip_address(client_ip)
            for network_str in trusted_networks:
                if "/" in network_str:
                    # C'est un réseau
                    network = ipaddress.ip_network(network_str)
                    if ip_obj in network:
                        # En mode dev et avec une IP de confiance, on retourne une clé fictive
                        return "dev_mode_trusted_ip"
                else:
                    # C'est une IP unique
                    if client_ip == network_str:
                        return "dev_mode_trusted_ip"
        except ValueError:
            # IP invalide, continuer avec la vérification normale
            pass
    
    # Récupérer la clé API (priorité au header)
    key = x_api_key or api_key
    
    if not key:
        logger.warning(f"Tentative d'accès sans clé API: {request.url.path}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API manquante"
        )
    
    # Vérifier la clé API
    if key != settings.API_KEY:
        logger.warning(f"Tentative d'accès avec clé API invalide: {request.url.path}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide"
        )
    
    return key

# Fonction pour créer une dépendance qui vérifie des autorisations spécifiques
def require_scopes(scopes: List[str]) -> Callable:
    """
    Crée une dépendance qui vérifie que la clé API a les autorisations requises.
    
    Pour une implémentation complète, cette fonction devrait vérifier les autorisations
    associées à la clé API dans une base de données ou un cache.
    
    Args:
        scopes: Liste des autorisations requises
        
    Returns:
        Dépendance à utiliser dans les routes
    """
    async def check_scopes(api_key: str = Depends(get_api_key)) -> str:
        # Dans une implémentation complète, on récupérerait les autorisations
        # associées à la clé API depuis la base de données
        
        # Pour l'instant, on suppose que la clé a toutes les autorisations
        return api_key
    
    return check_scopes