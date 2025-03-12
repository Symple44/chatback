# core/middleware/security.py
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import ipaddress
import time
from datetime import datetime
import uuid
from typing import List, Set, Dict, Optional, Union, Callable

from core.config.config import settings
from core.utils.logger import get_logger

logger = get_logger("api_security")

class APISecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware pour sécuriser l'API avec une vérification de clé API.
    
    Caractéristiques:
    - Vérification de clé API dans les headers
    - Exemptions pour des chemins spécifiques
    - Exemptions pour des adresses IP spécifiques
    - Mode de développement pour faciliter les tests locaux
    - Limitation de débit (rate limiting) basique
    """
    
    def __init__(
        self, 
        app,
        api_key: str = None,
        excluded_paths: List[str] = None,
        trusted_ips: List[str] = None,
        protected_docs: bool = False,
        dev_mode: bool = False,
        rate_limit_enabled: bool = True,
        rate_limit_max: int = 100,
        rate_limit_window: int = 60,
    ):
        super().__init__(app)
        self.api_key = api_key or settings.security.API_KEY
        self.excluded_paths = set(excluded_paths or [])
        self.trusted_ips = set(trusted_ips or [])
        self.dev_mode = dev_mode or settings.DEBUG
        self.protected_docs = protected_docs
        self.rate_limit_enabled = rate_limit_enabled
        self.rate_limit_max = rate_limit_max
        self.rate_limit_window = rate_limit_window
        self.rate_limit_storage: Dict[str, List[float]] = {}
        
        # Conversion des IP/réseaux en objets ipaddress pour faciliter la vérification
        self.trusted_networks = []
        for ip in self.trusted_ips:
            try:
                if "/" in ip:
                    # C'est un réseau, comme 192.168.0.0/24
                    self.trusted_networks.append(ipaddress.ip_network(ip))
                else:
                    # C'est une IP unique
                    self.trusted_networks.append(ipaddress.ip_address(ip))
            except ValueError:
                logger.warning(f"IP ou réseau invalide ignoré: {ip}")
        
        # Si la documentation n'est pas protégée, ajouter les chemins de documentation aux exclusions
        if not self.protected_docs:
            self.excluded_paths.update(["/docs", "/redoc", "/openapi.json"])
        
        # Toujours exclure les chemins de health check
        self.excluded_paths.update(["/api/health", "/api/ping"])
        
        # Logger la configuration
        self._log_configuration()
    
    def _log_configuration(self):
        """Logger la configuration du middleware."""
        logger.info("Configuration de l'API Security Middleware:")
        logger.info(f"Mode développement: {self.dev_mode}")
        logger.info(f"API Key configurée: {'Oui' if self.api_key else 'Non'}")
        logger.info(f"Chemins exclus: {len(self.excluded_paths)}")
        logger.info(f"Nombre de réseaux/IPs de confiance: {len(self.trusted_networks)}")
        logger.info(f"Rate limiting activé: {self.rate_limit_enabled}")
        
    def _is_path_excluded(self, path: str) -> bool:
        """Vérifier si le chemin est exclu de la vérification."""
        # Vérification exacte
        if path in self.excluded_paths:
            return True
        
        # Vérification par préfixe
        for excluded in self.excluded_paths:
            if excluded.endswith("*") and path.startswith(excluded[:-1]):
                return True
        
        return False
    
    def _is_ip_trusted(self, client_ip: str) -> bool:
        """Vérifier si l'IP du client est de confiance."""
        try:
            ip_obj = ipaddress.ip_address(client_ip)
            
            # Vérifier les adresses IP et réseaux de confiance
            for network in self.trusted_networks:
                if isinstance(network, ipaddress.IPv4Network) or isinstance(network, ipaddress.IPv6Network):
                    if ip_obj in network:
                        return True
                elif ip_obj == network:
                    return True
                    
            return False
        except ValueError:
            logger.warning(f"IP client invalide: {client_ip}")
            return False
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """
        Vérifier les limites de débit pour une IP donnée.
        Retourne True si la requête est autorisée, False sinon.
        """
        if not self.rate_limit_enabled:
            return True
            
        current_time = time.time()
        
        # Initialiser ou nettoyer les timestamps pour l'IP
        if client_ip not in self.rate_limit_storage:
            self.rate_limit_storage[client_ip] = []
        
        # Supprimer les timestamps plus anciens que la fenêtre
        self.rate_limit_storage[client_ip] = [
            t for t in self.rate_limit_storage[client_ip]
            if current_time - t < self.rate_limit_window
        ]
        
        # Vérifier si le nombre de requêtes dépasse la limite
        if len(self.rate_limit_storage[client_ip]) >= self.rate_limit_max:
            return False
        
        # Ajouter le timestamp de la requête actuelle
        self.rate_limit_storage[client_ip].append(current_time)
        return True
    
    async def dispatch(self, request: Request, call_next):
        """Traiter la requête et vérifier l'authentification."""
        # Obtenir l'IP du client
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        
        # Vérifier le rate limit
        if not self._check_rate_limit(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Trop de requêtes",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": str(uuid.uuid4())
                }
            )
        
        # Chemins de documentation
        docs_paths = ["/docs", "/redoc", "/openapi.json"]
        is_docs_path = path in docs_paths or any(path.startswith(p + "#") for p in docs_paths)
        
        # Pour les chemins de documentation, vérifier le domaine et l'IP
        if is_docs_path and self.protected_docs:
            # Vérification du domaine
            host = request.headers.get("Host", "")
            
            # Si le domaine est api.symple.fr, bloquer l'accès même si l'IP est de confiance
            if "api.symple.fr" in host:
                logger.warning(f"Tentative d'accès à la documentation depuis le domaine interdit: {host} (IP: {client_ip})")
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "detail": "Accès à la documentation refusé depuis ce domaine",
                        "error_code": "DOCS_ACCESS_DENIED",
                        "timestamp": datetime.utcnow().isoformat(),
                        "request_id": str(uuid.uuid4())
                    }
                )
            
            # Vérification de l'IP seulement si le domaine est autorisé
            if not self._is_ip_trusted(client_ip) or not self.dev_mode:
                logger.warning(f"Tentative d'accès non autorisé à la documentation depuis {client_ip}, host: {host}")
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "detail": "Accès à la documentation refusé. Utilisez une IP locale ou un domaine autorisé.",
                        "error_code": "DOCS_ACCESS_DENIED",
                        "timestamp": datetime.utcnow().isoformat(),
                        "request_id": str(uuid.uuid4())
                    }
                )
            
        # Vérifier les exemptions standard
        if self._is_path_excluded(path) or self.dev_mode or self._is_ip_trusted(client_ip):
            return await call_next(request)
        
        # Vérifier la clé API
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        
        if not api_key or api_key != self.api_key:
            logger.warning(f"Tentative d'accès non autorisé depuis {client_ip} vers {path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": "Clé API invalide ou manquante",
                    "error_code": "INVALID_API_KEY",
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": str(uuid.uuid4())
                }
            )
        
        # Requête valide
        return await call_next(request)