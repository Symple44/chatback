# core/llm/auth_manager.py
from huggingface_hub import login, HfApi
import os
from typing import Optional
from core.config.config import settings
from core.utils.logger import get_logger

logger = get_logger("auth_manager")

class HuggingFaceAuthManager:
    def __init__(self):
        """Initialise le gestionnaire d'authentification Hugging Face."""
        self.token = settings.HUGGING_FACE_HUB_TOKEN
        self.api = HfApi()
        self._is_authenticated = False

    async def setup_auth(self) -> bool:
        """
        Configure l'authentification Hugging Face.
        
        Returns:
            bool: True si l'authentification réussit, False sinon
        """
        try:
            if not self.token:
                logger.warning("Token Hugging Face non configuré")
                return False

            # Test du token existant
            if self._check_existing_token():
                logger.info("Token Hugging Face existant valide")
                self._is_authenticated = True
                return True

            # Tentative de login
            login(token=self.token)
            
            # Vérification de l'authentification
            if self.verify_auth():
                logger.info("Authentification Hugging Face réussie")
                self._is_authenticated = True
                return True
            
            logger.error("Échec de l'authentification Hugging Face")
            return False

        except Exception as e:
            logger.error(f"Erreur d'authentification Hugging Face: {e}")
            return False

    def _check_existing_token(self) -> bool:
        """
        Vérifie si un token existant est valide.
        
        Returns:
            bool: True si le token est valide
        """
        try:
            # Vérifie si le token est déjà dans l'environnement
            existing_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if not existing_token:
                return False

            # Vérifie si le token est valide
            return self.verify_auth()

        except Exception as e:
            logger.error(f"Erreur vérification token existant: {e}")
            return False

    def verify_auth(self) -> bool:
        """
        Vérifie si l'authentification est valide.
        
        Returns:
            bool: True si l'authentification est valide
        """
        try:
            # Tente de récupérer les informations du compte
            whoami = self.api.whoami()
            return bool(whoami)

        except Exception as e:
            logger.error(f"Erreur vérification authentification: {e}")
            return False

    @property
    def is_authenticated(self) -> bool:
        """
        Vérifie si le client est authentifié.
        
        Returns:
            bool: Statut d'authentification
        """
        return self._is_authenticated

    def get_model_access(self, model_name: str) -> bool:
        """
        Vérifie si le token a accès au modèle spécifié.
        
        Args:
            model_name: Nom du modèle à vérifier
            
        Returns:
            bool: True si l'accès est autorisé
        """
        try:
            if not self.is_authenticated:
                return False

            # Vérifie les droits d'accès
            model_info = self.api.model_info(model_name)
            return not model_info.private or model_info.gated == "none"

        except Exception as e:
            logger.error(f"Erreur vérification accès modèle {model_name}: {e}")
            return False

    async def cleanup(self):
        """Nettoie les ressources d'authentification."""
        self._is_authenticated = False
