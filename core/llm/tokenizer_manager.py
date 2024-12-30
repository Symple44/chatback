# core/llm/tokenizer_manager.py
from transformers import AutoTokenizer, GenerationConfig
from typing import Dict, Any
import torch
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("tokenizer_manager")

class TokenizerManager:
    def __init__(self):
        """Initialise le gestionnaire de tokenizer."""
        self._setup_tokenizer()

    def _setup_tokenizer(self):
        """Configure le tokenizer."""
        try:
            logger.info("Initialisation du tokenizer...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                revision=settings.MODEL_REVISION,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.generation_config = GenerationConfig(**settings.generation_config)
            logger.info(f"Tokenizer initialisé: {settings.MODEL_NAME}")

        except Exception as e:
            logger.error(f"Erreur configuration tokenizer: {e}")
            raise

    def encode(self, text: str, max_length: int = None) -> Dict[str, torch.Tensor]:
        """Encode le texte en tokens."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

    def decode_and_clean(self, token_ids: torch.Tensor) -> str:
        """Décode et nettoie la sortie du modèle."""
        response = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        if "Réponse:" in response:
            response = response.split("Réponse:")[-1].strip()

        return response

    async def cleanup(self):
        """Nettoie les ressources du tokenizer."""
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
