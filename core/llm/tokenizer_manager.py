# core/llm/tokenizer_manager.py
from transformers import AutoTokenizer, GenerationConfig
from typing import Dict, Any, Optional
import torch
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("tokenizer_manager")

class TokenizerManager:
    def __init__(self):
        self.tokenizer = None
        self._setup_tokenizer()

    def _setup_tokenizer(self):
        try:
            logger.info("Initialisation du tokenizer...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                revision=settings.MODEL_REVISION,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.generation_config = GenerationConfig(
                do_sample=bool(settings.DO_SAMPLE),
                temperature=float(settings.TEMPERATURE),
                top_p=float(settings.TOP_P),
                top_k=int(settings.TOP_K),
                max_new_tokens=int(settings.MAX_NEW_TOKENS),
                min_new_tokens=int(settings.MIN_NEW_TOKENS),
                repetition_penalty=float(settings.REPETITION_PENALTY),
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            logger.info(f"Tokenizer initialisé: {settings.MODEL_NAME}")

        except Exception as e:
            logger.error(f"Erreur configuration tokenizer: {e}")
            raise

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Encode le texte en tokens."""
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=max_length is not None,
            return_tensors=return_tensors
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
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
