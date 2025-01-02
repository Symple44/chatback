from transformers import AutoTokenizer, GenerationConfig
from typing import Dict, Any, Optional
import torch
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("tokenizer_manager")

class TokenizerManager:
    def __init__(self):
        self.tokenizer = None
        self.max_length = settings.MAX_INPUT_LENGTH 
        self._setup_tokenizer()

    def _setup_tokenizer(self):
        try:
            logger.info("Initialisation du tokenizer...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                revision=settings.MODEL_REVISION,
                trust_remote_code=True,
                padding_side='left'  # Optionnel, mais peut aider
            )
            
            # Gestion intelligente du pad_token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
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

    def __call__(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Tokenize le texte."""
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=True,
            return_tensors=return_tensors
        )
        
    def encode_with_truncation(
        self, 
        text: str, 
        max_length: Optional[int] = None, 
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Encode et tronque le texte avec gestion de la longueur maximale.
        """
        if max_length is None:
            max_length = self.max_length

        # Encode avec troncature
        encoded = self.tokenizer(
            text, 
            truncation=True, 
            max_length=max_length, 
            padding=True, 
            return_tensors=return_tensors
        )
        return encoded


    def decode_and_clean(
        self, 
        token_ids: torch.Tensor, 
        skip_special_tokens: bool = True,
        skip_assistant_token: bool = True
    ) -> str:
        """
        Décode et nettoie la sortie du modèle.
        
        Args:
            token_ids: Tokens à décoder
            skip_special_tokens: Ignorer les tokens spéciaux
            skip_assistant_token: Ignorer le token assistant et ne garder que la réponse
        """
        response = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=True
        )

        # Nettoyage des balises de rôle si demandé
        if skip_assistant_token:
            # Supprime tout ce qui précède la balise assistant
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1]
            # Supprime la balise de fin si présente
            if "</|assistant|>" in response:
                response = response.split("</|assistant|>")[0]
            
        return response.strip()

    async def cleanup(self):
        """Nettoie les ressources."""
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
