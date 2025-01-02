from transformers import AutoTokenizer, GenerationConfig
from typing import Dict, Any, Optional, List
import torch
import re
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
        messages: List[Dict],
        max_length: Optional[int] = None, 
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Encode et tronque le texte avec gestion de la longueur maximale.
        """
        try:
            # Utiliser apply_chat_template pour générer le texte
            tokenized_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=return_tensors
            )
            
            # Créer un masque d'attention
            attention_mask = tokenized_text.ne(self.tokenizer.pad_token_id)
            
            # Tronquer si nécessaire
            if max_length and tokenized_text.shape[1] > max_length:
                tokenized_text = tokenized_text[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
            
            return {
                "input_ids": tokenized_text,
                "attention_mask": attention_mask
            }
        except Exception as e:
            logger.error(f"Erreur de tokenisation Llama-3.1: {e}")
            raise


    def decode_and_clean(
        self, 
        token_ids: torch.Tensor, 
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        try:
            # Décodage initial 
            full_response = self.tokenizer.decode(
                token_ids, 
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )

            # Vérification de la longueur minimale
            if len(response) < 5:
                logger.warning(f"Réponse générée trop courte. Contenu brut : {full_response}")
                response = "Je m'excuse, je n'ai pas pu générer une réponse appropriée."

            return response

        except Exception as e:
            logger.error(f"Erreur de décodage: {e}")
            logger.error(f"Tokens décodés bruts : {full_response}")
            return "Une erreur est survenue lors de la génération de la réponse."

    async def cleanup(self):
        """Nettoie les ressources."""
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
