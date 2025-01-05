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
        self._initialized = False
        
    async def initialize(self):
        """Initialise le tokenizer de manière asynchrone."""
        if self._initialized:
            return
        try:
            logger.info("Initialisation du tokenizer...")
            
            # Configuration du tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                revision=settings.MODEL_REVISION,
                trust_remote_code=True,
                padding_side='left',
                use_fast=True  # Utiliser le tokenizer rapide si disponible
            )
            
            # Configuration des tokens spéciaux pour Mistral
            if "mistral" in settings.MODEL_NAME.lower():
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = 'left'  # Important pour le chat
                special_tokens = {
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "pad_token": "</s>",
                    "unk_token": "<unk>",
                }
                self.tokenizer.add_special_tokens(special_tokens)
            
            # Configuration spécifique pour Mistral
            if "mistral" in settings.MODEL_NAME.lower():
                # Mistral utilise déjà les bons tokens, pas besoin d'ajout spécial
                self.bos_token = "<s>"
                self.eos_token = "</s>"
                self.pad_token = self.tokenizer.eos_token
            else:
                # Gestion pour les autres modèles
                if self.tokenizer.pad_token is None:
                    if self.tokenizer.eos_token:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    else:
                        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Configuration de la génération optimisée pour Mistral
            self.generation_config = GenerationConfig(
                do_sample=bool(settings.DO_SAMPLE),
                temperature=float(settings.TEMPERATURE),
                top_p=float(settings.TOP_P),
                top_k=int(settings.TOP_K),
                max_new_tokens=int(settings.MAX_NEW_TOKENS),
                min_new_tokens=int(settings.MIN_NEW_TOKENS),
                repetition_penalty=float(settings.REPETITION_PENALTY),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # Paramètres spécifiques Mistral
                use_cache=True,
                typical_p=0.95,  # Ajout du typical sampling pour Mistral
                encoder_repetition_penalty=1.0
            )
            
            logger.info(f"Tokenizer initialisé: {settings.MODEL_NAME}")
            return self.tokenizer
            
        except Exception as e:
            logger.error(f"Erreur initialisation tokenizer: {e}")
            raise

    def encode_with_truncation(
        self, 
        messages: List[Dict],
        max_length: Optional[int] = None, 
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Encode et tronque le texte avec gestion de la longueur maximale.
        Adapté pour le format Mistral et Llama.
        """
        try:
            is_mistral = "mistral" in settings.MODEL_NAME.lower()
            max_length = max_length or self.max_length
            
            # Validation et réorganisation des messages pour tous les modèles
            validated_messages = []
            has_system = False
            
            # Traitement des messages existants
            for msg in messages:
                if msg["role"] == "system":
                    if not has_system:  # On ne garde que le premier message système
                        validated_messages.append(msg)
                        has_system = True
                elif msg["role"] in ["user", "assistant"]:
                    validated_messages.append(msg)
            
            # Vérification de l'alternance user/assistant
            final_messages = []
            if has_system:
                final_messages.append(validated_messages[0])  # Ajout du message système
                validated_messages = validated_messages[1:]   # Retrait du message système
            
            # On s'assure que ça commence par un user
            if validated_messages and validated_messages[0]["role"] == "assistant":
                validated_messages.pop(0)
            
            # Construction de la séquence alternée
            current_role = "user"
            for msg in validated_messages:
                if msg["role"] == current_role:
                    final_messages.append(msg)
                    current_role = "assistant" if current_role == "user" else "user"
            
            # S'assurer que le dernier message est de l'utilisateur
            if final_messages and final_messages[-1]["role"] == "assistant":
                final_messages.pop()
            
            # Tokenization avec le template approprié
            if is_mistral:
                # Template Mistral
                tokenized = self.tokenizer.apply_chat_template(
                    final_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors=return_tensors
                )
            else:
                # Template pour Llama et autres modèles
                tokenized = self.tokenizer(
                    self.tokenizer.apply_chat_template(
                        final_messages,
                        add_generation_prompt=True
                    ),
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors=return_tensors
                )
            
            # Extraction des input_ids et création du masque d'attention
            input_ids = tokenized.get('input_ids', tokenized)
            attention_mask = torch.ones_like(input_ids) if isinstance(tokenized, torch.Tensor) else tokenized.get('attention_mask')
            
            # Tronquer si nécessaire
            if input_ids.shape[1] > max_length:
                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

        except Exception as e:
            logger.error(f"Erreur de tokenisation: {e}")
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

            if "mistral" in settings.MODEL_NAME.lower():
                # Pattern spécifique pour Mistral
                assistant_pattern = r'\[/INST\](.*?)(?:\[INST\]|$)'
            else:
                # Pattern par défaut
                assistant_pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)(<\|eot_id\|>|$)'
            
            match = re.search(assistant_pattern, full_response, re.DOTALL | re.IGNORECASE)
            
            if match:
                response = match.group(1).strip()
            else:
                response = full_response.strip()

            # Nettoyage supplémentaire
            response = re.sub(r'\s+', ' ', response)  # Normaliser les espaces
            response = response.strip()

            # Nettoyage spécifique Mistral
            if "mistral" in settings.MODEL_NAME.lower():
                response = response.replace("[/INST]", "").replace("[INST]", "").strip()

            # Vérification de la longueur minimale
            if len(response) < 5:
                logger.warning(f"Réponse générée trop courte. Contenu brut : {full_response}")
                response = "Je m'excuse, je n'ai pas pu générer une réponse appropriée."

            return response

        except Exception as e:
            logger.error(f"Erreur de décodage: {e}")
            return {
                "response": "Une erreur est survenue lors de la génération de la réponse.",
                "confidence_score": 0.0,
                "processing_time": 0.0,
                "tokens_used": 0,
                "metadata": {
                    "error": True,
                    "error_message": str(e)
                }
            }

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
            max_length=max_length or self.max_length,
            padding=padding,
            truncation=True,
            return_tensors=return_tensors
        )

    async def cleanup(self):
        """Nettoie les ressources."""
        if hasattr(self, 'tokenizer'):
            del self.tokenizer