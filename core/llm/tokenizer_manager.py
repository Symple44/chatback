# core/llm/tokenizer_manager.py
from transformers import AutoTokenizer, PreTrainedTokenizer, GenerationConfig
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import re
from dataclasses import dataclass
from enum import Enum

from core.config.config import settings
from core.config.models import AVAILABLE_MODELS, EMBEDDING_MODELS, SUMMARIZER_MODELS
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger("tokenizer_manager")

class TokenizerType(Enum):
    """Types de tokenizers supportés."""
    CHAT = "chat"
    SUMMARIZER = "summarizer"
    EMBEDDING = "embedding"

@dataclass
class TokenizerConfig:
    """Configuration pour un tokenizer."""
    padding_side: str = "left"
    truncation_side: str = "right"
    max_length: int = 2048
    add_special_tokens: bool = True
    use_fast: bool = True
    trust_remote_code: bool = True

class TokenizerManager:
    """Gestionnaire centralisé des tokenizers."""
    
    def __init__(self):
        self.tokenizers: Dict[TokenizerType, PreTrainedTokenizer] = {}
        self.configs: Dict[TokenizerType, TokenizerConfig] = {}
        self._initialized = False
        self.default_type = TokenizerType.CHAT

    async def initialize(self):
        """Initialise les tokenizers."""
        if self._initialized:
            return

        try:
            logger.info("Initialisation des tokenizers...")
            
            # Configuration par type
            configs = {
                TokenizerType.CHAT: TokenizerConfig(
                    padding_side="left",
                    max_length=settings.MAX_CONTEXT_LENGTH
                ),
                TokenizerType.SUMMARIZER: TokenizerConfig(
                    padding_side="right",
                    max_length=settings.MAX_INPUT_LENGTH
                ),
                TokenizerType.EMBEDDING: TokenizerConfig(
                    padding_side="right",
                    max_length=512  # Taille standard pour les embeddings
                )
            }

            # Récupération des chemins des modèles depuis la configuration
            model_paths = {
                TokenizerType.CHAT: AVAILABLE_MODELS[settings.MODEL_NAME]["path"],
                TokenizerType.SUMMARIZER: SUMMARIZER_MODELS[settings.MODEL_NAME_SUMMARIZER]["path"],
                TokenizerType.EMBEDDING: EMBEDDING_MODELS[settings.EMBEDDING_MODEL]["path"]
            }

            logger.info(f"Chemins des modèles: {model_paths}")

            for tokenizer_type, model_path in model_paths.items():
                config = configs[tokenizer_type]
                tokenizer = await self._initialize_tokenizer(
                    model_path,
                    config,
                    tokenizer_type
                )
                self.tokenizers[tokenizer_type] = tokenizer
                self.configs[tokenizer_type] = config

            self._initialized = True
            logger.info("Tokenizers initialisés avec succès")

        except Exception as e:
            logger.error(f"Erreur initialisation tokenizers: {e}")
            raise

    async def _initialize_tokenizer(
        self,
        model_path: str,
        config: TokenizerConfig,
        tokenizer_type: TokenizerType
    ) -> PreTrainedTokenizer:
        """Initialise un tokenizer spécifique."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                revision=settings.MODEL_REVISION,
                padding_side=config.padding_side,
                truncation_side=config.truncation_side,
                use_fast=config.use_fast,
                trust_remote_code=config.trust_remote_code
            )

            # Configuration spécifique pour Mistral
            if "mistral" in model_path.lower() and tokenizer_type == TokenizerType.CHAT:
                tokenizer.pad_token = tokenizer.eos_token
                special_tokens = {
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "pad_token": "</s>",
                    "unk_token": "<unk>",
                }
                tokenizer.add_special_tokens(special_tokens)

            return tokenizer

        except Exception as e:
            logger.error(f"Erreur initialisation tokenizer {model_path}: {e}")
            raise

    def get_tokenizer(self, tokenizer_type: Optional[TokenizerType] = None) -> PreTrainedTokenizer:
        """Récupère un tokenizer spécifique."""
        if not self._initialized:
            raise RuntimeError("TokenizerManager non initialisé")

        t_type = tokenizer_type or self.default_type
        if t_type not in self.tokenizers:
            raise ValueError(f"Type de tokenizer non supporté: {t_type}")
            
        return self.tokenizers[t_type]

    def encode_with_truncation(
        self,
        messages: Union[str, List[Dict]],
        max_length: Optional[int] = None,
        tokenizer_type: Optional[TokenizerType] = None,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Encode et tronque le texte avec gestion de la longueur maximale."""
        try:
            tokenizer = self.get_tokenizer(tokenizer_type)
            config = self.configs[tokenizer_type or self.default_type]
            max_length = max_length or config.max_length

            # Gestion différente selon le type d'entrée
            if isinstance(messages, str):
                encoded = tokenizer(
                    messages,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors=return_tensors
                )
            else:
                # Pour les conversations (chat)
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors=return_tensors
                )

                # Conversion en format attendu si nécessaire
                if isinstance(encoded, torch.Tensor):
                    encoded = {
                        'input_ids': encoded,
                        'attention_mask': torch.ones_like(encoded)
                    }

            # Tronquer si nécessaire
            if encoded['input_ids'].shape[1] > max_length:
                encoded = {
                    'input_ids': encoded['input_ids'][:, :max_length],
                    'attention_mask': encoded['attention_mask'][:, :max_length]
                }

            return encoded

        except Exception as e:
            logger.error(f"Erreur d'encodage: {e}")
            raise

    def decode_and_clean(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        tokenizer_type: Optional[TokenizerType] = None,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """Décode et nettoie le texte généré."""
        try:
            tokenizer = self.get_tokenizer(tokenizer_type)

            # Décodage initial
            full_response = tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )

            # Patterns de nettoyage selon le type de modèle
            if "mistral" in str(tokenizer.__class__).lower():
                assistant_pattern = r'\[/INST\](.*?)(?:\[INST\]|$)'
            else:
                assistant_pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)(<\|eot_id\|>|$)'

            # Extraction de la réponse
            match = re.search(assistant_pattern, full_response, re.DOTALL | re.IGNORECASE)
            response = match.group(1).strip() if match else full_response.strip()

            # Nettoyage supplémentaire
            response = re.sub(r'\s+', ' ', response)  # Normaliser les espaces
            response = response.strip()

            # Vérification de la longueur minimale
            if len(response) < 5:
                logger.warning(f"Réponse générée trop courte: {full_response}")
                return "Je m'excuse, je n'ai pas pu générer une réponse appropriée."

            return response

        except Exception as e:
            logger.error(f"Erreur décodage: {e}")
            return "Une erreur est survenue lors du décodage de la réponse."

    def count_tokens(
        self,
        text: str,
        tokenizer_type: Optional[TokenizerType] = None
    ) -> int:
        """Compte le nombre de tokens dans un texte."""
        try:
            tokenizer = self.get_tokenizer(tokenizer_type)
            return len(tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Erreur comptage tokens: {e}")
            return 0

    def __call__(
        self,
        text: str,
        tokenizer_type: Optional[TokenizerType] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Interface directe pour la tokenization."""
        tokenizer = self.get_tokenizer(tokenizer_type)
        return tokenizer(text, **kwargs)

    async def cleanup(self):
        """Nettoie les ressources."""
        self.tokenizers.clear()
        self.configs.clear()
        self._initialized = False
        logger.info("Tokenizer Manager nettoyé")