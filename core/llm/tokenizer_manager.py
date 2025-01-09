# core/llm/tokenizer_manager.py
from transformers import AutoTokenizer, PreTrainedTokenizer, GenerationConfig
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import re
from dataclasses import dataclass
from enum import Enum

from core.config.config import settings
from core.config.models import (
    AVAILABLE_MODELS,
    EMBEDDING_MODELS,
    SUMMARIZER_MODELS,
    MODEL_PERFORMANCE_CONFIGS
)
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
        self.tokenizers: Dict[str, PreTrainedTokenizer] = {}  # Changé pour utiliser le nom du modèle comme clé
        self.configs: Dict[str, TokenizerConfig] = {}
        self._initialized = False
        self.current_models = {
            TokenizerType.CHAT: None,
            TokenizerType.SUMMARIZER: None,
            TokenizerType.EMBEDDING: None
        }

    def _get_model_config(self, model_name: str, model_type: TokenizerType) -> Optional[Dict]:
        """Récupère la configuration d'un modèle selon son type."""
        if model_type == TokenizerType.CHAT:
            return AVAILABLE_MODELS.get(model_name)
        elif model_type == TokenizerType.EMBEDDING:
            return EMBEDDING_MODELS.get(model_name)
        elif model_type == TokenizerType.SUMMARIZER:
            return SUMMARIZER_MODELS.get(model_name)
        return None

    async def _initialize_tokenizer(
        self,
        model_path: str,
        config: TokenizerConfig,
        tokenizer_type: TokenizerType
    ) -> PreTrainedTokenizer:
        """Initialise un tokenizer spécifique."""
        try:
            logger.info(f"Initialisation du tokenizer pour {model_path}")
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
                    max_length=512
                )
            }

            # Chargement des tokenizers pour les modèles de chat
            for model_name in AVAILABLE_MODELS.keys():
                await self.load_tokenizer(
                    model_name, 
                    TokenizerType.CHAT, 
                    configs[TokenizerType.CHAT]
                )
                
            # Chargement des tokenizers pour les summarizers
            await self.load_tokenizer(
                settings.MODEL_NAME_SUMMARIZER,
                TokenizerType.SUMMARIZER,
                configs[TokenizerType.SUMMARIZER]
            )
                
            # Chargement des tokenizers pour l'embedding
            await self.load_tokenizer(
                settings.EMBEDDING_MODEL,
                TokenizerType.EMBEDDING,
                configs[TokenizerType.EMBEDDING]
            )

            self._initialized = True
            logger.info("Tokenizers initialisés avec succès")

        except Exception as e:
            logger.error(f"Erreur initialisation tokenizers: {e}")
            raise

    async def load_tokenizer(
        self,
        model_name: str,
        tokenizer_type: TokenizerType,
        config: TokenizerConfig,
        force_reload: bool = False
    ) -> PreTrainedTokenizer:
        """Charge ou recharge un tokenizer."""
        try:
            model_config = self._get_model_config(model_name, tokenizer_type)
            if not model_config:
                raise ValueError(f"Configuration non trouvée pour {model_name} , {tokenizer_type} ")

            tokenizer_key = f"{tokenizer_type.value}_{model_name}"
            
            # Si le tokenizer existe déjà et qu'on ne force pas le rechargement
            if not force_reload and tokenizer_key in self.tokenizers:
                return self.tokenizers[tokenizer_key]

            # Initialisation du nouveau tokenizer
            tokenizer = await self._initialize_tokenizer(
                model_config["path"],
                config,
                tokenizer_type
            )
            
            # Configuration spécifique pour Mixtral
            if "mixtral" in model_name.lower():
                tokenizer.pad_token = tokenizer.eos_token
                special_tokens = {
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "pad_token": "</s>",
                    "unk_token": "<unk>",
                }
                tokenizer.add_special_tokens(special_tokens)

            # Mise à jour des dictionnaires
            self.tokenizers[tokenizer_key] = tokenizer
            self.configs[tokenizer_key] = config
            self.current_models[tokenizer_type] = model_name

            logger.info(f"Tokenizer chargé pour {model_name} ({tokenizer_type.value})")
            return tokenizer

        except Exception as e:
            logger.error(f"Erreur chargement tokenizer {model_name}: {e}")
            raise

    def get_tokenizer(
        self,
        model_name: Optional[str] = None,
        tokenizer_type: Optional[TokenizerType] = TokenizerType.CHAT
    ) -> PreTrainedTokenizer:
        """Récupère un tokenizer spécifique."""
        if not self._initialized:
            raise RuntimeError("TokenizerManager non initialisé")

        if model_name is None:
            model_name = self.current_models[tokenizer_type]
            if model_name is None:
                raise ValueError(f"Aucun modèle actif pour le type {tokenizer_type}")

        tokenizer_key = f"{tokenizer_type.value}_{model_name}"
        if tokenizer_key not in self.tokenizers:
            raise ValueError(f"Tokenizer non trouvé pour {model_name}")

        return self.tokenizers[tokenizer_key]

    async def change_model(
        self,
        new_model_name: str,
        tokenizer_type: TokenizerType,
        force_reload: bool = False
    ) -> PreTrainedTokenizer:
        """Change le modèle actif pour un type donné."""
        try:
            # Vérification si le changement est nécessaire
            if new_model_name == self.current_models[tokenizer_type] and not force_reload:
                return self.get_tokenizer(new_model_name, tokenizer_type)

            # Chargement du nouveau tokenizer
            config = self.configs.get(f"{tokenizer_type.value}_{self.current_models[tokenizer_type]}")
            if not config:
                config = TokenizerConfig()  # Config par défaut

            tokenizer = await self.load_tokenizer(
                new_model_name,
                tokenizer_type,
                config,
                force_reload=True
            )

            return tokenizer

        except Exception as e:
            logger.error(f"Erreur changement de modèle: {e}")
            raise

    def encode_with_truncation(
        self,
        messages: Union[str, List[Dict]],
        max_length: Optional[int] = None,
        model_name: Optional[str] = None,
        tokenizer_type: Optional[TokenizerType] = TokenizerType.CHAT,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Encode et tronque le texte avec gestion de la longueur maximale."""
        try:
            tokenizer = self.get_tokenizer(model_name, tokenizer_type)
            config = self.configs[f"{tokenizer_type.value}_{self.current_models[tokenizer_type]}"]
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
        model_name: Optional[str] = None,
        tokenizer_type: Optional[TokenizerType] = TokenizerType.CHAT,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """Décode et nettoie le texte généré."""
        try:
            tokenizer = self.get_tokenizer(model_name, tokenizer_type)
            full_response = tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )

            # Pour les modèles Mistral
            if "mistral" in str(tokenizer.__class__).lower():
                # Enlever tout ce qui est avant le premier [/INST]
                parts = full_response.split("[/INST]")
                if len(parts) > 1:
                    response = parts[-1].strip()
                else:
                    response = full_response.strip()
            else:
                # Pattern pour autres modèles
                pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)(<\|eot_id\|>|$)'
                match = re.search(pattern, full_response, re.DOTALL | re.IGNORECASE)
                response = match.group(1).strip() if match else full_response.strip()

            # Nettoyage supplémentaire
            response = re.sub(r'\s+', ' ', response)
            response = response.strip()
            # Enlever les [INST] résiduels s'il y en a
            response = re.sub(r'\[INST\].*?$', '', response)

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
        model_name: Optional[str] = None,
        tokenizer_type: Optional[TokenizerType] = TokenizerType.CHAT
    ) -> int:
        """Compte le nombre de tokens dans un texte."""
        try:
            tokenizer = self.get_tokenizer(model_name, tokenizer_type)
            return len(tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Erreur comptage tokens: {e}")
            return 0

    async def cleanup(self):
        """Nettoie les ressources."""
        self.tokenizers.clear()
        self.configs.clear()
        self.current_models = {t: None for t in TokenizerType}
        self._initialized = False
        logger.info("Tokenizer Manager nettoyé")