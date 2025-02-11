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
            
            # Configuration de base
            tokenizer_kwargs = {
                "revision": settings.MODEL_REVISION,
                "padding_side": config.padding_side,
                "truncation_side": config.truncation_side,
                "use_fast": config.use_fast,
                "trust_remote_code": config.trust_remote_code
            }
            
            # Configuration spécifique selon le type de modèle
            if "t5" in model_path.lower():
                tokenizer_kwargs.update({
                    "legacy": False,
                    "model_max_length": config.max_length,
                    "use_fast": True  # Force l'utilisation du fast tokenizer
                })
            elif "mt5" in model_path.lower():
                # Configuration spécifique pour MT5
                tokenizer_kwargs.update({
                    "model_max_length": config.max_length,
                    "use_fast": False  # Utilise le slow tokenizer pour éviter les warnings
                })
                
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                **tokenizer_kwargs
            )

            # Configuration spécifique post-chargement
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
            for model_name in SUMMARIZER_MODELS.keys():
                await self.load_tokenizer(
                    model_name,
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
            # Récupération de la configuration du modèle
            model_configs = {
                TokenizerType.CHAT: AVAILABLE_MODELS,
                TokenizerType.SUMMARIZER: SUMMARIZER_MODELS,
                TokenizerType.EMBEDDING: EMBEDDING_MODELS
            }
            
            model_config = model_configs[tokenizer_type].get(model_name)
            if not model_config:
                raise ValueError(f"Configuration non trouvée pour {model_name}")

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
            
            # Stockage du tokenizer
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
        return_tensors: str = "pt",
        return_text: bool = False
    ) -> Union[Dict[str, torch.Tensor], str]:
        try:
            tokenizer = self.get_tokenizer(model_name, tokenizer_type)
            if not tokenizer:
                raise ValueError(f"Tokenizer non trouvé pour {model_name}")

            config = self.configs[f"{tokenizer_type.value}_{self.current_models[tokenizer_type]}"]
            max_length = max_length or config.max_length

            # Nouveau code pour template chat Mistral
            if isinstance(messages, list):
                # Filtrage et préparation des messages
                chat_messages = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    
                    if role == "system":
                        # Les messages système sont ajoutés comme contexte
                        chat_messages.append({"role": "system", "content": content})
                    elif role == "user":
                        chat_messages.append({"role": "user", "content": content})
                    elif role == "assistant":
                        chat_messages.append({"role": "assistant", "content": content})

                # Utilisation du template de chat spécifique à Mistral
                chat_text = tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                chat_text = messages

            # Retourner le texte formaté si demandé
            if return_text:
                return chat_text

            # Sinon, retourner les tokens encodés
            encoded = tokenizer(
                chat_text,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors=return_tensors
            )

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
            if not tokenizer:
                raise ValueError(f"Tokenizer non trouvé pour {model_name}")

            # Décodage initial
            response = tokenizer.decode(
                token_ids,
                skip_special_tokens=False,  # On garde les tokens spéciaux pour trouver les marqueurs
                clean_up_tokenization_spaces=False
            )

            # Extraction de la réponse selon le format Mistral
            if "[/INST]" in response and "</s>" in response:
                # Trouver le début de la réponse effective (après [/INST])
                response_start = response.rfind("[/INST]") + len("[/INST]")
                # Trouver la fin de la réponse (avant </s>)
                response_end = response.find("</s>", response_start)

                if response_end == -1:  # Si pas de </s>, prendre jusqu'à la fin
                    response = response[response_start:]
                else:
                    response = response[response_start:response_end]
            
            response = re.sub(r'\[RESPONSE_TYPE\].*?\[/RESPONSE_TYPE\]\s*', '', response, flags=re.DOTALL)
            response = re.sub(r'\[SYSTEM_PROMPT\].*?\[/SYSTEM_PROMPT\]\s*', '', response, flags=re.DOTALL)
            response = re.sub(r'\[CONTEXT\].*?\[/CONTEXT\]\s*', '', response, flags=re.DOTALL)

            # Nettoyage final des espaces
            response = response.strip()

            return response

        except Exception as e:
            logger.error(f"Erreur décodage pour {model_name}: {e}")
            logger.debug(f"Réponse avant nettoyage: {response}")
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