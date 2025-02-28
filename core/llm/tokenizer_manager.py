# core/llm/tokenizer_manager.py
from transformers import AutoTokenizer, PreTrainedTokenizer, GenerationConfig
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import re
from dataclasses import dataclass
from enum import Enum

from core.config.config import settings
from core.config.models import (
    CHAT_MODELS,
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
            return CHAT_MODELS.get(model_name)
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
                "revision": settings.models.MODEL_REVISION,
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
                # Configuration spécifique pour MT5 - utilisez le slow tokenizer et ignorez les warnings
                tokenizer_kwargs.update({
                    "model_max_length": config.max_length,
                    "use_fast": False  # Utilise le slow tokenizer pour éviter les warnings
                })
                
                # Supprimer temporairement les avertissements pour MT5
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, 
                                        message="The sentencepiece tokenizer.*")
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        **tokenizer_kwargs
                    )
                    return tokenizer
            
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

                # Définir explicitement le pad_token_id et eos_token_id
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

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
                    max_length=settings.models.MAX_CONTEXT_LENGTH
                ),
                TokenizerType.SUMMARIZER: TokenizerConfig(
                    padding_side="right",
                    max_length=settings.models.MAX_INPUT_LENGTH
                ),
                TokenizerType.EMBEDDING: TokenizerConfig(
                    padding_side="right",
                    max_length=512
                )
            }

            # Chargement des tokenizers pour les modèles de chat
            for model_name in CHAT_MODELS.keys():
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
                settings.models.EMBEDDING_MODEL,
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
    ) -> Optional[PreTrainedTokenizer]:
        """Charge ou recharge un tokenizer."""
        try:
            tokenizer_key = f"{tokenizer_type.value}_{model_name}"
            
            # Si le tokenizer existe déjà et qu'on ne force pas le rechargement
            if not force_reload and tokenizer_key in self.tokenizers:
                return self.tokenizers[tokenizer_key]

            # Cas spécial pour les modèles SentenceTransformers - ils gèrent leur propre tokenization
            if tokenizer_type == TokenizerType.EMBEDDING and "sentence-transformers" in model_name:
                logger.info(f"Modèle SentenceTransformers détecté: {model_name}, pas de tokenizer HF à charger")
                self.tokenizers[tokenizer_key] = None
                self.configs[tokenizer_key] = config
                self.current_models[tokenizer_type] = model_name
                return None

            # Récupération de la configuration du modèle
            model_configs = {
                TokenizerType.CHAT: CHAT_MODELS,
                TokenizerType.SUMMARIZER: SUMMARIZER_MODELS,
                TokenizerType.EMBEDDING: EMBEDDING_MODELS
            }
            
            model_config = model_configs[tokenizer_type].get(model_name)
            if not model_config:
                raise ValueError(f"Configuration non trouvée pour {model_name}")

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
            config = self.configs[f"{tokenizer_type.value}_{self.current_models[tokenizer_type]}"]
            max_length = max_length or config.max_length

            # Pour le texte brut
            text_to_encode = messages[0]["content"] if isinstance(messages, list) else messages
            
            if return_text:
                return text_to_encode

            # Sinon, on fait uniquement la tokenisation sans appliquer de template
            return tokenizer(
                text_to_encode,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors=return_tensors
            )

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
        """
        Décode et nettoie la réponse du modèle.
        
        Args:
            token_ids: Tensor de tokens à décoder
            skip_special_tokens: Si True, ignore les tokens spéciaux
            model_name: Nom du modèle (optionnel)
            tokenizer_type: Type de tokenizer
            clean_up_tokenization_spaces: Si True, nettoie les espaces
            
        Returns:
            Texte nettoyé
        """
        try:
            tokenizer = self.get_tokenizer(model_name, tokenizer_type)
            if not tokenizer:
                raise ValueError(f"Tokenizer non trouvé pour {model_name}")

            # 1. Décodage initial sans nettoyage
            response = tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )

            # 2. Extraction de la réponse après la dernière balise [/INST]
            parts = response.split("[/INST]")
            if len(parts) > 1:
                # Prendre uniquement la dernière partie après [/INST]
                response = parts[-1]
            
            # 3. Nettoyage des balises restantes
            response = re.sub(r'<s>|</s>', '', response)
            response = re.sub(r'\[SYSTEM_PROMPT\].*?\[/SYSTEM_PROMPT\]', '', response, flags=re.DOTALL)
            response = re.sub(r'\[RESPONSE_TYPE\].*?\[/RESPONSE_TYPE\]', '', response, flags=re.DOTALL)
            response = re.sub(r'\[CONTEXT\].*?\[/CONTEXT\]', '', response, flags=re.DOTALL)
            
            # 4. Nettoyage des espaces et formatage
            response = re.sub(r'\s+', ' ', response)  # Remplace les espaces multiples par un seul
            response = response.strip()  # Supprime les espaces au début et à la fin
            
            # 5. Vérification finale
            if not response:
                return "Désolé, je n'ai pas pu générer une réponse valide."
                
            return response

        except Exception as e:
            logger.error(f"Erreur décodage pour {model_name}: {e}")
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