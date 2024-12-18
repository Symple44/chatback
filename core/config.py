# core/config.py
from pydantic_settings import BaseSettings
from typing import Dict, Optional, List
import os
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

class Settings(BaseSettings):
    # Paramètres de base
    APP_NAME: str = "Symple Chatbot"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Paramètres hardware
    USE_CPU_ONLY: bool = os.getenv("USE_CPU_ONLY", "true").lower() == "true"
    MAX_THREADS: int = int(os.getenv("MAX_THREADS", "8"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "1"))
    
    # Chemins et fichiers
    DOCUMENTS_DIR: str = "documents"
    SYNC_STATUS_FILE: str = "sync_status.json"
    GOOGLE_DRIVE_CREDENTIALS_PATH: str = "core/storage/credentials.json"
    
    # Configuration Base de données
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/db")
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    
    # Configuration Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_MAX_MEMORY: str = "512mb"
    REDIS_MAX_MEMORY_POLICY: str = "allkeys-lru"
    REDIS_MAX_CONNECTIONS: int = 10
    CACHE_TTL: int = 3600  # 1 heure
    
    # Configuration Elasticsearch
    ELASTICSEARCH_HOST: str = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
    ELASTICSEARCH_USER: str = os.getenv("ELASTICSEARCH_USER", "")
    ELASTICSEARCH_PASSWORD: str = os.getenv("ELASTICSEARCH_PASSWORD", "")
    ELASTICSEARCH_CA_CERT: Optional[str] = os.getenv("ELASTICSEARCH_CA_CERT")
    ELASTICSEARCH_CLIENT_CERT: Optional[str] = os.getenv("ELASTICSEARCH_CLIENT_CERT")
    ELASTICSEARCH_CLIENT_KEY: Optional[str] = os.getenv("ELASTICSEARCH_CLIENT_KEY")
    ELASTICSEARCH_EMBEDDING_DIM: int = 384
    ES_BULK_SIZE: int = 25
    
    # Configuration du modèle
    MODEL_NAME: str = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v0.1")
    USE_QUANTIZATION: bool = False
    USE_INT8: bool = False
    USE_4BIT: bool = False
    MAX_INPUT_LENGTH: int = 4096
    MAX_OUTPUT_LENGTH: int = 2048
    
    # Paramètres de contexte et génération
    CONTEXT_CONFIDENCE_THRESHOLD: float = 0.7
    MAX_RELEVANT_DOCS: int = 3
    DIRECT_RESPONSE_THRESHOLD: float = 0.8
    ACTION_VERBS: List[str] = [
        'cliquer', 'sélectionner', 'choisir', 'saisir',
        'valider', 'remplir', 'ouvrir', 'fermer', 'créer',
        'entrer', 'modifier', 'supprimer', 'imprimer',
        'visualiser', 'rechercher', 'ajouter'
    ]
    PROCEDURE_KEYWORDS: List[str] = [
        'procédure', 'étapes', 'comment', 'suivez', 'méthode',
        'instructions', 'guide', 'marche à suivre', 'processus',
        'démarche', 'mode opératoire'
    ]

    # Messages système
    NO_PROCEDURE_MESSAGE: str = "Je n'ai pas trouvé de procédure spécifique pour cette demande dans la documentation."
    NO_CONTEXT_MESSAGE: str = "Je ne trouve pas d'information pertinente dans la documentation pour répondre à cette question."
    INVALID_RESPONSE_MESSAGE: str = "Je ne peux pas générer une réponse appropriée pour cette demande."
    ERROR_MESSAGE: str = "Désolé, une erreur est survenue lors de la génération de la réponse."
    STREAM_ERROR_MESSAGE: str = "Erreur lors de la génération de la réponse."
    
    # Paramètres de streaming
    STREAM_DELAY: float = 0.05
    STREAM_CHUNK_SIZE: int = 10
    
    # Optimisation mémoire
    LOW_CPU_MEM_USAGE: bool = False
    MIN_RESPONSE_LENGTH: int = 20
    DEVICE_MAP: str = "cpu" if USE_CPU_ONLY else "auto"
    MAX_MEMORY: Dict[str, str] = {
        "cpu": "12GiB"
    }
    
    # Paramètres de génération optimisés
    MAX_NEW_TOKENS: int = 150
    MIN_NEW_TOKENS: int = 30
    DO_SAMPLE: bool = True
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.75
    TOP_K: int = 40
    REPETITION_PENALTY: float = 1.3
    NO_REPEAT_NGRAM_SIZE: int = 3
    USE_CACHE: bool = True
    NUM_BEAMS: int = 3
    EARLY_STOPPING: bool = False

    @property
    def GENERATION_CONFIG(self) -> Dict:
        """Retourne la configuration de génération comme un dictionnaire."""
        return {
            "max_new_tokens": self.MAX_NEW_TOKENS,
            "min_new_tokens": self.MIN_NEW_TOKENS,
            "do_sample": self.DO_SAMPLE,
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "top_k": self.TOP_K,
            "repetition_penalty": self.REPETITION_PENALTY,
            "no_repeat_ngram_size": self.NO_REPEAT_NGRAM_SIZE,
            "use_cache": self.USE_CACHE,
            "num_beams": self.NUM_BEAMS,
            "pad_token_id": self.tokenizer.pad_token_id if hasattr(self, 'tokenizer') else None,
            "eos_token_id": self.tokenizer.eos_token_id if hasattr(self, 'tokenizer') else None
        }
    
    # Configuration embedding
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 8
    
    # Traitement des documents
    CHUNK_SIZE: int = 256
    CHUNK_OVERLAP: int = 20
    MAX_CHUNKS_PER_DOC: int = 50
    
    # Intervalles de nettoyage
    CLEANUP_INTERVAL: int = 60  # 1 minute
    SYNC_INTERVAL_HOURS: int = 24
    
    # Configuration de cache
    USE_DISK_CACHE: bool = True
    CACHE_DIR: str = "model_cache"
    
    # Configuration Google Drive
    GOOGLE_DRIVE_FOLDER_ID: str = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
    
    # Template de chat amélioré avec instructions plus explicites
    CHAT_TEMPLATE: str = """<|system|>Tu es un assistant technique qui explique la documentation CM Manager.
Réponds UNIQUEMENT avec la procédure exacte tirée de la documentation.

Format de réponse attendu :
Pour [action], suivez ces étapes :
1. [Première étape technique]
2. [Deuxième étape technique]
...

Documentation disponible :
{context}

Question : {query}

Procédure technique :</s>
<|assistant|>"""

    # Prompt système plus strict
    SYSTEM_PROMPT: str = """Je suis l'assistant CM Manager. Je fournis uniquement des informations basées sur la documentation officielle."""

    class Config:
        env_file = ".env"
        case_sensitive = True

# Instance unique des paramètres
settings = Settings()