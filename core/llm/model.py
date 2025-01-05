# core/llm/model.py
from typing import Dict, List, Optional, Union, Generator
from datetime import datetime
from contextlib import nullcontext
import torch
import asyncio
import re
import threading
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, GenerationConfig
from sentence_transformers import SentenceTransformer
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .cuda_manager import CUDAManager
from core.utils.memory_manager import MemoryManager
from .model_manager import ModelManager
from .prompt_system import PromptSystem
from .summarizer import DocumentSummarizer
from .tokenizer_manager import TokenizerManager
from .auth_manager import HuggingFaceAuthManager
from core.chat.context_analyzer import context_analyzer

logger = get_logger("model")

class ModelInference:
    def __init__(self):
        """Initialise les composants de base du modèle d'inférence."""
        self.auth_manager = HuggingFaceAuthManager()
        self.cuda_manager = CUDAManager()
        self.memory_manager = MemoryManager()
        self.tokenizer_manager = TokenizerManager()
        self.model_manager = None 
        self.model = None
        self.tokenizer = None
        self.embedding_model = None
        self._initialized = False
        self.prompt_system = PromptSystem()
        self.summarizer = DocumentSummarizer()
        
        # Paramètre de chat
        self.max_context_docs = settings.MAX_CONTEXT_DOCS
    async def initialize(self):
        """Initialise le modèle de manière asynchrone."""
        if not self._initialized:
            try:
                logger.info("Démarrage de l'initialisation du modèle...")

                # 1. Authentification Hugging Face
                if not await self.auth_manager.setup_auth():
                    raise ValueError("Échec de l'authentification Hugging Face")

                # 2. Vérification de l'accès au modèle
                if not self.auth_manager.get_model_access(settings.MODEL_NAME):
                    raise ValueError(f"Accès non autorisé au modèle {settings.MODEL_NAME}")

                # 3. Initialisation CUDA
                await self.cuda_manager.initialize()
                logger.info("CUDA initialisé")

                # 4. Initialisation du TokenizerManager et ModelManager après CUDA
                await self.tokenizer_manager.initialize()
                self.model_manager = ModelManager(self.cuda_manager, self.tokenizer_manager)
                await self.model_manager.initialize()
                
                # 5. Initialisation du modèle d'embedding
                await self._initialize_embedding_model()
                logger.info("Modèle d'embedding initialisé")

                # 6. Chargement du modèle principal
                self.model, self.tokenizer = await self.model_manager.load_model(settings.MODEL_NAME)
                self.tokenizer_manager.tokenizer = self.tokenizer
                logger.info(f"Modèle principal {settings.MODEL_NAME} chargé")

                # 7. Initialisation du summarizer
                await self.summarizer.initialize()
                logger.info("Summarizer initialisé")
                
                self._initialized = True
                logger.info("Initialisation complète du modèle terminée avec succès")

            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du modèle: {e}")
                await self.cleanup()
                raise
                
    async def _initialize_embedding_model(self):
        """Initialise le modèle d'embeddings."""
        try:
            logger.info(f"Initialisation du modèle d'embeddings {settings.EMBEDDING_MODEL}")
            
            # Utilisation du device de CUDA manager
            device = self.cuda_manager.current_device
            
            # Chargement du modèle d'embeddings
            self.embedding_model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=str(device),  # Conversion en string du torch.device
                cache_folder=str(settings.CACHE_DIR)
            )
            
            # Configuration de la quantization si nécessaire
            if settings.USE_8BIT and str(device) != "cpu":
                self.embedding_model.half()  # Conversion en FP16
            
            logger.info(f"Modèle d'embeddings chargé sur {device}")
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle d'embeddings: {e}")
            raise
    
    async def change_model(self, model_name: str):
        """Change le modèle en cours d'utilisation."""
        try:
            if not self._initialized:
                raise RuntimeError("Le système n'est pas initialisé")

            # Vérification de l'accès
            if not self.auth_manager.get_model_access(model_name):
                raise ValueError(f"Accès non autorisé au modèle {model_name}")
                
            # Chargement du nouveau modèle
            self.model, self.tokenizer = await self.model_manager.load_model(model_name)
            self.tokenizer_manager.tokenizer = self.tokenizer
            
            logger.info(f"Modèle changé avec succès pour: {model_name}")
            
        except Exception as e:
            logger.error(f"Erreur changement de modèle: {e}")
            raise

    async def create_embedding(self, text: str) -> List[float]:
        """Crée un embedding pour un texte donné."""
        if not self._initialized:
            raise RuntimeError("Le modèle n'est pas initialisé")
            
        try:
            # Normalisation du texte
            if not isinstance(text, str):
                text = str(text)
            
            # Génération de l'embedding
            with torch.no_grad():
                embedding = self.embedding_model.encode(
                    text,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )
                
            # Conversion en liste
            return embedding.cpu().tolist()
            
        except Exception as e:
            logger.error(f"Erreur création embedding: {e}")
            metrics.increment_counter("embedding_errors")
            raise
    
    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            # Nettoyage dans l'ordre inverse de l'initialisation
            if hasattr(self, 'model_loader') and self.model_loader:
                await self.model_loader.cleanup()
            
            if hasattr(self, 'tokenizer_manager'):
                await self.tokenizer_manager.cleanup()
            
            if hasattr(self, 'embedding_model'):
                del self.embedding_model
            
            if hasattr(self, 'summarizer'):
                await self.summarizer.cleanup()
            
            await self.cuda_manager.cleanup()
            
            self._initialized = False
            logger.info("Toutes les ressources ont été nettoyées")
        except Exception as e:
            logger.error(f"Erreur nettoyage ressources: {e}")
            
    async def generate_response(
        self,
        messages: List[Dict],
        language: str = "fr",
        response_type: str = "comprehensive",
        **kwargs
    ) -> Dict:
        """
        Génère une réponse basée sur les messages formatés.
        """
        try:
            start_time = datetime.utcnow()
            metrics.increment_counter("generation_requests")

            # Configuration de la génération selon le type
            response_config = settings.RESPONSE_TYPES.get(response_type, settings.RESPONSE_TYPES["comprehensive"])
            generation_config = self._get_generation_config(response_type)

            # Tokenisation des messages
            inputs = self.tokenizer_manager.encode_with_truncation(
                messages,
                max_length=settings.MAX_INPUT_LENGTH,
                return_tensors="pt"
            )
            
            # Transfert sur le device approprié
            inputs = {
                k: v.to(self.model.device) 
                for k, v in inputs.items()
            }
            
            # Génération avec fallback
            outputs = await self._generate_with_fallback(inputs, generation_config)
            response_text = self.tokenizer_manager.decode_and_clean(outputs[0])

            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "response": response_text,
                "tokens_used": self._count_tokens(inputs, outputs),
                "processing_time": processing_time,
                "metadata": {
                    "language": language,
                    "model": settings.MODEL_NAME,
                    "response_type": response_type,
                    "generation_config": generation_config.to_dict(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Erreur génération: {e}")
            metrics.increment_counter("generation_errors")
            return {
                "response": "Une erreur est survenue lors de la génération de la réponse.",
                "tokens_used": {"total": 0},
                "processing_time": 0.0,
                "error": str(e)
            }
            
    async def generate_streaming_response(
        self,
        query: str,
        context_docs: Optional[List[Dict]] = None,
        language: str = "fr",
        response_type: str = "comprehensive"
    ) -> Generator[str, None, None]:
        """
        Génère une réponse en streaming.
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Configuration du streaming
            streamer = TextIteratorStreamer(
                self.tokenizer_manager.tokenizer,
                skip_prompt=True,
                timeout=settings.STREAM_TIMEOUT
            )

            # Préparation du contexte
            validated_docs = await self._validate_and_prepare_context(context_docs)
            context_analysis = await self._analyze_context_relevance(validated_docs, query)

            # Construction du prompt
            prompt_messages = await self.prompt_system.build_chat_prompt(
                query=query,
                context_docs=validated_docs,
                context_info=context_analysis,
                response_type=response_type
            )

            # Configuration de la génération
            generation_config = self._get_generation_config(response_type)
            generation_config_dict = {
                "max_new_tokens": int(generation_config.max_new_tokens),
                "temperature": float(generation_config.temperature),
                "top_p": float(generation_config.top_p),
                "do_sample": bool(generation_config.do_sample),
                "pad_token_id": self.tokenizer_manager.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer_manager.tokenizer.eos_token_id,
                "streamer": streamer
            }

            # Tokenisation
            inputs = self.tokenizer_manager.encode_with_truncation(
                messages=prompt_messages,
                max_length=settings.MAX_INPUT_LENGTH
            )
            
            # Transfert explicite vers le GPU
            inputs = {
                "input_ids": inputs["input_ids"].to(self.model.device),
                "attention_mask": inputs["attention_mask"].to(self.model.device)
            }

            # Démarrage de la génération en arrière-plan
            generation_thread = threading.Thread(
                target=self._generate_in_thread,
                args=(inputs, generation_config_dict)
            )
            generation_thread.start()

            # Stream des tokens
            for token in streamer:
                if token.strip():
                    yield token

            # Attente de la fin de la génération
            generation_thread.join()

        except Exception as e:
            logger.error(f"Erreur génération streaming: {e}")
            yield f"\nDésolé, une erreur est survenue: {str(e)}"

    def _generate_in_thread(
        self, 
        inputs: Dict[str, torch.Tensor], 
        generation_config_dict: Dict
    ) -> None:
        """
        Génère le texte dans un thread séparé.
        """
        try:
            with torch.no_grad():
                self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_config_dict
                )
        except Exception as e:
            logger.error(f"Erreur génération thread: {e}")
            raise

    def _count_tokens(self, inputs: Dict[str, torch.Tensor], outputs: torch.Tensor) -> Dict[str, int]:
        """Compte les tokens utilisés."""
        input_tokens = inputs["input_ids"].shape[1]
        output_tokens = outputs.shape[1]
        new_tokens = max(0, output_tokens - input_tokens)
        
        return {
            "input": input_tokens,
            "output": new_tokens,
            "total": input_tokens + new_tokens
        }
            
    def _get_generation_config(self, response_type: str) -> GenerationConfig:
        """Configuration de génération adaptée au type de réponse."""
        # Récupération des paramètres de base depuis les settings
        base_config = settings.RESPONSE_TYPES.get(response_type, settings.RESPONSE_TYPES["comprehensive"])
        
        return GenerationConfig(
            max_new_tokens=int(base_config.get("max_tokens", settings.MAX_NEW_TOKENS)),
            min_new_tokens=int(settings.MIN_NEW_TOKENS),
            temperature=float(base_config.get("temperature", settings.TEMPERATURE)),
            top_p=float(settings.TOP_P),
            top_k=int(settings.TOP_K),
            do_sample=bool(settings.DO_SAMPLE),
            num_beams=int(settings.NUM_BEAMS),
            repetition_penalty=float(settings.REPETITION_PENALTY),
            length_penalty=float(settings.LENGTH_PENALTY),
            early_stopping=True,
            pad_token_id=self.tokenizer_manager.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer_manager.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            diversity_penalty=0.0,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            bad_words_ids=None,
            remove_invalid_values=True
        )

    async def _validate_and_prepare_context(
        self,
        context_docs: Optional[List[Dict]]
    ) -> List[Dict]:
        """
        Valide et prépare les documents de contexte.
        """
        if not context_docs:
            return []

        validated_docs = []
        total_tokens = 0

        for doc in context_docs:
            try:
                # Extraction et nettoyage du contenu
                content = doc.get("content", "").strip()
                if not content:
                    continue

                # Estimation des tokens
                token_count = len(self.tokenizer_manager(content)["input_ids"])
                if total_tokens + token_count > settings.MAX_INPUT_LENGTH:
                    break

                validated_docs.append({
                    **doc,
                    "token_count": token_count,
                    "processed_content": content
                })
                total_tokens += token_count

            except Exception as e:
                logger.warning(f"Erreur traitement document: {e}")
                continue

        return validated_docs[:self.max_context_docs]

    async def _analyze_context_relevance(
        self,
        docs: List[Dict],
        query: str
    ) -> Dict:
        """Analyse la pertinence du contexte par rapport à la requête."""
        if not docs:
            return {
                "has_context": False,
                "confidence": 0.0,
                "relevance_scores": [],
                "total_tokens": 0
            }

        # Calcul des scores de similarité
        query_embedding = await self.create_embedding(query)
        relevance_scores = []

        for doc in docs:
            try:
                doc_content = doc.get("processed_content", "")
                doc_embedding = await self.create_embedding(doc_content)
                score = float(torch.cosine_similarity(
                    torch.tensor(query_embedding),
                    torch.tensor(doc_embedding),
                    dim=0
                ))
                relevance_scores.append(score)
            except Exception as e:
                logger.warning(f"Erreur calcul similarité: {e}")
                relevance_scores.append(0.0)

        # Calcul de la confiance globale
        avg_confidence = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        total_tokens = sum(doc.get("token_count", 0) for doc in docs)

        # Utilisation du ContextAnalyzer
        context_analysis = await context_analyzer.analyze_context(
            query=query,
            context_docs=docs,
            context_confidence=avg_confidence,
            summarizer=self.summarizer if hasattr(self, 'summarizer') else None,
            embeddings_model=self.embedding_model if hasattr(self, 'embedding_model') else None
        )

        return {
            **context_analysis,
            "has_context": True,
            "confidence": avg_confidence,
            "relevance_scores": relevance_scores,
            "total_tokens": total_tokens
        }

    async def _generate_with_fallback(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_config: GenerationConfig
    ) -> torch.Tensor:
        """Génère une réponse avec gestion des erreurs et fallback."""
        try:
            # Première tentative avec configuration complète
            outputs = self.model.generate(
                **inputs,  # inputs contient déjà attention_mask
                generation_config=generation_config,
                # Enlever attention_mask des paramètres additionnels
                output_attentions=False,
                output_hidden_states=False,
                return_dict_in_generate=False,
            )
            return outputs

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("OOM détecté, tentative avec configuration réduite")
                torch.cuda.empty_cache()
                
                # Configuration réduite
                fallback_config = GenerationConfig(
                    max_new_tokens=min(512, generation_config.max_new_tokens),
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=generation_config.pad_token_id,
                    eos_token_id=generation_config.eos_token_id,
                    repetition_penalty=1.0,
                    length_penalty=1.0
                )

                return self.model.generate(
                    **inputs,
                    generation_config=fallback_config
                )
            raise
