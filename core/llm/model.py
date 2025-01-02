# core/llm/model.py
from typing import Dict, List, Optional, Union, Generator
from datetime import datetime
from contextlib import nullcontext
import torch
import asyncio
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, GenerationConfig
from sentence_transformers import SentenceTransformer
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from .cuda_manager import CUDAManager
from .memory_manager import MemoryManager
from .prompt_system import PromptSystem
from .summarizer import DocumentSummarizer
from .tokenizer_manager import TokenizerManager
from .auth_manager import HuggingFaceAuthManager

logger = get_logger("model")

class ModelInference:
    def __init__(self):
        """Initialise les composants de base du modèle d'inférence."""
        self.auth_manager = HuggingFaceAuthManager()
        self.cuda_manager = CUDAManager()
        self.memory_manager = MemoryManager()
        self.tokenizer_manager = TokenizerManager()
        self.model = None
        self.embedding_model = None
        self._initialized = False
        self.prompt_system = PromptSystem()
        self.summarizer = DocumentSummarizer()

    async def initialize(self):
        """Initialise le modèle de manière asynchrone."""
        if not self._initialized:
            try:
                # Authentification Hugging Face
                if not await self.auth_manager.setup_auth():
                    raise ValueError("Échec de l'authentification Hugging Face")

                # Vérification de l'accès au modèle
                if not self.auth_manager.get_model_access(settings.MODEL_NAME):
                    raise ValueError(f"Accès non autorisé au modèle {settings.MODEL_NAME}")

                # Initialisation des modèles
                await self._initialize_model()
                await self._initialize_embedding_model()
                await self.summarizer.initialize()
                
                self._initialized = True
                logger.info(f"Modèle {settings.MODEL_NAME} initialisé avec succès")

            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du modèle: {e}")
                raise

    async def _initialize_model(self):
        """Configure et charge le modèle."""
        try:
            logger.info(f"Chargement du modèle {settings.MODEL_NAME}")

            # Récupération de la configuration mémoire depuis settings via memory_manager
            max_memory = self.memory_manager.get_optimal_memory_config()
            
            # Paramètres de chargement en utilisant les settings
            load_params = {
                "pretrained_model_name_or_path": settings.MODEL_NAME,
                "revision": settings.MODEL_REVISION,
                "device_map": "auto",
                "max_memory": max_memory,
                "trust_remote_code": True
            }

            # Configuration du type de données en fonction des settings
            if settings.USE_FP16:
                load_params["torch_dtype"] = torch.float16
            elif settings.USE_8BIT:
                load_params["load_in_8bit"] = True
            elif settings.USE_4BIT:
                load_params["load_in_4bit"] = True
                if hasattr(settings, "BNB_4BIT_COMPUTE_DTYPE"):
                    load_params["bnb_4bit_compute_dtype"] = getattr(torch, settings.BNB_4BIT_COMPUTE_DTYPE)

            logger.info(f"Configuration de chargement: {load_params}")

            # Chargement du modèle avec autocast si FP16 est activé
            if settings.USE_FP16:
                with torch.amp.autocast('cuda'):
                    self.model = AutoModelForCausalLM.from_pretrained(**load_params)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(**load_params)

            # Post-initialisation
            model_device = next(self.model.parameters()).device
            logger.info(f"Modèle chargé sur {model_device}")
            logger.info(f"Type de données: {next(self.model.parameters()).dtype}")
            metrics.increment_counter("model_loads")

        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise

    async def _initialize_embedding_model(self):
        """Initialise le modèle d'embeddings."""
        try:
            logger.info(f"Chargement du modèle d'embeddings {settings.EMBEDDING_MODEL}")
            
            # Configuration du device
            device = "cuda" if torch.cuda.is_available() and not settings.USE_CPU_ONLY else "cpu"
            
            # Chargement du modèle d'embeddings
            self.embedding_model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=device,
                cache_folder=str(settings.CACHE_DIR)
            )
            
            logger.info(f"Modèle d'embeddings chargé sur {device}")
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle d'embeddings: {e}")
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

    async def generate_response(
        self,
        query: str,
        context_docs: Optional[List[Dict]] = None,
        context_summary: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
        language: str = "fr",
    ) -> Dict:
        try:
            start_time = datetime.utcnow()
            metrics.increment_counter("generation_requests")
            logger.info(f"Début génération réponse pour query: {query[:100]}...")

            # 1. Analyse et résumé du contexte si non fourni
            if context_docs and not context_summary:
                context_summary = await self.summarizer.summarize_documents(context_docs)

            # 2. Préparation du prompt basé sur l'analyse thématique
            prompt_context = ""
            needs_clarification = False

            if context_summary:
                # Si des clarifications sont nécessaires et pas d'historique
                if context_summary.get("clarifications_needed", False) and not conversation_history:
                    needs_clarification = True
                    prompt_context = f"""
                    Plusieurs thèmes ont été identifiés:
                    {context_summary.get('structured_summary', '')}
                    
                    Avant de répondre, je devrais demander des clarifications à l'utilisateur.
                    """
                else:
                    prompt_context = context_summary.get('structured_summary', '')

            # 3. Construction du prompt complet
            with metrics.timer("prompt_construction"):
                prompt = self.prompt_system.build_chat_prompt(
                    messages=conversation_history or [],
                    context=prompt_context,
                    query=query,
                    lang=language
                )

            # 4. Configuration de génération
            generation_config = GenerationConfig(**settings.generation_config)

            # 5. Génération avec gestion automatique de la RAM/VRAM
            with metrics.timer("model_inference"):
                context_manager = torch.amp.autocast('cuda') if settings.USE_FP16 else nullcontext()

                with context_manager:
                    with torch.no_grad():
                        # Tokenisation
                        inputs = self.tokenizer_manager(
                            prompt,
                            max_length=settings.MAX_INPUT_LENGTH,
                            return_tensors="pt"
                        ).to(self.model.device)

                        input_tokens_count = len(inputs.input_ids[0])
                        logger.info(f"Nombre de tokens d'entrée : {input_tokens_count}")

                        # Si des clarifications sont nécessaires
                        if needs_clarification:
                            # Génération d'une réponse demandant des précisions
                            questions = context_summary.get('questions', [])
                            response_text = self._generate_clarification_response(
                                questions=questions,
                                themes=context_summary.get('themes', [])
                            )
                            output_tokens = len(self.tokenizer_manager(response_text)['input_ids'])
                        else:
                            # Génération normale
                            outputs = self.model.generate(
                                **inputs,
                                generation_config=generation_config,
                                use_cache=True
                            )
                            response_text = self.tokenizer_manager.decode_and_clean(outputs[0])
                            output_tokens = len(outputs[0]) - input_tokens_count

            # 6. Calcul du score de confiance et métriques
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            confidence_score = self._calculate_confidence(
                context_docs,
                has_clarification=needs_clarification
            )

            response = {
                "response": response_text,
                "confidence_score": confidence_score,
                "processing_time": processing_time,
                "tokens_used": {
                    "input": input_tokens_count,
                    "output": output_tokens,
                    "total": input_tokens_count + output_tokens
                },
                "needs_clarification": needs_clarification,
                "themes": context_summary.get('themes', []) if context_summary else [],
                "metadata": {
                    "model_version": settings.MODEL_NAME,
                    "timestamp": datetime.utcnow().isoformat(),
                    "language": language
                }
            }

            # Log des métriques
            metrics.increment_counter("successful_generations")
            metrics.set_gauge("last_generation_time", processing_time)
            metrics.set_gauge("last_tokens_used", response["tokens_used"]["total"])

            logger.info(f"Génération réussie en {processing_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Erreur génération: {str(e)}", exc_info=True)
            metrics.increment_counter("generation_errors")
            raise

    def _generate_clarification_response(
        self,
        questions: List[str],
        themes: List[str]
    ) -> str:
        """Génère une réponse demandant des clarifications."""
        if not questions:
            questions = [
                f"Pourriez-vous préciser si votre question concerne l'un de ces thèmes : {', '.join(themes)}?"
            ]
        
        response = (
            "Pour mieux vous aider, j'aurais besoin de quelques précisions. "
            f"\n\n{questions[0]}"
        )
        
        if len(questions) > 1:
            response += f"\n\nJe pourrais aussi avoir besoin de savoir :\n" + \
                       "\n".join(f"- {q}" for q in questions[1:])
        
        return response

    def _calculate_confidence(
        self,
        context_docs: Optional[List[Dict]],
        has_clarification: bool = False
    ) -> float:
        """Calcule le score de confiance en tenant compte des clarifications."""
        if has_clarification:
            # Score de confiance réduit si des clarifications sont nécessaires
            return 0.5
            
        if not context_docs:
            return 0.3
            
        scores = [doc.get("score", 0.0) for doc in context_docs]
        return min(sum(scores) / len(scores) if scores else 0.5, 1.0)
        
    def _format_context_docs(self, docs: List[Dict]) -> str:
        """Formate les documents de contexte."""
        if not docs:
            return ""
            
        formatted_docs = []
        for doc in docs:
            if "processed_sections" in doc:
                # Utilise les sections prétraitées
                sections = doc["processed_sections"]
                content = "\n\n".join(s["content"] for s in sections)
            else:
                content = doc.get("content", "")
                
            formatted_docs.append(content)
            
        return "\n\n---\n\n".join(formatted_docs)
        
    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            await self.memory_manager.cleanup()
            await self.cuda_manager.cleanup()
            await self.tokenizer_manager.cleanup()
            
            if self.model is not None:
                try:
                    self.model.cpu()
                    del self.model
                    self.model = None
                except Exception as e:
                    logger.warning(f"Erreur nettoyage modèle génératif: {e}")
                    
            if self.embedding_model is not None:
                try:
                    self.embedding_model.cpu()
                    del self.embedding_model
                    self.embedding_model = None
                except Exception as e:
                    logger.warning(f"Erreur nettoyage modèle d'embeddings: {e}")

            self._initialized = False

        except Exception as e:
            logger.error(f"Erreur nettoyage ressources: {e}")
