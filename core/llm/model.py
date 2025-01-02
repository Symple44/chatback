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
        
        # Nouveaux paramètres de configuration
        self.max_context_docs = 6
        #self.min_confidence_threshold = 0.6
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

            # Initialiser TokenizerManager avec le tokenizer
            self.tokenizer_manager = TokenizerManager()
            
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
        response_type: str = "comprehensive"
    ) -> Dict:
        """
        Génère une réponse en tenant compte du contexte et de l'historique.
        
        Args:
            query: Question de l'utilisateur
            context_docs: Documents pertinents (optionnel)
            context_summary: Résumé du contexte (optionnel)
            conversation_history: Historique de la conversation
            language: Code de langue
            response_type: Type de réponse ("comprehensive", "concise", "technical")
            
        Returns:
            Dict contenant la réponse et les métadonnées
        """
        try:
            start_time = datetime.utcnow()
            metrics.increment_counter("generation_requests")

            # 1. Validation et préparation du contexte
            validated_docs = await self._validate_and_prepare_context(context_docs)
            context_info = await self._analyze_context_relevance(validated_docs, query)

            # 1.5 Analyse et résumé du contexte
            if validated_docs:
                # Générer le résumé seulement si nécessaire
                if not context_summary:
                    try:
                        context_summary = await self.summarizer.summarize_documents(validated_docs)
                        logger.debug(f"Résumé généré: {context_summary}")
                        
                        # Mise à jour du context_info avec les informations du résumé
                        if isinstance(context_summary, dict):
                            context_info.update({
                                "themes": context_summary.get("themes", []),
                                "key_points": context_summary.get("key_points", []),
                                "needs_clarification": context_summary.get("needs_clarification", False)
                            })
                    except Exception as e:
                        logger.error(f"Erreur génération résumé: {e}")
                        # En cas d'erreur, on utilise une version simplifiée du contexte
                        context_summary = {
                            "raw_summary": "\n".join(
                                doc.get("content", "")[:200] for doc in validated_docs[:2]
                            )
                        }
                        
            # 2. Construction du prompt avec le nouveau format en utilisant prioritairement le résumé
            prompt = await self.prompt_system.build_chat_prompt(
                query=query,
                context_docs=validated_docs,
                context_summary=context_summary,
                conversation_history=conversation_history,
                context_info=context_info,
                language=language,
                response_type=response_type
            )
            
            logger.info(prompt)
            # 3. Génération avec gestion de la mémoire
            with metrics.timer("model_inference"):
                generation_config = self._get_generation_config(response_type)
                
                with torch.amp.autocast('cuda', enabled=settings.USE_FP16):
                    with torch.no_grad():
                        # Tokenisation avec gestion de la longueur
                        inputs = self.tokenizer_manager.encode_with_truncation(
                            prompt,
                            max_length=settings.MAX_INPUT_LENGTH,
                            return_tensors="pt"
                        ).to(self.model.device)
                        
                        # Génération
                        outputs = await self._generate_with_fallback(
                            inputs,
                            generation_config
                        )
                        
                        response_text = self.tokenizer_manager.decode_and_clean(
                            outputs[0],
                            skip_assistant_token=True
                        )

            # 4. Post-traitement et calcul des métriques
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "response": response_text,
                "confidence_score": context_info["confidence"],
                "processing_time": processing_time,
                "tokens_used": {
                    "input": len(inputs["input_ids"][0]),
                    "output": len(outputs[0]) - len(inputs["input_ids"][0]),
                    "total": len(outputs[0])
                },
                "context_info": context_info,
                "metadata": {
                    "model_version": settings.MODEL_NAME,
                    "response_type": response_type,
                    "language": language,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Erreur génération: {str(e)}", exc_info=True)
            metrics.increment_counter("generation_errors")
            raise
    
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
            early_stopping=True
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
        """
        Analyse la pertinence du contexte par rapport à la requête.
        """
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

        return {
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
        """
        Génère une réponse avec gestion des erreurs et fallback.
        """
        try:
            # Première tentative avec configuration complète
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
            return outputs
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Libération de la mémoire
                torch.cuda.empty_cache()
                logger.warning("OOM détecté, tentative avec configuration réduite")

                # Configuration réduite
                fallback_config = GenerationConfig(
                    max_new_tokens=min(512, generation_config.max_new_tokens),
                    do_sample=False,
                    num_beams=1
                )

                return self.model.generate(
                    **inputs,
                    generation_config=fallback_config
                )
            raise
