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
        conversation_history: Optional[List[Dict]] = None,
        language: str = "fr",
    ) -> Dict:
        try:
            start_time = datetime.utcnow()
            metrics.increment_counter("generation_requests")
            logger.info(f"Début génération réponse pour query: {query[:100]}...")

            # 1. Préparation du contexte
            relevant_sections = []
            if context_docs:
                for doc in context_docs:
                    if "processed_sections" in doc:
                        # Utilise les sections prétraitées
                        relevant_sections.extend([
                            section for section in doc["processed_sections"]
                            if isinstance(section, dict) and section.get("content")
                        ])
                    else:
                        # Fallback sur le contenu brut
                        relevant_sections.append({
                            "content": doc.get("content", ""),
                            "importance_score": 1.0,
                            "metadata": doc.get("metadata", {})
                        })

            # Tri et sélection des sections les plus pertinentes
            relevant_sections.sort(key=lambda x: float(x.get("importance_score", 0)), reverse=True)
            top_sections = relevant_sections[:3]  # Limite aux 3 meilleures sections

            # 2. Préparation de l'historique
            processed_history = []
            if conversation_history:
                # Limite l'historique aux 5 derniers messages
                for msg in conversation_history[-5:]:
                    if isinstance(msg, dict):
                        processed_history.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })

            # 3. Construction du contexte formaté
            formatted_context = "\n\n".join([
                f"[Source: {section.get('metadata', {}).get('source', 'Document')}]\n"
                f"{section['content']}"
                for section in top_sections
            ])

            # 4. Génération du prompt 
            with metrics.timer("prompt_construction"):
                prompt = self.prompt_system.build_chat_prompt(
                    messages=processed_history,
                    context=formatted_context,
                    query=query,
                    lang=language
                )

            # 5. Configuration de génération
            generation_config = GenerationConfig(**{
                k: v for k, v in settings.generation_config.items() 
                if k in GenerationConfig.__init__.__code__.co_varnames
            })

            # 6. Génération avec gestion automatique de la RAM/VRAM
            with metrics.timer("model_inference"):
                if settings.USE_FP16:
                    context_manager = torch.amp.autocast('cuda')
                else:
                    context_manager = nullcontext()

                with context_manager:
                    with torch.no_grad():
                        # Tokenisation
                        inputs = self.tokenizer_manager(
                            prompt,
                            max_length=settings.MAX_INPUT_LENGTH,
                            return_tensors="pt"
                        ).to(self.model.device)
                        
                        input_tokens_count = len(inputs.input_ids[0])
                        logger.info(f"Nombre de tokens d'entrée : {input_tokens_count} / {settings.MAX_INPUT_LENGTH}")
                        
                        # Génération
                        outputs = self.model.generate(
                            **inputs,
                            generation_config=generation_config,
                            use_cache=True
                        )

                        # Décodage et nettoyage
                        response_text = self.tokenizer_manager.decode_and_clean(outputs[0])
                        

            # 7. Post-traitement et métriques
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            input_tokens = len(inputs.input_ids[0])
            output_tokens = len(outputs[0]) - input_tokens

            # Calcul du score de confiance basé sur les sections utilisées
            confidence_score = sum(
                float(section.get("importance_score", 0)) 
                for section in top_sections
            ) / max(len(top_sections), 1)

            # 8. Préparation de la réponse
            response = {
                "response": self._extract_response(response_text),
                "confidence_score": min(confidence_score, 1.0),
                "processing_time": processing_time,
                "tokens_used": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                },
                "context_info": {
                    "num_sections_used": len(top_sections),
                    "sources": [
                        section.get("metadata", {}).get("source")
                        for section in top_sections
                    ]
                },
                "metadata": {
                    "model_version": settings.MODEL_NAME,
                    "timestamp": datetime.utcnow().isoformat(),
                    "language": language
                }
            }

            # Log des métriques
            metrics.increment_counter("successful_generations")
            metrics.set_gauge("last_generation_time", processing_time)
            metrics.set_gauge("last_tokens_used", input_tokens + output_tokens)

            logger.info(f"Génération réussie en {processing_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Erreur génération: {str(e)}", exc_info=True)
            metrics.increment_counter("generation_errors")
            raise
    
    def _extract_response(self, response_text: str) -> str:
        """
        Extrait uniquement la partie réponse du texte généré
        """
        try:
            # Étapes de nettoyage
            # 1. Supprimer le contenu système
            response = response_text.split('</|system|>')[-1]
            
            # 2. Supprimer le contexte
            response = response.split('</|context|>')[-1]
            
            # 3. Supprimer l'historique
            response = response.split('</|history|>')[-1]
            
            # 4. Extraire la partie après <|assistant|>
            response = response.split('<|assistant|>')[-1]
            
            # 5. Supprimer les balises résiduelles
            response = response.replace('<|user|>', '').replace('Réponse :', '').strip()
            
            return response

        except Exception as e:
            logger.error(f"Erreur extraction réponse: {e}")
            return response_text.strip()
        
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
        
    def _build_fallback_prompt(self, query: str) -> str:
        """Construit un prompt de secours en cas d'erreur."""
        return f"Question: {query}\nRéponse:"

    def _calculate_confidence(self, context_docs: Optional[List[Dict]]) -> float:
        """Calcule le score de confiance."""
        if not context_docs:
            return 0.5
        scores = [doc.get("score", 0.0) for doc in context_docs]
        return min(sum(scores) / len(scores) if scores else 0.5, 1.0)

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
