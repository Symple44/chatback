# core/llm/summarizer.py
import torch
import torch.amp
import re
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import defaultdict
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger("summarizer")

class DocumentSummarizer:
    def __init__(self):
        """Initialise le résumeur de contexte avec analyse thématique."""
        self.model_name = settings.MODEL_NAME_SUMMARIZER
        self.tokenizer = None
        self.model = None
        self._initialized = False
        
        self.max_length = 512
        self.min_length = 50
        
        self.generation_params = {
            "max_length": self.max_length,
            "min_length": self.min_length,
            "num_beams": 4,
            "length_penalty": 2.0,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "do_sample": False,
            "temperature": 1.0
        }

    async def initialize(self):
        """Initialise le modèle de résumé."""
        if self._initialized:
            return

        try:
            logger.info(f"Initialisation du résumeur de contexte: {self.model_name}")
            
            # Configuration du tokenizer avec padding côté gauche pour optimiser le traitement batch
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=self.max_length,
                padding_side='left'
            )

            # Configuration optimisée pour l'inférence
            model_config = {
                "torch_dtype": torch.bfloat16,  # Utilisation de bfloat16 pour la cohérence
                "low_cpu_mem_usage": True,
                "device_map": "auto",
                "max_memory": {0: "4GiB", "cpu": "24GB"}  # Allocation mémoire réduite
            }

            # Chargement du modèle avec autocast pour bfloat16
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    **model_config
                )

            # Force le modèle en mode évaluation
            self.model.eval()

            # Configuration CUDA si disponible
            if torch.cuda.is_available() and not settings.USE_CPU_ONLY:
                if not hasattr(self.model, "device_map"):  # Si pas de device_map auto
                    self.model = self.model.to("cuda")
                # Activation de la fusion des kernels cuDNN
                torch.backends.cudnn.benchmark = True

            self._initialized = True
            logger.info("Résumeur de contexte initialisé")
            
        except Exception as e:
            logger.error(f"Erreur initialisation summarizer: {e}")
            raise

    async def summarize_documents(self, documents: List[Dict]) -> Dict[str, str]:
        """Crée un résumé structuré avec analyse thématique des documents."""
        if not self._initialized:
            await self.initialize()

        try:
            with metrics.timer("context_analysis"):
                # Analyse des documents par application/thème
                grouped_docs = self._group_documents(documents)
                themes, clarifications = self._analyze_themes(grouped_docs)
                
                # Génération du résumé principal avec gestion mémoire optimisée
                main_summary = await self._generate_summary(documents)
                questions = self._generate_clarifying_questions(themes, clarifications)
                
                return {
                    "structured_summary": self._format_structured_summary(
                        main_summary, themes, clarifications, questions
                    ),
                    "themes": themes,
                    "clarifications_needed": bool(clarifications),
                    "questions": questions,
                    "raw_summary": main_summary
                }

        except Exception as e:
            logger.error(f"Erreur génération résumé: {e}")
            metrics.increment_counter("summarization_errors")
            return self._get_default_summary()

    async def _generate_summary(self, documents: List[Dict]) -> str:
        """Génère le résumé principal des documents avec optimisation CUDA."""
        try:
            content_to_summarize = self._prepare_content(documents)
            if not content_to_summarize:
                return ""

            # Tokenisation optimisée
            inputs = self.tokenizer(
                content_to_summarize,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            # Envoi sur GPU si disponible
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Génération avec autocast bfloat16
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                summary_ids = self.model.generate(
                    **inputs,
                    **self.generation_params
                )

            # Décodage optimisé
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Nettoyage mémoire explicite
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return summary.strip()

        except Exception as e:
            logger.error(f"Erreur génération résumé principal: {e}")
            return ""

    def _format_structured_summary(
        self,
        main_summary: str,
        themes: List[str],
        clarifications: List[str],
        questions: List[str]
    ) -> str:
        """Formate le résumé structuré."""
        return f"""
Thèmes principaux identifiés:
{self._format_themes(themes)}

Résumé du contexte:
{main_summary}

Points nécessitant clarification:
{self._format_clarifications(clarifications)}

Questions suggérées:
{self._format_questions(questions)}
""".strip()

    def _get_default_summary(self) -> Dict[str, str]:
        """Retourne un résumé par défaut en cas d'erreur."""
        return {
            "structured_summary": "Une erreur est survenue lors de la génération du résumé.",
            "themes": [],
            "clarifications_needed": False,
            "questions": [],
            "raw_summary": ""
        }

    def _group_documents(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Groupe les documents par application/thème."""
        grouped = defaultdict(list)
        
        for doc in documents:
            if not isinstance(doc, dict):
                continue
                
            # Récupération des métadonnées
            metadata = doc.get("metadata", {})
            app_name = metadata.get("application", "unknown")
            
            # Ajout du document au groupe correspondant
            grouped[app_name].append(doc)
            
        return dict(grouped)

    def _analyze_themes(self, grouped_docs: Dict[str, List[Dict]]) -> Tuple[List[str], List[str]]:
        """Analyse les thèmes principaux et identifie les ambiguïtés."""
        themes = []
        clarifications = []
        
        for app_name, docs in grouped_docs.items():
            # Analyse des thèmes par application
            app_themes = set()
            for doc in docs:
                # Extraction des thèmes du contenu
                content = self._extract_content(doc)
                doc_themes = self._extract_themes(content)
                app_themes.update(doc_themes)
            
            # Ajout des thèmes identifiés
            themes.extend([f"{app_name}: {theme}" for theme in app_themes])
            
            # Identification des ambiguïtés
            if len(app_themes) > 1:
                clarifications.append(
                    f"Multiple thèmes détectés pour {app_name}: {', '.join(app_themes)}"
                )
        
        return themes, clarifications

    def _extract_themes(self, content: str) -> List[str]:
        """Extrait les thèmes principaux d'un contenu."""
        themes = set()
        
        # Recherche de mots-clés thématiques
        patterns = [
            r"(?i)configuration.*?(?:\.|$)",
            r"(?i)installation.*?(?:\.|$)",
            r"(?i)erreur.*?(?:\.|$)",
            r"(?i)problème.*?(?:\.|$)",
            r"(?i)mise à jour.*?(?:\.|$)",
            r"(?i)fonctionnalité.*?(?:\.|$)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                themes.add(match.group(0).strip().capitalize())
                
        return list(themes)

    def _generate_clarifying_questions(
        self,
        themes: List[str],
        clarifications: List[str]
    ) -> List[str]:
        """Génère des questions de clarification basées sur l'analyse."""
        questions = []
        
        # Questions basées sur les thèmes multiples
        if len(themes) > 1:
            themes_str = ", ".join(themes[:-1]) + " ou " + themes[-1]
            questions.append(f"Votre question concerne-t-elle {themes_str} ?")
        
        # Questions basées sur les ambiguïtés
        for clarification in clarifications:
            if "Multiple thèmes" in clarification:
                app = clarification.split("Multiple thèmes détectés pour ")[1].split(":")[0]
                questions.append(f"Pouvez-vous préciser le thème spécifique concernant {app} ?")
        
        return questions

    def _prepare_content(self, documents: List[Dict]) -> str:
        """Prépare le contenu pour la génération du résumé."""
        content_parts = []
        
        for doc in documents:
            if not isinstance(doc, dict):
                continue
                
            content = self._extract_content(doc)
            if content:
                # Ajout des métadonnées pertinentes
                metadata = doc.get("metadata", {})
                app_name = metadata.get("application", "unknown")
                content_parts.append(f"[Application: {app_name}]\n{content}")

        return "\n---\n".join(content_parts)

    def _extract_content(self, doc: Dict) -> str:
        """Extrait le contenu pertinent d'un document."""
        if "processed_sections" in doc:
            sections = sorted(
                doc["processed_sections"],
                key=lambda x: x.get("importance_score", 0),
                reverse=True
            )
            return "\n".join(section.get("content", "") for section in sections[:2])
        return doc.get("content", "").strip()

    def _format_themes(self, themes: List[str]) -> str:
        """Formate la liste des thèmes."""
        return "• " + "\n• ".join(themes) if themes else "Aucun thème spécifique identifié"

    def _format_clarifications(self, clarifications: List[str]) -> str:
        """Formate la liste des points nécessitant clarification."""
        return "• " + "\n• ".join(clarifications) if clarifications else "Aucune clarification nécessaire"

    def _format_questions(self, questions: List[str]) -> str:
        """Formate la liste des questions."""
        return "• " + "\n• ".join(questions) if questions else "Aucune question supplémentaire"

    async def cleanup(self):
        """Nettoie les ressources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._initialized = False