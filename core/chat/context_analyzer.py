# core/chat/context_analyzer.py
from typing import List, Dict, Optional, Set
from datetime import datetime
import re
import logging
from core.config.config import settings
from core.utils.logger import get_logger

logger = get_logger("context_analyzer")

class ContextAnalyzer:
    """Classe utilitaire pour l'analyse de contexte."""
    
    def __init__(self):
        self.min_query_length = settings.chat.MIN_QUERY_LENGTH
        self.max_themes = settings.chat.MAX_THEMES
        self.confidence_threshold = settings.chat.CONFIDENCE_THRESHOLD
        self.min_relevant_docs = settings.chat.MIN_RELEVANT_DOCS

    async def analyze_context(
        self,
        query: str,
        context_docs: List[Dict],
        context_confidence: float,
        summarizer = None,
        embeddings_model = None
    ) -> Dict:
        """
        Analyse complète du contexte incluant documents et requête.
        
        Args:
            query: Question de l'utilisateur
            context_docs: Documents de contexte
            context_confidence: Score de confiance
            summarizer: Instance optionnelle du summarizer
            embeddings_model: Instance optionnelle du modèle d'embeddings
            
        Returns:
            Dict contenant l'analyse complète
        """
        try:
            # Analyse sémantique de la requête
            query_analysis = await self._analyze_query(query)
            
            # Extraction et analyse des thèmes
            themes = set()
            ambiguity_score = 0.0
            all_content = ""

            for doc in context_docs:
                if doc.get("score", 0) < 0.5:
                    continue

                content = doc.get("content", "")
                all_content += f"\n{content}"
                doc_themes = self._extract_themes(content)
                themes.update(doc_themes)

                if len(doc_themes) > 1:
                    ambiguity_score += 0.2

            # Résumé du contexte si disponible
            if summarizer:
                try:
                    summary = await summarizer.summarize_documents(context_docs)
                except Exception as e:
                    logger.error(f"Erreur génération résumé: {e}")
                    summary = {
                        "structured_summary": all_content[:500],
                        "themes": list(themes),
                        "needs_clarification": False
                    }
            else:
                summary = {
                    "structured_summary": all_content[:500],
                    "themes": list(themes),
                    "needs_clarification": False
                }

            # Évaluation du besoin de clarification
            needs_clarification = any([
                len(themes) > self.max_themes,
                ambiguity_score > 0.5,
                context_confidence < self.confidence_threshold,
                len(context_docs) < self.min_relevant_docs,
                query_analysis.get("is_ambiguous", False),
                len(query.split()) < self.min_query_length
            ])

            # Construction du résultat final
            result = {
                "needs_clarification": needs_clarification,
                "context_confidence": context_confidence,
                "themes": list(themes),
                "ambiguity_score": ambiguity_score,
                "query_analysis": query_analysis,
                "response_type": self._determine_response_type(
                    query=query,
                    themes=themes,
                    confidence=context_confidence,
                    query_analysis=query_analysis
                ),
                "summary": summary.get("structured_summary") if summary else "",
                "clarification_reason": self._get_clarification_reason(
                    themes=themes,
                    confidence=context_confidence,
                    docs_count=len(context_docs),
                    query_analysis=query_analysis
                ),
                "key_concepts": query_analysis.get("key_concepts", []),
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "doc_count": len(context_docs),
                    "total_themes": len(themes)
                }
            }

            return result

        except Exception as e:
            logger.error(f"Erreur analyse contexte: {e}")
            return {
                "needs_clarification": True,
                "context_confidence": 0.0,
                "query_analysis": {
                    "is_ambiguous": True,
                    "key_concepts": [],
                    "type": "unknown"
                },
                "error": str(e)
            }

    async def _analyze_query(self, query: str) -> Dict:
        """Analyse sémantique de la requête."""
        return {
            "is_ambiguous": self._is_query_ambiguous(query),
            "type": self._get_query_type(query),
            "complexity": self._calculate_query_complexity(query),
            "key_concepts": self._extract_key_concepts(query)
        }

    def _is_query_ambiguous(self, query: str) -> bool:
        """Détection d'ambiguïté."""
        query_lower = query.lower()
        
        # Liste de verbes d'action spécifiques qui indiquent une intention claire
        clear_intent_verbs = {
            "créer", "modifier", "supprimer", "ajouter", "afficher",
            "commencer", "démarrer", "ouvrir", "fermer", "enregistrer"
        }
        
        # Si la requête commence par un verbe d'action clair, elle n'est pas ambiguë
        if any(query_lower.startswith(verb) for verb in clear_intent_verbs):
            return False
            
        # Si la requête commence par "comment" et contient un verbe d'action, elle n'est pas ambiguë
        if query_lower.startswith("comment") and any(verb in query_lower for verb in clear_intent_verbs):
            return False

        # Patterns d'ambiguïté réduits
        ambiguous_patterns = [
            r"^(ca|cela|ce|cette)\s*\?$",  # Questions très courtes avec démonstratifs
            r"^\s*\?\s*$",                  # Juste un point d'interrogation
            r"^[a-zA-Z]{1,2}\s*\?$",        # Une ou deux lettres suivies d'un point d'interrogation
            r"^qu(?:e|oi)\s+(?:est-ce|faire)\s*\?$"  # Questions très générales
        ]

        return any(re.search(pattern, query_lower) for pattern in ambiguous_patterns)

    def _get_query_type(self, query: str) -> str:
        """Détermine le type de requête."""
        query_lower = query.lower()
        
        if re.search(r"\b(comment|how|faire)\b", query_lower):
            return "procedural"
        if re.search(r"\b(pourquoi|why|raison)\b", query_lower):
            return "explanatory"
        if re.search(r"\b(différence|versus|vs|compare)\b", query_lower):
            return "comparative"
        if re.search(r"\b(est-ce que|est-il|possible)\b", query_lower):
            return "confirmation"
        return "informational"

    def _calculate_query_complexity(self, query: str) -> float:
        """Calcul de la complexité d'une requête."""
        words = query.split()
        
        # Facteurs de complexité ajustés
        factors = {
            "length": min(len(words) / 10, 1.0),  # Normalisé pour être plus permissif
            "technical_terms": len(re.findall(r"\b[A-Z][A-Za-z]*(?:\.[A-Za-z]+)+\b", query)) * 0.1,
            "specific_terms": len([w for w in words if w.lower() in {"comment", "créer", "modifier", "supprimer"}]) * 0.2,
        }
        
        # Si la requête contient des mots clés spécifiques, réduire la complexité
        if any(term in query.lower() for term in ["créer", "comment créer"]):
            return min(0.5, sum(factors.values()))
            
        return min(sum(factors.values()), 1.0)

    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extrait les concepts clés d'une requête."""
        stop_words = {"le", "la", "les", "un", "une", "des", "est", "sont", "a", "ont"}
        words = query.lower().split()
        return [w for w in words if w not in stop_words and len(w) > 3]

    def _extract_themes(self, content: str) -> Set[str]:
        """Extrait les thèmes principaux d'un contenu."""
        themes = set()
        
        # Patterns pour la détection de thèmes
        theme_patterns = [
            (r"(?i)configuration.*?(?:\.|$)", "Configuration"),
            (r"(?i)installation.*?(?:\.|$)", "Installation"),
            (r"(?i)erreur.*?(?:\.|$)", "Résolution d'erreurs"),
            (r"(?i)mise à jour.*?(?:\.|$)", "Mise à jour"),
            (r"(?i)sécurité.*?(?:\.|$)", "Sécurité"),
            (r"(?i)performance.*?(?:\.|$)", "Performance")
        ]
        
        for pattern, theme in theme_patterns:
            if re.search(pattern, content):
                themes.add(theme)
                
        return themes

    def _determine_response_type(
        self,
        query: str,
        themes: Set[str],
        confidence: float,
        query_analysis: Dict
    ) -> str:
        """Détermine le type de réponse approprié."""
        if query_analysis.get("is_ambiguous") or len(themes) > self.max_themes:
            return "clarification"

        complexity = query_analysis.get("complexity", 0.5)
        query_type = query_analysis.get("type", "informational")

        if query_type == "procedural" or complexity > 0.7:
            return "technical"
        if query_type == "confirmation" and confidence > 0.8:
            return "concise"
        if complexity < 0.3 and confidence > 0.9:
            return "concise"

        return "comprehensive"

    def _get_clarification_reason(
        self,
        themes: Set[str],
        confidence: float,
        docs_count: int,
        query_analysis: Dict
    ) -> str:
        """Détermine la raison principale nécessitant une clarification."""
        if len(themes) > self.max_themes:
            return "multiple_themes"
        if confidence < self.confidence_threshold:
            return "low_confidence"
        if docs_count < self.min_relevant_docs:
            return "insufficient_context"
        if query_analysis.get("is_ambiguous", False):
            return "query_ambiguity"
        return "general"

# Instance globale pour réutilisation
context_analyzer = ContextAnalyzer()
