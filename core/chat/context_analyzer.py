# core/utils/context_analyzer.py
from typing import List, Dict, Optional, Set
from datetime import datetime
import re
import logging
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("context_analyzer")

class ContextAnalyzer:
    """Classe utilitaire pour l'analyse de contexte."""
    
    def __init__(self):
        self.min_query_length = 3
        self.max_themes = 2
        self.confidence_threshold = 0.7
        self.min_relevant_docs = 2

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
        """Vérifie si une requête est ambiguë."""
        if len(query.split()) < self.min_query_length:
            return True

        ambiguous_patterns = [
            r"^(que|quoi|comment|pourquoi).{0,20}\?$",  # Questions trop courtes
            r"\b(tout|tous|general|global)\b",          # Termes trop généraux
            r"\b(autre|plus|encore)\b\s*\?",            # Demandes ouvertes
            r"\b(ca|cela|ce|cette)\b",                  # Références vagues
            r"\b(ils|elles|leur|ces)\b"                 # Pronoms sans contexte
        ]

        return any(re.search(pattern, query.lower()) for pattern in ambiguous_patterns)

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
        """Calcule la complexité d'une requête."""
        factors = {
            "length": len(query.split()) / 20,  # Normalisation par 20 mots
            "technical_terms": len(re.findall(r"\b[A-Z][A-Za-z]*(?:\.[A-Za-z]+)+\b", query)) * 0.2,
            "conjunctions": len(re.findall(r"\b(et|ou|mais|donc|car)\b", query)) * 0.1,
            "special_chars": len(re.findall(r"[^\w\s]", query)) * 0.05
        }
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
