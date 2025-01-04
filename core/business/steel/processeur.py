# core/business/steel/processeur.py
from typing import Dict, Optional, List
from datetime import datetime
from core.chat.base_processor import BaseProcessor
from .analyseur import AnalyseurConstructionMetallique
from core.utils.logger import get_logger

logger = get_logger("processeur_steel")

class ProcesseurSteel(BaseProcessor):
    """Processeur spécialisé pour la construction métallique."""

    def __init__(self, components):
        super().__init__(components)
        self.analyseur = AnalyseurConstructionMetallique()
        self.model = components.model
        self.es_client = components.es_client

    async def process_message(
        self,
        request: Dict,
        context: Optional[Dict] = None
    ) -> Dict:
        """Traite un message dans le contexte de la construction métallique."""
        try:
            # Extraction de la requête
            query = request["query"]
            metadata = request.get("metadata", {})

            # Analyse métier de la requête
            analyse_metier = await self.analyseur.analyser_requete(query)

            # Recherche de documents avec contexte métier
            relevant_docs = await self._rechercher_documents_contexte(
                query=query,
                analyse=analyse_metier,
                metadata=metadata
            )

            # Si besoin de clarification
            if analyse_metier.get("besoin_clarification", False):
                return await self._generer_reponse_clarification(
                    analyse_metier,
                    request,
                    relevant_docs
                )

            # Génération de la réponse avec contexte métier
            response = await self._generer_reponse_metier(
                query=query,
                analyse=analyse_metier,
                documents=relevant_docs,
                langue=request.get("language", "fr")
            )

            return {
                "response": response.get("response"),
                "confidence_score": analyse_metier.get("confiance", 0.0),
                "documents": relevant_docs,
                "tokens_used": response.get("tokens_used", {}),
                "metadata": {
                    "processor_type": "steel",
                    "process_type": analyse_metier.get("type_demande"),
                    "technical_context": analyse_metier.get("specifications", {}),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Erreur traitement message steel: {e}")
            return self._generer_reponse_erreur()

    async def _rechercher_documents_contexte(
        self,
        query: str,
        analyse: Dict,
        metadata: Dict
    ) -> List[Dict]:
        """Recherche des documents avec contexte métier."""
        try:
            # Construction du filtre métier
            filtre_metier = {
                "domain": "steel",
                "document_type": analyse.get("type_demande"),
            }

            if metadata.get("type_ouvrage"):
                filtre_metier["type_ouvrage"] = metadata["type_ouvrage"]

            # Recherche dans Elasticsearch
            return await self.es_client.search_documents(
                query=query,
                metadata_filter=filtre_metier,
                size=5
            )

        except Exception as e:
            logger.error(f"Erreur recherche documents: {e}")
            return []

    async def _generer_reponse_metier(
        self,
        query: str,
        analyse: Dict,
        documents: List[Dict],
        langue: str = "fr"
    ) -> Dict:
        """Génère une réponse adaptée au contexte métier."""
        try:
            # Construction du prompt enrichi
            prompt = f"""
En tant qu'assistant spécialisé en construction métallique, 
pour une demande de type '{analyse.get('type_demande', 'général')}', 
{self._construire_contexte_technique(analyse)}
Comment répondre à la question: {query}
            """

            # Génération de la réponse
            response = await self.model.generate_response(
                query=prompt,
                context_docs=documents,
                language=langue,
                response_type=self._determiner_type_reponse(analyse)
            )

            return response

        except Exception as e:
            logger.error(f"Erreur génération réponse: {e}")
            return self._generer_reponse_erreur()

    def _construire_contexte_technique(self, analyse: Dict) -> str:
        """Construit le contexte technique pour le prompt."""
        contexte = []
        specs = analyse.get("specifications", {})

        if "elements_identifies" in specs:
            elements = ", ".join(specs["elements_identifies"])
            if elements:
                contexte.append(f"concernant les éléments suivants : {elements}")

        return "\n".join(contexte) if contexte else ""

    def _determiner_type_reponse(self, analyse: Dict) -> str:
        """Détermine le type de réponse approprié."""
        type_demande = analyse.get("type_demande", "").lower()

        if type_demande in ["creation", "modification"]:
            return "technical"
        elif type_demande == "consultation":
            return "comprehensive"
        else:
            return "general"

    async def _generer_reponse_clarification(
        self,
        analyse: Dict,
        request: Dict,
        documents: List[Dict]
    ) -> Dict:
        """Génère une réponse de demande de clarification."""
        points_clarification = analyse.get("points_clarification", [])
        
        template_clarification = """
Pour mieux vous aider avec votre demande en construction métallique, j'ai besoin de quelques précisions :

{points}

{suggestions}
        """

        # Construction de la réponse
        response_text = template_clarification.format(
            points="\n".join(f"- {point}" for point in points_clarification),
            suggestions=self._generer_suggestions(analyse)
        )

        return {
            "response": response_text.strip(),
            "confidence_score": analyse.get("confiance", 0.0),
            "documents": documents,
            "metadata": {
                "needs_clarification": True,
                "clarification_points": points_clarification,
                "processor_type": "steel",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    def _generer_suggestions(self, analyse: Dict) -> str:
        """Génère des suggestions selon le contexte."""
        type_demande = analyse.get("type_demande", "").lower()
        suggestions = []

        if type_demande == "creation":
            suggestions.extend([
                "- Pouvez-vous préciser le type d'ouvrage (charpente, serrurerie, etc.) ?",
                "- Avez-vous des spécifications techniques particulières ?",
                "- Y a-t-il des contraintes de délai ?"
            ])
        elif type_demande == "consultation":
            suggestions.extend([
                "- Quel type d'information recherchez-vous spécifiquement ?",
                "- Dans quel contexte avez-vous besoin de ces informations ?"
            ])

        return "\nSuggestions pour clarifier votre demande :\n" + "\n".join(suggestions) if suggestions else ""

    def _generer_reponse_erreur(self) -> Dict:
        """Génère une réponse d'erreur standard."""
        return {
            "response": "Je suis désolé, une erreur est survenue lors du traitement de votre demande. "
                      "Pouvez-vous reformuler ou préciser votre question ?",
            "confidence_score": 0.0,
            "documents": [],
            "metadata": {
                "error": True,
                "processor_type": "steel",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
