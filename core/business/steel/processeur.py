# core/business/steel/processeur.py
from typing import Dict, Optional, List, Any
from datetime import datetime
from core.chat.base_processor import BaseProcessor
from .analyseur import AnalyseurConstructionMetallique
from .constantes import (
    PROCESSUS_METIER,
    MATERIAUX,
    NORMES,
    TRAITEMENTS_SURFACE,
    DOCUMENTS_REGLEMENTAIRES
)
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
        """
        Traite un message dans le contexte de la construction métallique.
        """
        try:
            start_time = datetime.utcnow()
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
            if analyse_metier.besoin_clarification:
                return await self._generer_reponse_clarification(
                    analyse_metier,
                    request,
                    start_time
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
                "confidence_score": analyse_metier.confiance,
                "documents": relevant_docs,
                "tokens_used": response.get("tokens_used", {}),
                "metadata": {
                    "processor_type": "steel",
                    "process_type": analyse_metier.type_demande,
                    "technical_context": analyse_metier.specifications,
                    "business_context": analyse_metier.contexte_supplementaire,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Erreur traitement message steel: {e}")
            raise

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

            # Ajout des filtres spécifiques
            if "type_ouvrage" in metadata:
                filtre_metier["type_ouvrage"] = metadata["type_ouvrage"]

            if analyse.get("specifications", {}).get("materiaux"):
                filtre_metier["materiaux"] = analyse["specifications"]["materiaux"]

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
            # Préparation du prompt avec contexte métier
            prompt_metier = self._construire_prompt_metier(
                query=query,
                analyse=analyse,
                documents=documents
            )

            # Génération de la réponse
            response = await self.model.generate_response(
                query=prompt_metier,
                context_docs=documents,
                language=langue,
                response_type=self._determiner_type_reponse(analyse)
            )

            return response

        except Exception as e:
            logger.error(f"Erreur génération réponse: {e}")
            raise

    def _construire_prompt_metier(
        self,
        query: str,
        analyse: Dict,
        documents: List[Dict]
    ) -> str:
        """Construit un prompt enrichi avec le contexte métier."""
        type_demande = analyse.get("type_demande", "")
        specs = analyse.get("specifications", {})

        # Base du prompt
        prompt_parts = [
            f"En tant qu'assistant spécialisé en construction métallique, ",
            f"pour une demande de type '{type_demande}', "
        ]

        # Ajout du contexte technique si présent
        if specs.get("materiaux"):
            mats = ", ".join(mat["type"] for mat in specs["materiaux"])
            prompt_parts.append(f"concernant les matériaux suivants: {mats}, ")

        if specs.get("traitements"):
            traitement = ", ".join(specs["traitements"])
            prompt_parts.append(f"avec traitement {traitement}, ")

        # Ajout des normes applicables
        if analyse.get("normes_applicables"):
            normes = ", ".join(analyse["normes_applicables"])
            prompt_parts.append(f"en respectant les normes {normes}, ")

        # Ajout de la question
        prompt_parts.append(f"\nComment répondre à la question: {query}")

        return "".join(prompt_parts)

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
        start_time: datetime
    ) -> Dict:
        """Génère une réponse de demande de clarification."""
        points_clarification = analyse.get("points_clarification", [])
        template_clarification = """
Pour mieux vous aider avec votre demande en construction métallique, j'ai besoin de quelques précisions :

{points}

{context_specifique}

{references_utiles}
        """

        # Contexte spécifique selon le type de demande
        context_specifique = self._get_contexte_specifique(analyse)
        
        # Construction de la réponse
        response_text = template_clarification.format(
            points="\n".join(f"- {point}" for point in points_clarification),
            context_specifique=context_specifique,
            references_utiles=self._get_references_utiles(analyse)
        )

        return {
            "response": response_text.strip(),
            "confidence_score": analyse.get("confiance", 0.0),
            "documents": [],
            "metadata": {
                "needs_clarification": True,
                "clarification_points": points_clarification,
                "processor_type": "steel",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    def _get_references_utiles(self, analyse: Dict) -> str:
        """Retourne les références utiles selon le contexte."""
        references = []
        type_demande = analyse.get("type_demande", "").lower()
        
        if type_demande == "creation":
            references.extend(DOCUMENTS_REGLEMENTAIRES.get("TECHNIQUE", []))
        elif "normes_applicables" in analyse:
            for norme in analyse["normes_applicables"]:
                if norme in NORMES["CONSTRUCTION"]:
                    references.append(f"- {norme}: {NORMES['CONSTRUCTION'][norme]}")

        if references:
            return "\nRéférences utiles :\n" + "\n".join(references)
        return ""

    def _get_contexte_specifique(self, analyse: Dict) -> str:
        """Retourne le contexte spécifique selon le type de demande."""
        type_demande = analyse.get("type_demande", "").lower()
        
        if type_demande in PROCESSUS_METIER:
            process = PROCESSUS_METIER[type_demande]
            return f"\nPour les {process['nom']}, nous avons besoin des documents suivants :\n" + \
                   "\n".join(f"- {doc}" for doc in process['documents'])
        return ""
