# core/business/steel/processeur.py
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from .analyseur import AnalyseurConstructionMetallique, AnalyseMetier
from .constantes import (
    PROCESSUS_METIER,
    MATERIAUX,
    NORMES,
    TRAITEMENTS_SURFACE
)
from .modeles import (
    TypeOuvrage,
    StatutDocument,
    SpecificationMatiere,
    SpecificationTechnique
)
from core.utils.logger import get_logger

logger = get_logger("processeur_steel")

class ProcesseurConstructionMetallique:
    """Processeur métier pour la construction métallique."""

    def __init__(self, components):
        self.components = components
        self.analyseur = AnalyseurConstructionMetallique()
        self._charger_templates_reponses()

    def _charger_templates_reponses(self):
        """Charge les templates de réponses métier."""
        self.templates = {
            "creation_devis": """
Pour créer un devis de {type_ouvrage}, voici la procédure dans 2CM Manager :

1. Accès
   - Menu "Devis" > "Nouveau devis"
   - Sélection du type : {type_ouvrage}

2. Informations client
   - Sélection du client ou création si nouveau
   - Vérification des conditions commerciales

3. Spécifications techniques
   {specs_techniques}

4. Documents nécessaires
   {documents_requis}

5. Points de vigilance
   {points_vigilance}

6. Normes applicables
   {normes}

Besoin de précisions sur un point particulier ?
            """,

            "consultation_technique": """
Spécifications techniques pour {type_ouvrage} :

1. Matériaux
   {specs_materiaux}

2. Normes applicables
   {normes}

3. Exigences qualité
   {exigences_qualite}

4. Traitements
   {traitements}

5. Documents associés
   {documents}
            """,

            "clarification": """
Pour mieux vous aider, j'ai besoin de précisions sur les points suivants :

{points_clarification}

Questions spécifiques :
{questions_specifiques}

Références utiles :
{references}
            """
        }

    async def traiter_requete(
        self,
        requete: str,
        contexte_session: Optional[Dict] = None
    ) -> Dict:
        """Traite une requête métier."""
        try:
            # Analyse de la requête avec le contexte métier
            analyse = await self.analyseur.analyser_requete(requete, contexte_session)

            # Si clarification nécessaire
            if analyse.besoin_clarification:
                return await self._generer_reponse_clarification(analyse)

            # Traitement selon le type de demande
            if analyse.type_demande == "creation":
                return await self._traiter_creation(analyse)
            elif analyse.type_demande == "consultation":
                return await self._traiter_consultation(analyse)
            elif analyse.type_demande == "modification":
                return await self._traiter_modification(analyse)
            else:
                return await self._traiter_demande_generique(analyse)

        except Exception as e:
            logger.error(f"Erreur traitement requête: {e}")
            return self._generer_reponse_erreur()

    async def _traiter_creation(self, analyse: AnalyseMetier) -> Dict:
        """Traite une demande de création."""
        try:
            # Détermination du type d'ouvrage
            type_ouvrage = self._determiner_type_ouvrage(analyse)

            # Récupération des documents pertinents
            docs_techniques = await self._rechercher_documents_techniques(
                type_ouvrage=type_ouvrage,
                specifications=analyse.specifications
            )

            # Construction des spécifications techniques
            specs_techniques = self._formater_specifications_techniques(
                analyse.specifications,
                analyse.normes_applicables
            )

            # Préparation des points de vigilance
            points_vigilance = self._identifier_points_vigilance(
                type_ouvrage=type_ouvrage,
                specifications=analyse.specifications,
                contexte=analyse.contexte_supplementaire
            )

            # Formatage de la réponse
            return {
                "type_reponse": "creation",
                "contenu": self.templates["creation_devis"].format(
                    type_ouvrage=type_ouvrage,
                    specs_techniques=specs_techniques,
                    documents_requis=self._formater_documents_requis(type_ouvrage),
                    points_vigilance=points_vigilance,
                    normes=self._formater_normes(analyse.normes_applicables)
                ),
                "documents_techniques": docs_techniques,
                "confiance": analyse.confiance
            }

        except Exception as e:
            logger.error(f"Erreur traitement création: {e}")
            return self._generer_reponse_erreur()

    async def _traiter_consultation(self, analyse: AnalyseMetier) -> Dict:
        """Traite une demande de consultation."""
        try:
            # Recherche des documents pertinents
            docs_pertinents = await self._rechercher_documents_techniques(
                specifications=analyse.specifications,
                contexte=analyse.contexte_supplementaire
            )

            # Extraction des informations techniques
            specs = self._extraire_specifications_techniques(docs_pertinents)

            return {
                "type_reponse": "consultation",
                "contenu": self.templates["consultation_technique"].format(
                    type_ouvrage=specs.get("type_ouvrage", "non spécifié"),
                    specs_materiaux=self._formater_specifications_materiaux(specs),
                    normes=self._formater_normes(analyse.normes_applicables),
                    exigences_qualite=self._formater_exigences_qualite(specs),
                    traitements=self._formater_traitements(specs),
                    documents=self._formater_documents_associes(docs_pertinents)
                ),
                "documents_references": docs_pertinents,
                "confiance": analyse.confiance
            }

        except Exception as e:
            logger.error(f"Erreur traitement consultation: {e}")
            return self._generer_reponse_erreur()

    async def _rechercher_documents_techniques(
        self,
        type_ouvrage: Optional[str] = None,
        specifications: Optional[Dict] = None,
        contexte: Optional[Dict] = None
    ) -> List[Dict]:
        """Recherche les documents techniques pertinents."""
        try:
            # Construction du filtre de recherche
            filtre = {}
            if type_ouvrage:
                filtre["type_ouvrage"] = type_ouvrage
            if specifications:
                filtre.update(self._construire_filtre_specs(specifications))
            if contexte:
                filtre.update(self._construire_filtre_contexte(contexte))

            # Recherche dans Elasticsearch
            return await self.components.es_client.search_documents(
                metadata_filter=filtre,
                size=5
            )

        except Exception as e:
            logger.error(f"Erreur recherche documents: {e}")
            return []

    def _construire_filtre_specs(self, specifications: Dict) -> Dict:
        """Construit le filtre de recherche pour les spécifications."""
        filtre = {}
        if "materiaux" in specifications:
            filtre["materiaux.type"] = specifications["materiaux"][0]["type"]
        if "traitements" in specifications:
            filtre["traitement_surface"] = specifications["traitements"][0]
        return filtre

    def _construire_filtre_contexte(self, contexte: Dict) -> Dict:
        """Construit le filtre de recherche pour le contexte."""
        filtre = {}
        if "environnement" in contexte:
            filtre["environnement"] = contexte["environnement"][0]
        if "localisation" in contexte:
            filtre["localisation"] = contexte["localisation"][0]
        return filtre

    def _formater_specifications_techniques(
        self,
        specifications: Dict,
        normes: List[str]
    ) -> str:
        """Formate les spécifications techniques."""
        sections = []
        
        # Matériaux
        if "materiaux" in specifications:
            sections.append("Matériaux requis:")
            for materiau in specifications["materiaux"]:
                sections.append(f"- {materiau['type']} {materiau.get('valeur', '')}")

        # Dimensions
        if "dimensions" in specifications:
            sections.append("\nDimensions:")
            for dim in specifications["dimensions"]:
                sections.append(f"- {dim['valeur']} {dim['unite']}")

        # Traitements
        if "traitements" in specifications:
            sections.append("\nTraitements:")
            for traitement in specifications["traitements"]:
                sections.append(f"- {traitement}")

        # Assemblages
        if "assemblages" in specifications:
            sections.append("\nAssemblages:")
            for assemblage in specifications["assemblages"]:
                sections.append(f"- {assemblage}")

        return "\n".join(sections)

    def _formater_documents_requis(self, type_ouvrage: str) -> str:
        """Formate la liste des documents requis."""
        docs_base = [
            "Plans d'exécution",
            "Note de calcul",
            "Nomenclature matériaux"
        ]

        docs_specifiques = {
            "charpente": [
                "Plan de charpente",
                "Note de descente de charges",
                "Plan de contreventement"
            ],
            "escalier": [
                "Plan de calepinage",
                "Détails des assemblages",
                "Plan de fabrication"
            ],
            "garde_corps": [
                "Plan d'implantation",
                "Détails de fixation",
                "Note justificative"
            ]
        }

        docs = docs_base + docs_specifiques.get(type_ouvrage, [])
        return "\n".join(f"- {doc}" for doc in docs)

    def _generer_reponse_erreur(self) -> Dict:
        """Génère une réponse d'erreur standard."""
        return {
            "type_reponse": "erreur",
            "contenu": "Je suis désolé, une erreur est survenue lors du traitement de votre demande. "
                      "Pouvez-vous reformuler ou préciser votre question ?",
            "confiance": 0.0
        }

# Instance globale
processeur = ProcesseurConstructionMetallique()
