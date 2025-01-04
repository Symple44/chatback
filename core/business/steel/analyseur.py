# core/business/steel/analyseur.py
import re
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from .constantes import (
    PROCESSUS_METIER, 
    MATERIAUX, 
    NORMES,
    TRAITEMENTS_SURFACE
)

@dataclass
class AnalyseMetier:
    """Résultat de l'analyse métier."""
    type_demande: str
    processus: Optional[str]
    materiaux: List[Dict]
    specifications: Dict
    normes_applicables: List[str]
    niveau_urgence: int
    contexte_supplementaire: Dict
    confiance: float
    besoin_clarification: bool
    points_clarification: List[str]

class AnalyseurConstructionMetallique:
    """Analyseur spécialisé pour la construction métallique."""

    def __init__(self):
        self._initialiser_patterns()

    def _initialiser_patterns(self):
        """Initialise les patterns de reconnaissance."""
        self.patterns_processus = {
            "creation": [
                r"(?:créer|nouveau|faire|réaliser|établir)",
                r"(?:nouvelle?|nouveau)\s+(?:devis|commande|projet)"
            ],
            "modification": [
                r"(?:modifier|changer|réviser|mettre à jour)",
                r"(?:modification|changement|révision)\s+(?:de|du|des)"
            ],
            "consultation": [
                r"(?:voir|consulter|afficher|montrer)",
                r"(?:où|comment)\s+(?:trouver|voir)"
            ],
            "validation": [
                r"(?:valider|approuver|confirmer)",
                r"(?:validation|approbation)\s+(?:de|du|des)"
            ]
        }

        self.patterns_specifications = {
            "profiles": r"(?:IPE|HEA|HEB|UPN|tube|cornière)\s*(\d+)?",
            "nuances": r"(?:S235|S275|S355)",
            "dimensions": r"(\d+(?:\.\d+)?)\s*(mm|cm|m)?",
            "traitements": r"(?:galvanisé|peint|thermolaqué)",
            "assemblages": r"(?:soudé|boulonné|vissé)"
        }

        self.patterns_contexte = {
            "localisation": r"(?:intérieur|extérieur|abrité|non\s+abrité)",
            "environnement": r"(?:marin|industriel|urbain|rural)",
            "contraintes": r"(?:accessibilité|hauteur|délai|budget)"
        }

    async def analyser_requete(
        self, 
        texte: str, 
        contexte: Optional[Dict] = None
    ) -> AnalyseMetier:
        """Analyse complète d'une requête avec gestion du contexte."""
        try:
            # Identification du type de demande et du processus
            type_demande, confiance_type = self._identifier_type_demande(texte)
            processus = self._identifier_processus(texte)

            # Extraction des spécifications techniques
            materiaux = self._extraire_materiaux(texte)
            specifications = self._extraire_specifications(texte)

            # Analyse du contexte métier
            contexte_metier = self._analyser_contexte_metier(texte, contexte)
            
            # Identification des normes applicables
            normes = self._identifier_normes(specifications, contexte_metier)

            # Évaluation du niveau d'urgence
            niveau_urgence = self._evaluer_urgence(texte)

            # Vérification du besoin de clarification
            besoin_clarification, points_clarification = self._verifier_besoin_clarification(
                type_demande=type_demande,
                specifications=specifications,
                contexte_metier=contexte_metier
            )

            # Calcul du score de confiance global
            confiance = self._calculer_confiance(
                confiance_type=confiance_type,
                specs_completes=bool(specifications),
                contexte_clair=bool(contexte_metier),
                processus_identifie=bool(processus)
            )

            return AnalyseMetier(
                type_demande=type_demande,
                processus=processus,
                materiaux=materiaux,
                specifications=specifications,
                normes_applicables=normes,
                niveau_urgence=niveau_urgence,
                contexte_supplementaire=contexte_metier,
                confiance=confiance,
                besoin_clarification=besoin_clarification,
                points_clarification=points_clarification
            )

        except Exception as e:
            logger.error(f"Erreur analyse requête: {e}")
            return self._generer_analyse_fallback()

    def _identifier_type_demande(self, texte: str) -> Tuple[str, float]:
        """Identifie le type de demande et retourne le niveau de confiance."""
        texte_lower = texte.lower()
        scores = {
            "creation": 0,
            "modification": 0,
            "consultation": 0,
            "validation": 0
        }

        # Analyse des patterns pour chaque type
        for type_demande, patterns in self.patterns_processus.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, texte_lower))
                scores[type_demande] += matches * 0.3  # Pondération des matches

        # Identification du type principal
        type_max = max(scores.items(), key=lambda x: x[1])
        confiance = min(type_max[1], 1.0)  # Normalisation entre 0 et 1

        return type_max[0], confiance

    def _extraire_materiaux(self, texte: str) -> List[Dict]:
        """Extrait les informations sur les matériaux."""
        materiaux = []
        texte_lower = texte.lower()

        # Recherche des profils
        for match in re.finditer(self.patterns_specifications["profiles"], texte_lower):
            profil = match.group(0)
            dimension = match.group(1) if match.groups() else None
            materiaux.append({
                "type": "profil",
                "valeur": profil,
                "dimension": dimension
            })

        # Recherche des nuances
        for match in re.finditer(self.patterns_specifications["nuances"], texte_lower):
            materiaux.append({
                "type": "nuance",
                "valeur": match.group(0)
            })

        return materiaux

    def _extraire_specifications(self, texte: str) -> Dict:
        """Extrait toutes les spécifications techniques."""
        specs = {}
        texte_lower = texte.lower()

        # Extraction des dimensions
        dimensions = re.finditer(self.patterns_specifications["dimensions"], texte_lower)
        if dimensions:
            specs["dimensions"] = [
                {
                    "valeur": match.group(1),
                    "unite": match.group(2) or "mm"
                } for match in dimensions
            ]

        # Extraction des traitements
        traitements = re.findall(self.patterns_specifications["traitements"], texte_lower)
        if traitements:
            specs["traitements"] = traitements

        # Extraction des types d'assemblages
        assemblages = re.findall(self.patterns_specifications["assemblages"], texte_lower)
        if assemblages:
            specs["assemblages"] = assemblages

        return specs

    def _analyser_contexte_metier(
        self, 
        texte: str, 
        contexte_existant: Optional[Dict]
    ) -> Dict:
        """Analyse le contexte métier complet."""
        contexte = contexte_existant or {}
        texte_lower = texte.lower()

        # Analyse de la localisation
        for match in re.finditer(self.patterns_contexte["localisation"], texte_lower):
            contexte.setdefault("localisation", []).append(match.group(0))

        # Analyse de l'environnement
        for match in re.finditer(self.patterns_contexte["environnement"], texte_lower):
            contexte.setdefault("environnement", []).append(match.group(0))

        # Analyse des contraintes
        for match in re.finditer(self.patterns_contexte["contraintes"], texte_lower):
            contexte.setdefault("contraintes", []).append(match.group(0))

        return contexte

    def _identifier_normes(
        self, 
        specifications: Dict,
        contexte: Dict
    ) -> List[str]:
        """Identifie les normes applicables selon le contexte."""
        normes_applicables = []

        # Normes de construction de base
        normes_applicables.append("NF EN 1090-2")

        # Normes selon l'environnement
        if contexte.get("environnement"):
            if "marin" in contexte["environnement"]:
                normes_applicables.append("NF EN ISO 12944-2")
            if "industriel" in contexte["environnement"]:
                normes_applicables.append("NF EN ISO 14713")

        # Normes selon les traitements
        if specifications.get("traitements"):
            if "galvanisé" in specifications["traitements"]:
                normes_applicables.append("NF EN ISO 1461")
            if "peint" in specifications["traitements"]:
                normes_applicables.append("NF EN ISO 12944")

        return normes_applicables

    def _verifier_besoin_clarification(
        self,
        type_demande: str,
        specifications: Dict,
        contexte_metier: Dict
    ) -> Tuple[bool, List[str]]:
        """Vérifie si des clarifications sont nécessaires."""
        points_clarification = []

        # Vérification des informations manquantes essentielles
        if not specifications.get("dimensions"):
            points_clarification.append("Dimensions requises")
            
        if type_demande == "creation" and not specifications.get("materiaux"):
            points_clarification.append("Type de matériau à préciser")

        if not contexte_metier.get("localisation"):
            points_clarification.append("Localisation de l'ouvrage (intérieur/extérieur)")

        # Pour les demandes de création, vérifications supplémentaires
        if type_demande == "creation":
            if not specifications.get("traitements"):
                points_clarification.append("Traitement de surface à définir")
            if not specifications.get("assemblages"):
                points_clarification.append("Type d'assemblage à préciser")

        return bool(points_clarification), points_clarification

    def _calculer_confiance(
        self,
        confiance_type: float,
        specs_completes: bool,
        contexte_clair: bool,
        processus_identifie: bool
    ) -> float:
        """Calcule le score de confiance global."""
        poids = {
            "type_demande": 0.4,
            "specifications": 0.3,
            "contexte": 0.2,
            "processus": 0.1
        }

        scores = {
            "type_demande": confiance_type,
            "specifications": 1.0 if specs_completes else 0.5,
            "contexte": 1.0 if contexte_clair else 0.5,
            "processus": 1.0 if processus_identifie else 0.0
        }

        return sum(poids[k] * scores[k] for k in poids)

    def _generer_analyse_fallback(self) -> AnalyseMetier:
        """Génère une analyse par défaut en cas d'erreur."""
        return AnalyseMetier(
            type_demande="inconnu",
            processus=None,
            materiaux=[],
            specifications={},
            normes_applicables=[],
            niveau_urgence=3,
            contexte_supplementaire={},
            confiance=0.0,
            besoin_clarification=True,
            points_clarification=["Impossible d'analyser la requête. Merci de reformuler."]
        )
