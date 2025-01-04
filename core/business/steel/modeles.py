# core/business/steel/modeles.py
"""Modèles de domaine pour la construction métallique."""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from decimal import Decimal
from enum import Enum

class TypeOuvrage(Enum):
    """Types d'ouvrages standards."""
    CHARPENTE = "charpente"
    BARDAGE = "bardage"
    SERRURERIE = "serrurerie"
    METALLERIE = "metallerie"
    ESCALIER = "escalier"
    GARDE_CORPS = "garde_corps"

class StatutDocument(Enum):
    """États possibles des documents."""
    BROUILLON = "brouillon"
    EN_COURS = "en_cours"
    A_VALIDER = "a_valider"
    VALIDE = "valide"
    REFUSE = "refuse"
    ARCHIVE = "archive"

@dataclass
class SpecificationMatiere:
    """Spécification des matériaux."""
    nuance: str  # S235, S275, S355
    type_profil: Optional[str]  # IPE, HEA, etc.
    dimensions: Dict[str, Decimal]  # hauteur, largeur, épaisseur
    traitement_surface: Optional[str]
    quantite: Decimal
    unite: str  # ML, KG, M2
    observations: Optional[str] = None

@dataclass
class SpecificationTechnique:
    """Spécifications techniques complètes."""
    materiaux: List[SpecificationMatiere]
    normes_applicables: List[str]
    exigences_qualite: Dict[str, str]
    traitements: List[str]
    contraintes_fabrication: Optional[Dict[str, str]] = None
    contraintes_pose: Optional[Dict[str, str]] = None

@dataclass
class Client:
    """Informations client."""
    id: str
    raison_sociale: str
    siret: str
    adresse: Dict[str, str]
    contact: Dict[str, str]
    categorie: str  # particulier, professionnel, public
    conditions_paiement: Optional[str] = None

@dataclass
class Devis:
    """Devis construction métallique."""
    numero: str
    date_creation: datetime
    client: Client
    type_ouvrage: TypeOuvrage
    specifications: SpecificationTechnique
    delai_execution: int  # en jours
    conditions_particulieres: Optional[str]
    montant_ht: Decimal
    tva: Decimal
    statut: StatutDocument
    validite: int  # en jours
    documents_lies: List[str]  # références aux documents associés
    historique_modifications: List[Dict]

@dataclass
class Commande:
    """Commande construction métallique."""
    numero: str
    date_creation: datetime
    devis_reference: Optional[str]
    client: Client
    specifications: SpecificationTechnique
    date_livraison_souhaitee: datetime
    priorite: int  # 1 (urgente) à 5 (normale)
    statut: str
    commentaires: Optional[str]
    documents_techniques: List[str]
    suivi_fabrication: Optional[Dict] = None
    suivi_pose: Optional[Dict] = None

@dataclass
class PlanificationFabrication:
    """Planning de fabrication."""
    commande_id: str
    debut_prevu: datetime
    fin_prevue: datetime
    ressources_necessaires: Dict[str, int]
    etapes: List[Dict[str, Any]]
    contraintes: List[str]
    statut: str

@dataclass
class ControleFabrication:
    """Contrôle qualité fabrication."""
    commande_id: str
    date_controle: datetime
    controleur: str
    points_controle: List[Dict[str, Any]]
    conformite: bool
    observations: Optional[str]
    actions_correctives: Optional[List[str]]

@dataclass
class DossierTechnique:
    """Dossier technique complet."""
    reference: str
    date_creation: datetime
    type_ouvrage: TypeOuvrage
    documents: Dict[str, str]  # type: chemin
    validations: List[Dict[str, Any]]
    statut: StatutDocument
