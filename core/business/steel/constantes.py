# core/business/steel/constantes.py
"""Constantes métier pour la construction métallique en France."""

PROCESSUS_METIER = {
    "DEVIS": {
        "nom": "devis",
        "etats": [
            "brouillon",
            "en_attente",
            "validé",
            "refusé",
            "expiré"
        ],
        "documents": [
            "chiffrage",
            "avant_metré",
            "note_technique"
        ]
    },
    "COMMANDE": {
        "nom": "commande",
        "etats": [
            "brouillon",
            "confirmée",
            "en_production",
            "en_attente_matériaux",
            "en_fabrication",
            "prête",
            "livrée"
        ],
        "documents": [
            "bon_commande",
            "plan_exécution",
            "fiche_débit"
        ]
    },
    "FABRICATION": {
        "nom": "fabrication",
        "etats": [
            "planifiée",
            "en_cours",
            "contrôle_qualité",
            "terminée"
        ],
        "documents": [
            "gamme_fabrication",
            "plan_atelier",
            "fiche_autocontrôle"
        ]
    },
    "POSE": {
        "nom": "pose",
        "etats": [
            "à_planifier",
            "planifiée",
            "en_cours",
            "réceptionnée"
        ],
        "documents": [
            "plan_pose",
            "pv_réception",
            "dossier_sécurité"
        ]
    }
}

# Normes françaises et européennes applicables
NORMES = {
    "CONSTRUCTION": {
        "NF EN 1090-1": "Exécution des structures en acier",
        "NF EN 1090-2": "Exigences techniques construction acier",
        "NF EN 1993": "Eurocode 3 - Calcul des structures en acier",
        "DTU 32.1": "Construction métallique : Charpente en acier"
    },
    "MATERIAUX": {
        "NF EN 10025": "Produits laminés à chaud en aciers de construction",
        "NF EN 10219": "Profils creux de construction"
    },
    "SOUDAGE": {
        "NF EN ISO 3834": "Exigences de qualité en soudage",
        "NF EN ISO 9606-1": "Qualification soudeurs - Acier"
    }
}

# Matériaux et profils standards
MATERIAUX = {
    "ACIER_CONSTRUCTION": {
        "S235": {
            "description": "Acier de construction courant",
            "resistance": "235 MPa",
            "usage": "Construction générale"
        },
        "S275": {
            "description": "Acier de construction standard",
            "resistance": "275 MPa",
            "usage": "Charpente métallique"
        },
        "S355": {
            "description": "Acier de construction haute résistance",
            "resistance": "355 MPa",
            "usage": "Structures importantes"
        }
    },
    "PROFILES": {
        "IPE": ["80", "100", "120", "140", "160", "180", "200", "220", "240", "270", "300", "330", "360", "400", "450", "500", "550", "600"],
        "HEA": ["100", "120", "140", "160", "180", "200", "220", "240", "260", "280", "300", "320", "340", "360", "400", "450", "500", "550", "600"],
        "HEB": ["100", "120", "140", "160", "180", "200", "220", "240", "260", "280", "300", "320", "340", "360", "400", "450", "500", "550", "600"],
        "UPN": ["80", "100", "120", "140", "160", "180", "200", "220", "240", "260", "280", "300"]
    }
}

# Types de traitements de surface
TRAITEMENTS_SURFACE = {
    "GALVANISATION": {
        "description": "Protection par immersion à chaud",
        "normes": ["NF EN ISO 1461"],
        "durabilité": "40-50 ans"
    },
    "PEINTURE": {
        "description": "Protection par système de peinture",
        "normes": ["NF EN ISO 12944"],
        "types": {
            "C1": "Intérieur chauffé",
            "C2": "Intérieur non chauffé",
            "C3": "Extérieur urbain et industriel modéré",
            "C4": "Industriel et côtier",
            "C5": "Industriel agressif et marin"
        }
    },
    "METALLISATION": {
        "description": "Protection par projection thermique",
        "normes": ["NF EN ISO 2063"],
        "durabilité": "20-30 ans"
    }
}

# Tolérances et contrôles
TOLERANCES = {
    "FABRICATION": {
        "longueur": "±2mm jusqu'à 6m, ±3mm au-delà",
        "rectitude": "L/1000 (L en mm)",
        "équerrage": "±1° sur angle droit"
    },
    "MONTAGE": {
        "verticalité": "H/500 (H = hauteur en mm)",
        "niveau": "±10mm sur la longueur totale",
        "alignement": "±5mm entre éléments adjacents"
    }
}

# Documents réglementaires
DOCUMENTS_REGLEMENTAIRES = {
    "QUALITE": [
        "Plan d'Assurance Qualité (PAQ)",
        "Procédures de soudage (QMOS/DMOS)",
        "Certificats matériaux (CCPU)"
    ],
    "SECURITE": [
        "Plan Particulier de Sécurité (PPSPS)",
        "Document Unique (DUER)",
        "Habilitations personnel"
    ],
    "TECHNIQUE": [
        "Note de calculs",
        "Plans d'exécution",
        "Plans de fabrication",
        "DOE (Dossier des Ouvrages Exécutés)"
    ]
}
