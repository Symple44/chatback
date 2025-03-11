# core/document_processing/table_extraction/models.py
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
import time
import hashlib

class PDFType(Enum):
    """Types de documents PDF."""
    SCANNED = "scanned"   # Document scanné (image)
    DIGITAL = "digital"   # Document numérique (avec texte sélectionnable)
    HYBRID = "hybrid"     # Document mixte (certaines pages scannées, d'autres numériques)

@dataclass
class TableRegion:
    """Région d'un tableau détecté dans une image."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0

@dataclass
class ProcessedTable:
    """Tableau traité résultant d'une extraction."""
    data: Any                  # DataFrame pandas ou autre structure de données
    page: int                  # Numéro de page (1-based)
    rows: int                  # Nombre de lignes
    columns: int               # Nombre de colonnes
    method: str                # Méthode d'extraction utilisée
    confidence: float = 0.0    # Score de confiance (0-1)
    region: Optional[TableRegion] = None  # Région du tableau dans la page

@dataclass
class TableData:
    """Représentation d'un tableau avec métadonnées."""
    table_id: int
    page: int
    rows: int
    columns: int
    data: Any
    extraction_method: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ImageData:
    """Représentation d'une image encodée avec métadonnées."""
    data: str                # Image encodée en base64
    mime_type: str           # Type MIME
    width: Optional[int] = None
    height: Optional[int] = None
    page: Optional[int] = None
    table_id: Optional[int] = None

@dataclass
class TableExtractionContext:
    """Contexte d'extraction de tableaux."""
    file_path: str           # Chemin du fichier PDF
    filename: str            # Nom du fichier
    file_size: int           # Taille du fichier en octets
    pages: Union[str, List[int]]  # Pages à analyser
    output_format: str       # Format de sortie
    ocr_config: Dict[str, Any]  # Configuration OCR
    pdf_type: PDFType        # Type de PDF (scanné ou numérique)
    pdf_type_confidence: float  # Confiance dans la détection du type
    extraction_id: str       # Identifiant unique de l'extraction
    complexity_score: float = 0.0  # Score de complexité du PDF
    
    def get_page_number(self, idx: int) -> int:
        """Convertit un index de page interne en numéro de page."""
        if isinstance(self.pages, list) and idx < len(self.pages):
            return self.pages[idx] + 1  # Convertit 0-based en 1-based
        return idx + 1  # Par défaut, 1-based

class ExtractionResult:
    """Résultat complet d'une extraction de tableaux."""
    
    def __init__(
        self,
        extraction_id: str,
        filename: str,
        file_size: int,
        tables_count: int,
        processing_time: float,
        tables: List[Dict[str, Any]],
        extraction_method_used: str,
        pdf_type: str,
        ocr_used: bool = False,
        images: Optional[List[Dict[str, Any]]] = None,
        search_results: Optional[Dict[str, Any]] = None,
        analysis: Optional[Dict[str, Any]] = None,
        status: str = "completed",
        message: Optional[str] = None
    ):
        """
        Initialise le résultat d'extraction.
        
        Args:
            extraction_id: Identifiant unique de l'extraction
            filename: Nom du fichier
            file_size: Taille du fichier en octets
            tables_count: Nombre de tableaux extraits
            processing_time: Temps de traitement en secondes
            tables: Liste de tableaux extraits
            extraction_method_used: Méthode d'extraction utilisée
            pdf_type: Type de PDF (scanné/numérique)
            ocr_used: Si l'OCR a été utilisé
            images: Images des tableaux extraits
            search_results: Résultats de recherche dans les tableaux
            analysis: Analyse des tableaux
            status: Statut de l'extraction
            message: Message informatif
        """
        self.extraction_id = extraction_id
        self.filename = filename
        self.file_size = file_size
        self.tables_count = tables_count
        self.processing_time = processing_time
        self.tables = tables
        self.extraction_method_used = extraction_method_used
        self.pdf_type = pdf_type
        self.ocr_used = ocr_used
        self.images = images
        self.search_results = search_results
        self.analysis = analysis
        self.status = status
        self.message = message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le résultat en dictionnaire."""
        return {
            "extraction_id": self.extraction_id,
            "filename": self.filename,
            "file_size": self.file_size,
            "tables_count": self.tables_count,
            "processing_time": self.processing_time,
            "tables": self.tables,
            "extraction_method_used": self.extraction_method_used,
            "pdf_type": self.pdf_type,
            "ocr_used": self.ocr_used,
            "images": self.images,
            "search_results": self.search_results,
            "analysis": self.analysis,
            "status": self.status,
            "message": self.message
        }
    
    def to_json(self) -> str:
        """Convertit le résultat en JSON."""
        return json.dumps(self.to_dict(), cls=TableExtractionJSONEncoder)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ExtractionResult':
        """Crée un objet ExtractionResult à partir d'une chaîne JSON."""
        data = json.loads(json_str)
        return cls(**data)

class TableExtractionJSONEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé pour les objets d'extraction de tableaux."""
    
    def default(self, obj):
        import pandas as pd
        import numpy as np
        
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        
        return super().default(obj)