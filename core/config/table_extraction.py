# core/config/table_extraction.py
import os
from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict, List, Optional, Union

class TableExtractionStrategy(str, Enum):
    """Stratégies d'extraction de tableaux disponibles."""
    AUTO = "auto"
    CAMELOT = "camelot"
    TABULA = "tabula"
    PDF_PLUMBER = "pdfplumber"
    OCR = "ocr"
    AI = "ai"
    HYBRID = "hybrid"
    ENHANSED = "enhanced"

class OCRConfig(BaseModel):
    """Configuration pour l'OCR des tableaux."""
    TESSERACT_CMD: str = Field(default_factory=lambda: os.getenv("TESSERACT_CMD", "/usr/bin/tesseract"))
    TESSERACT_LANG: str = Field(default_factory=lambda: os.getenv("TESSERACT_LANG", "fra+eng"))
    OCR_DPI: int = Field(default_factory=lambda: int(os.getenv("OCR_DPI", "300")))
    ENHANCE_IMAGE: bool = True
    DESKEW: bool = True
    PREPROCESS_TYPE: str = "thresh"  # Options: thresh, adaptive, blur
    PSM: int = 6  # Page segmentation mode pour Tesseract

class AIDetectionConfig(BaseModel):
    """Configuration pour la détection IA des tableaux."""
    ENABLED: bool = Field(default_factory=lambda: os.getenv("ENABLE_AI_TABLE_DETECTION", "true").lower() == "true")
    MODEL: str = Field(default_factory=lambda: os.getenv("TABLE_DETECTION_MODEL", "microsoft/table-transformer-detection"))
    CONFIDENCE_THRESHOLD: float = Field(default_factory=lambda: float(os.getenv("TABLE_DETECTION_THRESHOLD", "0.7")))
    MAX_TABLES: int = Field(default_factory=lambda: int(os.getenv("TABLE_DETECTION_MAX_TABLES", "10")))

class HybridStrategyConfig(BaseModel):
    """Configuration pour la stratégie hybride."""
    TABULA_WEIGHT: float = 1.0
    CAMELOT_WEIGHT: float = 0.9
    OCR_WEIGHT: float = 0.7
    AI_WEIGHT: float = 1.1
    COMBINE_METHOD: str = "weighted_sum"  # Options: weighted_sum, max_score, voting

class ValidationConfig(BaseModel):
    """Configuration pour la validation des tableaux."""
    MIN_ROWS: int = 2
    MIN_COLUMNS: int = 2
    MAX_EMPTY_RATIO: float = 0.7
    MAX_ERROR_RATIO: float = 0.3

class TableExtractionConfig(BaseModel):
    """Configuration complète pour l'extraction de tableaux."""
    DEFAULT_STRATEGY: TableExtractionStrategy = TableExtractionStrategy.AUTO
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # Secondes
    MAX_TABLES: int = 20
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 Mo
    OCR: OCRConfig = OCRConfig()
    AI_DETECTION: AIDetectionConfig = AIDetectionConfig()
    HYBRID: HybridStrategyConfig = HybridStrategyConfig()
    VALIDATION: ValidationConfig = ValidationConfig()
    SUPPORTED_OUTPUT_FORMATS: List[str] = ["json", "pandas", "csv", "html", "excel"]
    SUPPORTED_DOWNLOAD_FORMATS: List[str] = ["csv", "excel", "json", "html"]