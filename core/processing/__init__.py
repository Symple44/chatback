# core/processing/__init__.py
from .base_processor import (
    BaseProcessor,
    ProcessingResult,
    ProcessingStage,
    ProcessingError
)
from .processors.pdf_processor import PDFProcessor

__all__ = [
    'BaseProcessor',
    'ProcessingResult',
    'ProcessingStage',
    'ProcessingError',
    'PDFProcessor'
]