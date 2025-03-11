# core/document_processing/table_extraction/validators.py
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from core.utils.logger import get_logger
from .models import ProcessedTable, TableExtractionContext
from core.config.config import settings

logger = get_logger("table_validators")

class TableValidator:
    """
    Valide et corrige les tableaux extraits pour s'assurer de leur qualité.
    """
    
    def __init__(self):
        """Initialise le validateur de tableaux."""
        self.min_rows = settings.table_extraction.VALIDATION.MIN_ROWS
        self.min_columns = settings.table_extraction.VALIDATION.MIN_COLUMNS
        self.max_empty_ratio = settings.table_extraction.VALIDATION.MAX_EMPTY_RATIO
        self.max_error_ratio = settings.table_extraction.VALIDATION.MAX_ERROR_RATIO
    
    async def validate_tables(
        self, 
        tables: List[ProcessedTable],
        context: TableExtractionContext
    ) -> List[ProcessedTable]:
        """
        Valide une liste de tableaux extraits.
        
        Args:
            tables: Liste de tableaux à valider
            context: Contexte d'extraction
            
        Returns:
            Liste de tableaux validés
        """
        if not tables:
            return []
            
        validated_tables = []
        
        for table in tables:
            # Vérifier d'abord si le tableau est valide structurellement
            if not self._is_valid_structure(table):
                logger.debug(f"Tableau page {table.page} invalide structurellement")
                continue
                
            # Appliquer les corrections
            corrected_table = self._fix_table_errors(table)
            
            # Vérifier si le tableau reste valide après corrections
            if self._is_valid_content(corrected_table):
                validated_tables.append(corrected_table)
            else:
                logger.debug(f"Tableau page {table.page} contenu invalide après corrections")
        
        return validated_tables
    
    def _is_valid_structure(self, table: ProcessedTable) -> bool:
        """
        Vérifie si un tableau a une structure valide.
        
        Args:
            table: Tableau à vérifier
            
        Returns:
            True si la structure est valide
        """
        # Vérifier le nombre de lignes et colonnes
        if table.rows < self.min_rows or table.columns < self.min_columns:
            return False
            
        # Vérifier que les données sont bien un DataFrame ou convertibles
        if not isinstance(table.data, pd.DataFrame):
            try:
                if isinstance(table.data, (list, dict)):
                    # Tenter de convertir en DataFrame
                    _ = pd.DataFrame(table.data)
                else:
                    return False
            except Exception:
                return False
        
        return True
    
    def _is_valid_content(self, table: ProcessedTable) -> bool:
        """
        Vérifie si le contenu d'un tableau est valide.
        
        Args:
            table: Tableau à vérifier
            
        Returns:
            True si le contenu est valide
        """
        df = table.data
        
        # S'assurer que c'est un DataFrame
        if not isinstance(df, pd.DataFrame):
            return False
        
        # Vérifier si le tableau est vide
        if df.empty:
            return False
        
        # Vérifier le ratio de cellules vides
        empty_cells = df.isna().sum().sum()
        total_cells = df.size
        empty_ratio = empty_cells / total_cells if total_cells > 0 else 1.0
        
        if empty_ratio > self.max_empty_ratio:
            return False
        
        # Vérifier les erreurs potentielles
        error_count = self._count_error_cells(df)
        error_ratio = error_count / total_cells if total_cells > 0 else 0.0
        
        if error_ratio > self.max_error_ratio:
            return False
        
        return True
    
    def _count_error_cells(self, df: pd.DataFrame) -> int:
        """
        Compte les cellules avec erreurs potentielles.
        
        Args:
            df: DataFrame à analyser
            
        Returns:
            Nombre de cellules avec erreurs
        """
        error_count = 0
        
        # Convertir en chaînes pour vérification
        df_str = df.astype(str)
        
        # Vérifier les cellules avec caractères d'erreur courants
        error_markers = ['#', '?', '!', 'error', 'err', 'null', 'none']
        
        for col in df_str.columns:
            for marker in error_markers:
                error_count += df_str[col].str.contains(marker, case=False).sum()
        
        # Vérifier les valeurs incohérentes
        for col in df.columns:
            col_values = df[col]
            
            # Si la colonne semble numérique mais contient des non-numériques
            if pd.to_numeric(col_values, errors='coerce').notna().mean() > 0.7:
                error_count += col_values.apply(
                    lambda x: 1 if not isinstance(x, (int, float)) and not pd.isna(x) else 0
                ).sum()
        
        return error_count
    
    def _fix_table_errors(self, table: ProcessedTable) -> ProcessedTable:
        """
        Corrige les erreurs courantes dans un tableau.
        
        Args:
            table: Tableau à corriger
            
        Returns:
            Tableau corrigé
        """
        df = table.data.copy() if isinstance(table.data, pd.DataFrame) else pd.DataFrame(table.data)
        
        # 1. Nettoyage des noms de colonnes
        if any(pd.isna(col) for col in df.columns):
            df.columns = [f"Col_{i+1}" if pd.isna(col) else col for i, col in enumerate(df.columns)]
        
        # 2. Correction des valeurs erronnées
        for col in df.columns:
            # Remplacer les chaînes d'erreur par NaN
            error_values = ['#VALUE!', '#REF!', '#DIV/0!', '#ERROR!', 'error', 'null', 'none', 'nan']
            for err_val in error_values:
                df[col] = df[col].replace(err_val, np.nan)
            
            # Correction des types de données
            numeric_ratio = pd.to_numeric(df[col], errors='coerce').notna().mean()
            if numeric_ratio > 0.7:  # Colonne principalement numérique
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. Suppression des lignes dupliquées
        df = df.drop_duplicates()
        
        # 4. Correction des valeurs extrêmes
        for col in df.select_dtypes(include=[np.number]).columns:
            # Remplacer les valeurs extrêmes (outliers) par NaN
            q1 = df[col].quantile(0.1)
            q3 = df[col].quantile(0.9)
            iqr = q3 - q1
            lower_bound = q1 - (iqr * 3)
            upper_bound = q3 + (iqr * 3)
            
            # Remplacer les valeurs extrêmes
            df[col] = df[col].apply(
                lambda x: x if (lower_bound <= x <= upper_bound) or pd.isna(x) else np.nan
            )
        
        # Mettre à jour la structure
        return ProcessedTable(
            data=df,
            page=table.page,
            rows=len(df),
            columns=len(df.columns),
            method=table.method,
            confidence=table.confidence,
            region=table.region
        )