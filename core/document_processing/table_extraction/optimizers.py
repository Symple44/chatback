# core/document_processing/table_extraction/optimizers.py
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from core.utils.logger import get_logger
from .models import ProcessedTable, TableExtractionContext

logger = get_logger("table_optimizers")

class TableOptimizer:
    """
    Optimise les tableaux extraits pour améliorer leur qualité et leur lisibilité.
    """
    
    def __init__(self):
        """Initialise l'optimiseur de tableaux."""
        pass
    
    async def optimize_tables(
        self, 
        tables: List[ProcessedTable],
        context: TableExtractionContext
    ) -> List[ProcessedTable]:
        """
        Optimise une liste de tableaux extraits.
        
        Args:
            tables: Liste de tableaux à optimiser
            context: Contexte d'extraction
            
        Returns:
            Liste de tableaux optimisés
        """
        if not tables:
            return []
            
        optimized_tables = []
        
        for table in tables:
            # Optimiser chaque tableau individuellement
            optimized_table = await self._optimize_single_table(table)
            optimized_tables.append(optimized_table)
        
        # Optimisation des tableaux entre eux (fusion de tableaux fragmentés, etc.)
        if len(optimized_tables) > 1:
            optimized_tables = await self._optimize_table_relationships(optimized_tables)
        
        return optimized_tables
    
    async def _optimize_single_table(self, table: ProcessedTable) -> ProcessedTable:
        """
        Optimise un seul tableau.
        
        Args:
            table: Tableau à optimiser
            
        Returns:
            Tableau optimisé
        """
        df = table.data.copy() if isinstance(table.data, pd.DataFrame) else pd.DataFrame(table.data)
        
        # 1. Optimisation des types de données
        df = self._optimize_data_types(df)
        
        # 2. Optimisation des en-têtes
        df = self._optimize_headers(df)
        
        # 3. Nettoyage des valeurs
        df = self._clean_values(df)
        
        # 4. Identification et fusion des colonnes fragmentées
        df = self._merge_fragmented_columns(df)
        
        # Mettre à jour la structure
        return ProcessedTable(
            data=df,
            page=table.page,
            rows=len(df),
            columns=len(df.columns),
            method=table.method,
            confidence=table.confidence * 1.1,  # Légère augmentation de la confiance
            region=table.region
        )
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimise les types de données dans un DataFrame.
        
        Args:
            df: DataFrame à optimiser
            
        Returns:
            DataFrame avec types optimisés
        """
        # Pour chaque colonne
        for col in df.columns:
            # Vérifier si la colonne pourrait être numérique
            try:
                # Compter le nombre de valeurs qui peuvent être converties en nombre
                numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                num_values = len(df[col].dropna())
                
                # Si plus de 70% des valeurs sont numériques
                if num_values > 0 and numeric_count / num_values > 0.7:
                    # Essayer de détecter si c'est un entier ou un flottant
                    converted = pd.to_numeric(df[col], errors='coerce')
                    
                    # Vérifier si tous les chiffres sont des entiers
                    if (converted % 1 == 0).all():
                        df[col] = converted.astype('Int64')  # Int64 tolère les NaN
                    else:
                        df[col] = converted.astype('float')
                
                # Vérifier si c'est une date - ajouter formats courants
                elif df[col].dtype == object:
                    # Définir des formats de date courants
                    date_formats = [
                        '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', 
                        '%d.%m.%Y', '%Y.%m.%d'
                    ]
                    
                    # Essayer chaque format
                    for date_format in date_formats:
                        try:
                            # Utiliser un format spécifique pour éviter les avertissements
                            converted_dates = pd.to_datetime(df[col], format=date_format, errors='coerce')
                            date_count = converted_dates.notna().sum()
                            
                            if num_values > 0 and date_count / num_values > 0.7:
                                df[col] = converted_dates
                                break
                        except:
                            continue
                            
                    # Si aucun format n'a fonctionné et qu'on a pas encore converti
                    if df[col].dtype == object:
                        # Essayer sans format (en évitant l'avertissement)
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            converted_dates = pd.to_datetime(df[col], errors='coerce')
                            date_count = converted_dates.notna().sum()
                            
                            if num_values > 0 and date_count / num_values > 0.7:
                                df[col] = converted_dates
            
            except Exception as e:
                logger.debug(f"Erreur optimisation type colonne {col}: {e}")
        
        return df
    
    def _optimize_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimise les en-têtes de colonnes.
        
        Args:
            df: DataFrame à optimiser
            
        Returns:
            DataFrame avec en-têtes optimisés
        """
        # Nettoyage des en-têtes
        new_headers = []
        
        for col in df.columns:
            # Convertir en chaîne si ce n'est pas déjà le cas
            header = str(col).strip()
            
            # Remplacer les caractères non alphanumériques par des espaces
            header = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in header)
            
            # Nettoyer les espaces multiples
            header = ' '.join(header.split())
            
            # Si l'en-tête est vide, générer un nom générique
            if not header:
                header = f"Col_{len(new_headers)+1}"
                
            new_headers.append(header)
            
        # Gérer les doublons
        seen_headers = set()
        final_headers = []
        
        for header in new_headers:
            if header in seen_headers:
                counter = 1
                while f"{header}_{counter}" in seen_headers:
                    counter += 1
                header = f"{header}_{counter}"
            
            seen_headers.add(header)
            final_headers.append(header)
            
        # Appliquer les nouveaux en-têtes
        df.columns = final_headers
        return df
    
    def _clean_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les valeurs du DataFrame.
        
        Args:
            df: DataFrame à nettoyer
            
        Returns:
            DataFrame nettoyé
        """
        # Pour les colonnes de texte
        for col in df.select_dtypes(include=['object']):
            # Supprimer les espaces au début et à la fin
            if df[col].dtype == object:
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                
                # Remplacer les chaînes vides par NaN
                df[col] = df[col].replace('', np.nan)
                
                # Remplacer les valeurs non significatives
                non_significant = ['n/a', 'na', 'none', 'null', '-', '--']
                for val in non_significant:
                    df[col] = df[col].replace(val, np.nan)
        
        # Pour les colonnes numériques, arrondir pour plus de lisibilité
        for col in df.select_dtypes(include=[np.number]):
            # Vérifier s'il y a beaucoup de décimales
            decimals = df[col].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
            max_decimals = decimals.max()
            
            if max_decimals > 4:
                # Arrondir à 2 décimales pour plus de lisibilité
                df[col] = df[col].round(2)
        
        return df
    
    def _merge_fragmented_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Détecte et fusionne les colonnes qui pourraient être fragmentées.
        
        Args:
            df: DataFrame à optimiser
            
        Returns:
            DataFrame avec colonnes fusionnées si nécessaire
        """
        try:
            # Si peu de colonnes, pas besoin de fusionner
            if len(df.columns) <= 2:
                return df
                
            cols_to_merge = {}
            
            # Détecter les colonnes qui pourraient être des fragments
            for i, col1 in enumerate(df.columns[:-1]):
                for j, col2 in enumerate(df.columns[i+1:], i+1):
                    if self._should_merge_columns(df[col1], df[col2]):
                        if col1 not in cols_to_merge:
                            cols_to_merge[col1] = []
                        cols_to_merge[col1].append(col2)
            
            # Fusionner les colonnes
            for main_col, fragment_cols in cols_to_merge.items():
                # Vérifier que toutes les colonnes existent bien avant de fusionner
                valid_fragment_cols = [col for col in fragment_cols if col in df.columns]
                
                if not valid_fragment_cols:
                    continue
                    
                # Créer une copie du DataFrame pour éviter les avertissements de copie
                df = df.copy()
                
                # Fusionner les colonnes valides une par une
                for fragment_col in valid_fragment_cols:
                    try:
                        # Utiliser une fonction vectorisée au lieu de apply pour la performance
                        df[main_col] = df.apply(
                            lambda x: self._combine_values(x[main_col], x[fragment_col])
                            if fragment_col in x.index else x[main_col],  # Vérification supplémentaire
                            axis=1
                        )
                        
                        # Marquer la colonne fragment pour suppression
                        df = df.drop(columns=[fragment_col])
                    except KeyError as e:
                        # Log l'erreur et continuez
                        logger.warning(f"Erreur lors de la fusion des colonnes {main_col} et {fragment_col}: {e}")
                        continue
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur fusion colonnes: {e}")
            # En cas d'erreur, retourner le DataFrame original
            return df
    
    def _should_merge_columns(self, col1: pd.Series, col2: pd.Series) -> bool:
        """
        Détermine si deux colonnes devraient être fusionnées.
        
        Args:
            col1, col2: Colonnes à comparer
            
        Returns:
            True si les colonnes doivent être fusionnées
        """
        # Si les colonnes ont des types différents, ne pas fusionner
        if col1.dtype != col2.dtype:
            return False
            
        # Vérifier le modèle de non-NA
        na1 = col1.isna()
        na2 = col2.isna()
        
        # Si les deux colonnes ont beaucoup de NaN au même endroit, probablement pas des fragments
        if (na1 & na2).mean() > 0.5:
            return False
            
        # Si les colonnes sont complémentaires (NaN où l'autre a des valeurs)
        complementary = ((~na1 & na2) | (na1 & ~na2)).mean()
        if complementary > 0.7:
            return True
            
        return False
    
    def _combine_values(self, val1, val2):
        """Combine deux valeurs en priorisant les non-NA."""
        if pd.isna(val1):
            return val2
        if pd.isna(val2):
            return val1
        
        # Si les deux valeurs sont présentes, prendre la plus longue
        if isinstance(val1, str) and isinstance(val2, str):
            return val1 if len(val1) >= len(val2) else val2
            
        return val1  # Par défaut, prendre la première valeur
    
    async def _optimize_table_relationships(self, tables: List[ProcessedTable]) -> List[ProcessedTable]:
        """
        Optimise les relations entre différents tableaux.
        
        Args:
            tables: Liste de tableaux à optimiser
            
        Returns:
            Liste optimisée de tableaux
        """
        # Trier les tableaux par page
        tables.sort(key=lambda t: t.page)
        
        # Vérifier si des tableaux devraient être fusionnés
        merged_tables = []
        i = 0
        
        while i < len(tables):
            current = tables[i]
            
            # Chercher si le tableau suivant pourrait être une continuation
            if i + 1 < len(tables):
                next_table = tables[i + 1]
                
                # Si même page ou pages consécutives et structure similaire
                if ((current.page == next_table.page or current.page + 1 == next_table.page) and
                    current.columns == next_table.columns and
                    self._are_headers_similar(current.data, next_table.data)):
                    
                    # Fusion des tableaux
                    merged_data = pd.concat([current.data, next_table.data], ignore_index=True)
                    
                    # Utiliser la meilleure confiance des deux tableaux
                    confidence = max(current.confidence, next_table.confidence)
                    
                    # Créer le tableau fusionné
                    merged = ProcessedTable(
                        data=merged_data,
                        page=current.page,  # Utiliser la page du premier tableau
                        rows=len(merged_data),
                        columns=current.columns,
                        method=f"{current.method}-merged",
                        confidence=confidence,
                        region=current.region  # Garder la région du premier tableau
                    )
                    
                    merged_tables.append(merged)
                    i += 2  # Sauter les deux tableaux fusionnés
                    continue
            
            # Si pas de fusion, ajouter le tableau actuel
            merged_tables.append(current)
            i += 1
        
        return merged_tables
    
    def _are_headers_similar(self, df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        """
        Vérifie si deux DataFrames ont des en-têtes similaires.
        
        Args:
            df1, df2: DataFrames à comparer
            
        Returns:
            True si les en-têtes sont similaires
        """
        # Si le nombre de colonnes est différent, retourner False immédiatement
        if len(df1.columns) != len(df2.columns):
            return False
            
        # Comparer les en-têtes
        headers1 = [str(h).lower().strip() for h in df1.columns]
        headers2 = [str(h).lower().strip() for h in df2.columns]
        
        # Si les en-têtes sont identiques
        if headers1 == headers2:
            return True
            
        # Vérifier si les en-têtes sont similaires par similarité de chaîne
        similar_count = 0
        for h1, h2 in zip(headers1, headers2):
            # Calculer la similarité comme ratio de caractères communs
            common_chars = sum(1 for c in h1 if c in h2)
            max_len = max(len(h1), len(h2))
            
            if max_len > 0 and common_chars / max_len > 0.7:
                similar_count += 1
                
        # Si plus de 70% des en-têtes sont similaires
        return similar_count / len(headers1) > 0.7 if headers1 else False