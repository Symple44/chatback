# core/document_processing/table_extraction/exporter.py
from typing import List, Dict, Any, Optional, Union, BinaryIO, Tuple
import pandas as pd
import numpy as np
import os
import io
import json
import asyncio
from pathlib import Path
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
import time
import hashlib
from datetime import datetime

from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.config import settings

logger = get_logger("table_exporter")

class TableExporter:
    """
    Classe spécialisée pour l'exportation de tableaux extraits de PDFs dans différents formats.
    
    Cette classe supporte les formats d'exportation:
    - Excel (.xlsx): Un classeur avec plusieurs feuilles, une par tableau
    - CSV (.zip): Un fichier ZIP contenant plusieurs fichiers CSV, un par tableau
    - JSON (.json): Un fichier JSON contenant tous les tableaux
    - HTML (.html): Une page HTML avec tous les tableaux formatés
    - PDF (.pdf): Export des tableaux en PDF (permet de conserver la mise en forme exacte)
    - Markdown (.md): Export en format Markdown pour intégration dans des documents ou wikis
    """
    
    SUPPORTED_FORMATS = ["excel", "csv", "json", "html", "pdf", "markdown"]
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialise l'exportateur de tableaux.
        
        Args:
            temp_dir: Répertoire temporaire pour les fichiers d'export
        """
        self.temp_dir = temp_dir or os.path.join("temp", "pdf_exports")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
    
    async def export_tables(
        self, 
        tables: List[Dict[str, Any]], 
        format: str, 
        filename: str, 
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Exporte les tableaux dans le format spécifié.
        
        Args:
            tables: Liste de tableaux à exporter
            format: Format d'export ('excel', 'csv', 'json', 'html', 'pdf', 'markdown')
            filename: Nom du fichier original
            include_metadata: Inclure les métadonnées dans l'export
            
        Returns:
            Informations sur le fichier exporté
        """
        try:
            with metrics.timer(f"export_tables_{format}"):
                if not tables:
                    raise ValueError("Aucun tableau à exporter")
                
                format = format.lower()
                if format not in self.SUPPORTED_FORMATS:
                    raise ValueError(f"Format non supporté: {format}. Formats supportés: {', '.join(self.SUPPORTED_FORMATS)}")
                
                # Préparation du nom de fichier d'export
                base_filename = os.path.splitext(os.path.basename(filename))[0]
                export_filename = f"{base_filename}_tables"
                
                # Conversion des tableaux en DataFrames
                dataframes = await self._prepare_dataframes(tables, include_metadata)
                
                # Appeler la méthode d'export appropriée en fonction du format
                if format == "excel":
                    return await self._export_to_excel(dataframes, export_filename)
                elif format == "csv":
                    return await self._export_to_csv(dataframes, export_filename)
                elif format == "json":
                    return await self._export_to_json(dataframes, export_filename)
                elif format == "html":
                    return await self._export_to_html(dataframes, export_filename)
                elif format == "pdf":
                    return await self._export_to_pdf(dataframes, export_filename)
                elif format == "markdown":
                    return await self._export_to_markdown(dataframes, export_filename)
                
        except Exception as e:
            logger.error(f"Erreur export tableaux: {e}")
            raise
    
    async def _prepare_dataframes(
        self, 
        tables: List[Dict[str, Any]], 
        include_metadata: bool = True
    ) -> List[pd.DataFrame]:
        """
        Convertit les tableaux en DataFrames Pandas optimisés.
        
        Args:
            tables: Liste de tableaux à convertir
            include_metadata: Inclure les métadonnées dans les DataFrames
            
        Returns:
            Liste de DataFrames
        """
        dataframes = []
        
        # Conversion asynchrone et parallèle des tableaux en DataFrames
        conversion_tasks = []
        for table in tables:
            task = asyncio.create_task(self._convert_table_to_dataframe(table, include_metadata))
            conversion_tasks.append(task)
        
        # Attendre toutes les conversions
        results = await asyncio.gather(*conversion_tasks, return_exceptions=True)
        
        # Traiter les résultats et filtrer les erreurs
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Erreur conversion tableau {i+1}: {result}")
            else:
                dataframes.append(result)
                
        return dataframes
    
    async def _convert_table_to_dataframe(
        self, 
        table: Dict[str, Any], 
        include_metadata: bool = True
    ) -> pd.DataFrame:
        """
        Convertit un tableau en DataFrame Pandas.
        
        Args:
            table: Tableau à convertir
            include_metadata: Inclure les métadonnées
            
        Returns:
            DataFrame optimisé
        """
        try:
            # Extraire les données du tableau
            data = table.get("data")
            
            # Si les données sont déjà un DataFrame
            if isinstance(data, pd.DataFrame):
                df = data.copy()
            # Si les données sont une liste de dictionnaires
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
            # Si les données sont une chaîne de caractères (CSV, JSON ou HTML)
            elif isinstance(data, str):
                # Essayer de détecter le format et parser
                if data.startswith("<table") or "<tr>" in data:
                    # Format HTML
                    df = pd.read_html(data)[0]
                elif data.startswith("[") or data.startswith("{"):
                    # Format JSON
                    try:
                        df = pd.read_json(io.StringIO(data))
                    except:
                        # Essayer d'autres formats JSON
                        json_data = json.loads(data)
                        if isinstance(json_data, list):
                            df = pd.DataFrame(json_data)
                        else:
                            df = pd.DataFrame([json_data])
                else:
                    # Supposer que c'est du CSV
                    df = pd.read_csv(io.StringIO(data))
            else:
                # Format inconnu, créer un DataFrame vide
                df = pd.DataFrame()
            
            # Nettoyer et optimiser le DataFrame
            # 1. Supprimer les lignes et colonnes vides
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # 2. Optimiser les types de données pour réduire la consommation de mémoire
            for col in df.columns:
                # Conversion des colonnes numériques
                if pd.to_numeric(df[col], errors='coerce').notna().all():
                    if (df[col] % 1 == 0).all():  # Si toutes les valeurs sont des entiers
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    else:
                        df[col] = pd.to_numeric(df[col], downcast='float')
                # Optimisation des colonnes de texte avec catégories
                elif df[col].dtype == 'object' and df[col].nunique() < len(df[col]) * 0.5:
                    df[col] = df[col].astype('category')
            
            # 3. Ajouter les métadonnées si demandé
            if include_metadata:
                # Ajouter la page et l'ID du tableau comme colonnes si disponibles
                if "page" in table:
                    df["_page"] = table["page"]
                if "table_id" in table:
                    df["_table_id"] = table["table_id"]
            
            return df
        
        except Exception as e:
            logger.error(f"Erreur conversion tableau: {e}")
            # Retourner un DataFrame minimal avec les données brutes
            if "data" in table and isinstance(table["data"], list):
                return pd.DataFrame(table["data"])
            return pd.DataFrame()
    
    async def _export_to_excel(self, dataframes: List[pd.DataFrame], export_filename: str) -> Dict[str, Any]:
        """
        Exporte les DataFrames en fichier Excel.
        
        Args:
            dataframes: Liste de DataFrames à exporter
            export_filename: Nom du fichier d'export (sans extension)
            
        Returns:
            Informations sur le fichier exporté
        """
        try:
            # Créer un fichier Excel en mémoire
            output = io.BytesIO()
            
            # Utiliser un thread pour ne pas bloquer la boucle asyncio
            loop = asyncio.get_event_loop()
            
            def create_excel():
                with pd.ExcelWriter(output, engine='xlsxwriter', engine_kwargs={'options': {'constant_memory': True}}) as writer:
                    for i, df in enumerate(dataframes):
                        # Remplacer les NaN par None pour éviter les problèmes d'export
                        df = df.replace({pd.NA: None, np.nan: None})
                        
                        # Nom de la feuille
                        sheet_name = f"Table_{i+1}"
                        if len(sheet_name) > 31:  # Excel limite à 31 caractères
                            sheet_name = sheet_name[:31]
                        
                        # Écrire le DataFrame dans la feuille
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Obtenir l'objet worksheet
                        worksheet = writer.sheets[sheet_name]
                        
                        # Format des en-têtes (gras, fond gris clair)
                        header_format = writer.book.add_format({
                            'bold': True,
                            'bg_color': '#D3D3D3',
                            'border': 1
                        })
                        
                        # Appliquer le format aux en-têtes
                        for col_num, value in enumerate(df.columns.values):
                            worksheet.write(0, col_num, value, header_format)
                        
                        # Ajuster automatiquement la largeur des colonnes
                        for col_num, (col_name, col_data) in enumerate(df.items()):
                            max_len = max(
                                col_data.astype(str).map(len).max(),
                                len(str(col_name))
                            ) + 2  # Ajouter une marge
                            worksheet.set_column(col_num, col_num, min(max_len, 50))  # Limiter à 50 caractères max
                        
                        # Ajouter une table Excel avec filtres automatiques
                        worksheet.add_table(0, 0, len(df), len(df.columns) - 1, {
                            'header_row': True,
                            'style': 'Table Style Medium 2',
                            'columns': [{'header': col} for col in df.columns]
                        })
            
            # Exécuter la création d'Excel dans un thread
            await loop.run_in_executor(self.executor, create_excel)
            
            # Finaliser le fichier
            output.seek(0)
            filename = f"{export_filename}.xlsx"
            export_path = os.path.join(self.temp_dir, filename)
            
            # Écrire le fichier sur disque
            with open(export_path, "wb") as f:
                f.write(output.getvalue())
            
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            return {
                "content": output.getvalue(),
                "filename": filename,
                "media_type": media_type,
                "file_size": len(output.getvalue()),
                "path": export_path,
                "download_url": f"/api/pdf/tables/download/{filename}"
            }
            
        except Exception as e:
            logger.error(f"Erreur export Excel: {e}")
            raise
    
    async def _export_to_csv(self, dataframes: List[pd.DataFrame], export_filename: str) -> Dict[str, Any]:
        """
        Exporte les DataFrames en fichiers CSV compressés dans un ZIP.
        
        Args:
            dataframes: Liste de DataFrames à exporter
            export_filename: Nom du fichier d'export (sans extension)
            
        Returns:
            Informations sur le fichier exporté
        """
        try:
            # Créer un fichier ZIP en mémoire
            output = io.BytesIO()
            
            # Utiliser un thread pour ne pas bloquer la boucle asyncio
            loop = asyncio.get_event_loop()
            
            def create_zip():
                with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for i, df in enumerate(dataframes):
                        # CORRECTION: Gestion spécifique des colonnes catégorielles
                        # Convertir les types catégoriels en str pour éviter les erreurs
                        df_copy = df.copy()
                        
                        # Identifier et convertir les colonnes catégorielles
                        categorical_columns = df_copy.select_dtypes(include=['category']).columns
                        for col in categorical_columns:
                            df_copy[col] = df_copy[col].astype(str)
                        
                        # Remplacer les NaN par des chaînes vides pour CSV
                        df_copy = df_copy.fillna('')
                        
                        # Créer le CSV en mémoire
                        csv_buffer = io.StringIO()
                        df_copy.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        
                        # Ajouter au ZIP
                        filename = f"table_{i+1}.csv"
                        zipf.writestr(filename, csv_buffer.getvalue())
                    
                    # Ajouter un fichier README.txt avec des informations
                    readme = f"""Tables exportées: {len(dataframes)}
    Date d'exportation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Format: CSV
    Encodage: UTF-8
    Séparateur: ,
    """
                    zipf.writestr("README.txt", readme)
            
            # Exécuter la création du ZIP dans un thread
            await loop.run_in_executor(self.executor, create_zip)
            
            # Finaliser le fichier
            output.seek(0)
            filename = f"{export_filename}.zip"
            export_path = os.path.join(self.temp_dir, filename)
            
            # Écrire le fichier sur disque
            with open(export_path, "wb") as f:
                f.write(output.getvalue())
            
            return {
                "content": output.getvalue(),
                "filename": filename,
                "media_type": "application/zip",
                "file_size": len(output.getvalue()),
                "path": export_path,
                "download_url": f"/api/pdf/tables/download/{filename}"
            }
            
        except Exception as e:
            logger.error(f"Erreur export CSV: {e}")
            raise
    
    async def _export_to_json(self, dataframes: List[pd.DataFrame], export_filename: str) -> Dict[str, Any]:
        """
        Exporte les DataFrames en fichier JSON.
        
        Args:
            dataframes: Liste de DataFrames à exporter
            export_filename: Nom du fichier d'export (sans extension)
            
        Returns:
            Informations sur le fichier exporté
        """
        try:
            # Créer un fichier JSON en mémoire
            output = io.BytesIO()
            
            # Fonction pour assurer la sérialisation JSON correcte
            def json_serializer(obj):
                if isinstance(obj, (pd.Series, pd.Index)):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient="records")
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                return str(obj)
            
            # Préparer les données JSON
            json_data = []
            for i, df in enumerate(dataframes):
                # Remplacer les NaN par None pour JSON
                df = df.replace({pd.NA: None, np.nan: None})
                
                # Convertir le DataFrame en dictionnaire
                table_data = {
                    "table_id": i + 1,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "data": df.to_dict(orient="records")
                }
                
                # Extraire les métadonnées si présentes
                if "_page" in df.columns:
                    table_data["page"] = df["_page"].iloc[0]
                if "_table_id" in df.columns:
                    table_data["original_id"] = df["_table_id"].iloc[0]
                
                json_data.append(table_data)
            
            # Sérialiser en JSON avec indentation pour lisibilité
            loop = asyncio.get_event_loop()
            json_str = await loop.run_in_executor(
                self.executor,
                lambda: json.dumps(json_data, default=json_serializer, indent=2)
            )
            
            # Écrire dans le buffer
            output.write(json_str.encode('utf-8'))
            output.seek(0)
            
            # Sauvegarder le fichier
            filename = f"{export_filename}.json"
            export_path = os.path.join(self.temp_dir, filename)
            
            with open(export_path, "wb") as f:
                f.write(output.getvalue())
            
            return {
                "content": output.getvalue(),
                "filename": filename,
                "media_type": "application/json",
                "file_size": len(output.getvalue()),
                "path": export_path,
                "download_url": f"/api/pdf/tables/download/{filename}"
            }
            
        except Exception as e:
            logger.error(f"Erreur export JSON: {e}")
            raise
    
    async def _export_to_html(self, dataframes: List[pd.DataFrame], export_filename: str) -> Dict[str, Any]:
        """
        Exporte les DataFrames en fichier HTML interactif.
        
        Args:
            dataframes: Liste de DataFrames à exporter
            export_filename: Nom du fichier d'export (sans extension)
            
        Returns:
            Informations sur le fichier exporté
        """
        try:
            # Créer un fichier HTML en mémoire
            output = io.BytesIO()
            
            # Style CSS moderne avec support pour le responsive design
            html_style = """
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap');
                
                body {
                    font-family: 'Open Sans', sans-serif;
                    margin: 20px;
                    color: #333;
                    background-color: #f9f9f9;
                }
                
                h1 {
                    color: #2c3e50;
                    margin-bottom: 20px;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }
                
                h2 {
                    color: #2980b9;
                    margin-top: 30px;
                    margin-bottom: 15px;
                }
                
                .metadata {
                    color: #7f8c8d;
                    font-size: 0.9em;
                    margin-bottom: 10px;
                }
                
                .table-container {
                    margin-bottom: 30px;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 10px;
                    overflow-x: auto;
                }
                
                .dataTables_wrapper {
                    padding: 10px;
                    overflow-x: auto;
                }
                
                table.dataTable {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }
                
                table.dataTable thead th {
                    background-color: #3498db;
                    color: white;
                    padding: 10px;
                    text-align: left;
                    font-weight: 600;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }
                
                table.dataTable tbody tr:nth-child(odd) {
                    background-color: #f2f2f2;
                }
                
                table.dataTable tbody tr:hover {
                    background-color: #e1f5fe;
                }
                
                table.dataTable tbody td {
                    padding: 8px 10px;
                    border-bottom: 1px solid #ddd;
                }
                
                @media (max-width: 768px) {
                    body {
                        margin: 10px;
                    }
                    
                    h1 {
                        font-size: 1.5rem;
                    }
                    
                    h2 {
                        font-size: 1.2rem;
                    }
                    
                    table.dataTable tbody td,
                    table.dataTable thead th {
                        padding: 6px;
                        font-size: 0.9rem;
                    }
                }
            </style>
            """
            
            # JavaScript pour interactivité (tri, recherche, pagination)
            html_script = """
            <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/datatables/1.10.21/js/jquery.dataTables.min.js"></script>
            <script>
                $(document).ready(function() {
                    $('.data-table').DataTable({
                        paging: true,
                        searching: true,
                        ordering: true,
                        info: true,
                        pageLength: 15,
                        lengthMenu: [[10, 15, 25, 50, -1], [10, 15, 25, 50, "Tous"]],
                        language: {
                            search: "Rechercher:",
                            lengthMenu: "Afficher _MENU_ entrées",
                            info: "Affichage de _START_ à _END_ sur _TOTAL_ entrées",
                            paginate: {
                                first: "Premier",
                                last: "Dernier",
                                next: "Suivant",
                                previous: "Précédent"
                            }
                        },
                        responsive: true
                    });
                });
            </script>
            """
            
            # Générer le contenu HTML
            html_content = f"""<!DOCTYPE html>
            <html lang="fr">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Export des tableaux</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/datatables/1.10.21/css/jquery.dataTables.min.css">
                {html_style}
            </head>
            <body>
                <h1>Export des tableaux</h1>
                <div class="metadata">
                    <p>Date d'exportation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Nombre de tableaux: {len(dataframes)}</p>
                </div>
            """
            
            # Ajouter chaque tableau
            for i, df in enumerate(dataframes):
                # Extraire les métadonnées si présentes
                page_num = df["_page"].iloc[0] if "_page" in df.columns else "N/A"
                table_id = df["_table_id"].iloc[0] if "_table_id" in df.columns else i+1
                
                # Nettoyer le DataFrame pour l'affichage
                display_df = df.copy()
                if "_page" in display_df.columns:
                    display_df = display_df.drop("_page", axis=1)
                if "_table_id" in display_df.columns:
                    display_df = display_df.drop("_table_id", axis=1)
                
                # Remplacer les NaN par des chaînes vides
                display_df = display_df.fillna('')
                
                html_content += f"""
                <h2>Tableau {table_id}</h2>
                <div class="metadata">Page: {page_num}</div>
                <div class="table-container">
                    {display_df.to_html(index=False, classes='display data-table', border=0, escape=True)}
                </div>
                """
            
            html_content += f"""
                {html_script}
            </body>
            </html>
            """
            
            # Écrire dans le buffer
            output.write(html_content.encode('utf-8'))
            output.seek(0)
            
            # Sauvegarder le fichier
            filename = f"{export_filename}.html"
            export_path = os.path.join(self.temp_dir, filename)
            
            with open(export_path, "wb") as f:
                f.write(output.getvalue())
            
            return {
                "content": output.getvalue(),
                "filename": filename,
                "media_type": "text/html",
                "file_size": len(output.getvalue()),
                "path": export_path,
                "download_url": f"/api/pdf/tables/download/{filename}"
            }
            
        except Exception as e:
            logger.error(f"Erreur export HTML: {e}")
            raise
    
    async def _export_to_pdf(self, dataframes: List[pd.DataFrame], export_filename: str) -> Dict[str, Any]:
        """
        Exporte les DataFrames en fichier PDF.
        
        Args:
            dataframes: Liste de DataFrames à exporter
            export_filename: Nom du fichier d'export (sans extension)
            
        Returns:
            Informations sur le fichier exporté
        """
        try:
            # Import de la bibliothèque dans la fonction pour éviter des imports inutiles
            try:
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import A4, landscape
                from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import cm
                from reportlab.pdfbase import pdfmetrics
                from reportlab.pdfbase.ttfonts import TTFont
            except ImportError:
                logger.error("Bibliothèque reportlab non disponible. Installation requise.")
                raise ImportError("Bibliothèque reportlab requise pour l'export PDF.")
            
            # Créer un fichier PDF en mémoire
            buffer = io.BytesIO()
            
            # Fonction pour créer le PDF (à exécuter dans un thread)
            def create_pdf():
                # Enregistrer les polices
                try:
                    # Essayer d'utiliser une police plus moderne si disponible
                    # Si échec, utilise les polices par défaut de reportlab
                    pdfmetrics.registerFont(TTFont('OpenSans', '/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf'))
                    pdfmetrics.registerFont(TTFont('OpenSansBold', '/usr/share/fonts/truetype/open-sans/OpenSans-Bold.ttf'))
                    has_custom_font = True
                except:
                    has_custom_font = False
                
                # Créer le document
                doc = SimpleDocTemplate(
                    buffer,
                    pagesize=landscape(A4),
                    title=f"Tableaux exportés",
                    author="Système d'extraction de tableaux"
                )
                
                # Styles
                styles = getSampleStyleSheet()
                if has_custom_font:
                    title_style = ParagraphStyle(
                        'CustomTitle',
                        parent=styles['Heading1'],
                        fontName='OpenSansBold',
                        fontSize=16,
                        spaceAfter=12,
                        textColor=colors.darkblue
                    )
                    normal_style = ParagraphStyle(
                        'CustomNormal',
                        parent=styles['Normal'],
                        fontName='OpenSans',
                        fontSize=10
                    )
                else:
                    title_style = styles['Heading1']
                    normal_style = styles['Normal']
                
                # Contenu du document
                content = []
                
                # Titre principal
                content.append(Paragraph(f"Tableaux exportés", title_style))
                content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
                content.append(Paragraph(f"Nombre de tableaux: {len(dataframes)}", normal_style))
                content.append(Spacer(1, 1*cm))
                
                # Pour chaque tableau
                for i, df in enumerate(dataframes):
                    # Extraire les métadonnées si présentes
                    page_num = df["_page"].iloc[0] if "_page" in df.columns else "N/A"
                    table_id = df["_table_id"].iloc[0] if "_table_id" in df.columns else i+1
                    
                    # Nettoyer le DataFrame pour l'affichage
                    display_df = df.copy()
                    if "_page" in display_df.columns:
                        display_df = display_df.drop("_page", axis=1)
                    if "_table_id" in display_df.columns:
                        display_df = display_df.drop("_table_id", axis=1)
                    
                    # Remplacer les NaN par des chaînes vides
                    display_df = display_df.fillna('')
                    
                    # Titre du tableau
                    content.append(Paragraph(f"Tableau {table_id} (Page: {page_num})", title_style))
                    content.append(Spacer(1, 0.5*cm))
                    
                    # Préparer les données du tableau
                    data = [display_df.columns.tolist()]
                    for row in display_df.values:
                        # Convertir tous les éléments en chaînes
                        data.append([str(x) for x in row])
                    
                    # Créer le tableau
                    table = Table(data, repeatRows=1)
                    
                    # Style du tableau
                    table_style = TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ])
                    
                    # Alterner les couleurs des lignes
                    for row in range(1, len(data)):
                        if row % 2 == 0:
                            table_style.add('BACKGROUND', (0, row), (-1, row), colors.lightgrey)
                    
                    table.setStyle(table_style)
                    
                    # Ajouter le tableau au contenu
                    content.append(table)
                    content.append(Spacer(1, 1*cm))
                
                # Construire le document
                doc.build(content)
            
            # Exécuter la création du PDF dans un thread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, create_pdf)
            
            # Finaliser le fichier
            buffer.seek(0)
            
            # Sauvegarder le fichier
            filename = f"{export_filename}.pdf"
            export_path = os.path.join(self.temp_dir, filename)
            
            with open(export_path, "wb") as f:
                f.write(buffer.getvalue())
            
            return {
                "content": buffer.getvalue(),
                "filename": filename,
                "media_type": "application/pdf",
                "file_size": len(buffer.getvalue()),
                "path": export_path,
                "download_url": f"/api/pdf/tables/download/{filename}"
            }
            
        except Exception as e:
            logger.error(f"Erreur export PDF: {e}")
            raise
    
    async def _export_to_markdown(self, dataframes: List[pd.DataFrame], export_filename: str) -> Dict[str, Any]:
        """
        Exporte les DataFrames en fichier Markdown.
        
        Args:
            dataframes: Liste de DataFrames à exporter
            export_filename: Nom du fichier d'export (sans extension)
            
        Returns:
            Informations sur le fichier exporté
        """
        try:
            # Créer un fichier Markdown en mémoire
            output = io.StringIO()
            
            # Entête du document Markdown
            output.write(f"# Tableaux exportés\n\n")
            output.write(f"Date d'exportation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            output.write(f"Nombre de tableaux: {len(dataframes)}  \n\n")
            
            # Pour chaque tableau
            for i, df in enumerate(dataframes):
                # Extraire les métadonnées si présentes
                page_num = df["_page"].iloc[0] if "_page" in df.columns else "N/A"
                table_id = df["_table_id"].iloc[0] if "_table_id" in df.columns else i+1
                
                # Nettoyer le DataFrame pour l'affichage
                display_df = df.copy()
                if "_page" in display_df.columns:
                    display_df = display_df.drop("_page", axis=1)
                if "_table_id" in display_df.columns:
                    display_df = display_df.drop("_table_id", axis=1)
                
                # Remplacer les NaN par des chaînes vides
                display_df = display_df.fillna('')
                
                # Titre du tableau
                output.write(f"## Tableau {table_id}\n\n")
                output.write(f"Page: {page_num}  \n\n")
                
                # Convertir le DataFrame en format Markdown
                markdown_table = display_df.to_markdown(index=False)
                output.write(markdown_table)
                output.write("\n\n")
            
            # Récupérer le contenu
            content = output.getvalue()
            output.close()
            
            # Sauvegarder le fichier
            filename = f"{export_filename}.md"
            export_path = os.path.join(self.temp_dir, filename)
            
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return {
                "content": content.encode('utf-8'),
                "filename": filename,
                "media_type": "text/markdown",
                "file_size": len(content.encode('utf-8')),
                "path": export_path,
                "download_url": f"/api/pdf/tables/download/{filename}"
            }
            
        except Exception as e:
            logger.error(f"Erreur export Markdown: {e}")
            raise

    def _prepare_dataframe_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare un DataFrame pour l'exportation en gérant les types spéciaux.
        
        Args:
            df: DataFrame à préparer
            
        Returns:
            DataFrame préparé pour l'exportation
        """
        # Créer une copie pour ne pas modifier l'original
        df_copy = df.copy()
        
        # 1. Convertir les colonnes catégorielles en str
        categorical_columns = df_copy.select_dtypes(include=['category']).columns
        for col in categorical_columns:
            df_copy[col] = df_copy[col].astype(str)
        
        # 2. Gérer les NaT dans les colonnes de dates
        datetime_columns = df_copy.select_dtypes(include=['datetime64']).columns
        for col in datetime_columns:
            df_copy[col] = df_copy[col].astype(str)
        
        # 3. Gérer les autres types complexes
        for col in df_copy.columns:
            # Vérifier si la colonne contient des types complexes (listes, dicts, etc.)
            if df_copy[col].apply(lambda x: isinstance(x, (list, dict, set, tuple))).any():
                df_copy[col] = df_copy[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict, set, tuple)) else x)
        
        return df_copy
    
    async def cleanup_old_exports(self, max_age_hours: int = 24):
        """
        Nettoie les anciens fichiers d'export.
        
        Args:
            max_age_hours: Âge maximum des fichiers en heures
        """
        try:
            now = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                
                if os.path.isfile(file_path):
                    file_age = now - os.path.getmtime(file_path)
                    
                    if file_age > max_age_seconds:
                        os.unlink(file_path)
                        logger.debug(f"Fichier d'export nettoyé: {filename}")
                        
            logger.info(f"Nettoyage des anciens exports terminé. Délai: {max_age_hours} heures")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage exports: {e}")
    
    async def cleanup(self):
        """Nettoie les ressources."""
        try:
            self.executor.shutdown(wait=False)
            logger.info("Exporteur de tableaux nettoyé")
        except Exception as e:
            logger.error(f"Erreur nettoyage exporteur: {e}")