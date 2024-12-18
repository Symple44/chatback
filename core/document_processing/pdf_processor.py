from PyPDF2 import PdfReader
from core.search import ElasticsearchClient
import os

class PDFProcessor:
    def __init__(self, es_client: ElasticsearchClient):
        self.es_client = es_client

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        logger.info(f"Extraction du texte de {file_path}")
        text_content = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    text_content += f"\n=== Page {page_num} ===\n{text}"
                    logger.info(f"Page {page_num} extraite avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction: {e}")
        return text_content

    def index_pdfs_in_directory(self, directory: str):
        """
        Indexe récursivement les PDFs d'un répertoire dans Elasticsearch
        """
        for subdir, _, files in os.walk(directory):
            product = os.path.basename(subdir)
            for filename in files:
                if filename.endswith(".pdf"):
                    file_path = os.path.join(subdir, filename)
                    try:
                        for text, page_number in self.extract_text_from_pdf(file_path):
                            self.es_client.index_document(
                                title=filename,
                                content=text,
                                metadata={
                                    "product": product,
                                    "page": page_number
                                }
                            )
                        print(f"Indexed: {filename}")
                    except Exception as e:
                        print(f"Failed to index {filename}: {e}")