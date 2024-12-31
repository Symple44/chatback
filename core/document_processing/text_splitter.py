# core/document_processing/text_splitter.py
from typing import List, Dict, Optional, Any, Iterator
import re
from dataclasses import dataclass
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger("text_splitter")

@dataclass
class TextChunk:
    """Chunk de texte avec métadonnées."""
    content: str
    index: int
    metadata: Dict[str, Any]
    prev_context: str = ""
    next_context: str = ""
    source: Optional[str] = None

class DocumentSplitter:
    def __init__(self):
        """Initialise le splitter de documents."""
        # Configuration générale
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.max_chunks = settings.MAX_CHUNKS_PER_DOC

        # Patterns de séparation
        self.separators = [
            "\n\n",      # Paragraphes
            "\n",        # Lignes
            ". ",        # Phrases
            ", ",        # Clauses
            " ",        # Mots
            ""          # Caractères
        ]

        # Patterns à nettoyer
        self.cleanup_patterns = [
            r"^\s*[\d\-•]+\s*$",           # Lignes avec uniquement des numéros/puces
            r"^Page \d+(\s*of\s*\d+)?$",   # En-têtes de page
            r"^\s*\[TOC\]",                # Table des matières
            r"_{3,}",                       # Lignes de séparation
            r"©.*?(?=\n|$)",               # Mentions de copyright
            r"\[.*?\]",                     # Références entre crochets
        ]

        # Patterns à préserver
        self.preserve_patterns = {
            "code_block": r"```[\s\S]*?```",
            "table": r"\|.*\|[\s\S]*?\|.*\|",
            "list_item": r"^\s*[\-\*]\s+.*$"
        }

    def split_documents(
        self,
        documents: List[Dict[str, Any]],
        min_chunk_size: int = 100
    ) -> List[TextChunk]:
        """
        Découpe plusieurs documents en chunks.
        
        Args:
            documents: Liste des documents à découper
            min_chunk_size: Taille minimale d'un chunk
            
        Returns:
            Liste des chunks créés
        """
        try:
            with metrics.timer("document_splitting"):
                all_chunks = []
                
                for doc in documents:
                    chunks = self.split_text(
                        text=doc["content"],
                        metadata={
                            "title": doc.get("title", "Unknown"),
                            "source": doc.get("source"),
                            **doc.get("metadata", {})
                        },
                        min_chunk_size=min_chunk_size
                    )
                    all_chunks.extend(chunks)

                metrics.increment_counter("documents_split", len(documents))
                return all_chunks

        except Exception as e:
            logger.error(f"Erreur découpage documents: {e}")
            metrics.increment_counter("document_splitting_errors")
            return []

    def split_document(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Découpe un document en chunks."""
        chunks = []
        
        try:
            text = self._preprocess_text(text)
            raw_chunks = self._split_into_chunks(text)
            
            for i, chunk_text in enumerate(raw_chunks):
                if len(chunk_text.strip()) < self.chunk_size / 4:  # Skip très petits chunks
                    continue
                    
                chunks.append({
                    "content": chunk_text,
                    "index": i,
                    "metadata": {
                        **(metadata or {}),
                        "chunk": i,
                        "total_chunks": len(raw_chunks)
                    }
                })
                
            return chunks
            
        except Exception as e:
            logger.error(f"Erreur découpage document: {e}")
            return []
    
    def split_text(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        min_chunk_size: int = 100
    ) -> List[TextChunk]:
        """
        Découpe un texte en chunks avec préservation du contexte.
        
        Args:
            text: Texte à découper
            metadata: Métadonnées du document
            min_chunk_size: Taille minimale d'un chunk
            
        Returns:
            Liste des chunks créés
        """
        try:
            # Prétraitement du texte
            text = self._preprocess_text(text)
            preserved_blocks = self._extract_preserved_blocks(text)
            
            # Découpage initial
            chunks = self._split_into_chunks(text)
            
            # Post-traitement et enrichissement
            processed_chunks = []
            for i, chunk_text in enumerate(chunks):
                # Vérification de la taille minimale
                if len(chunk_text.strip()) < min_chunk_size:
                    continue
                
                # Restauration des blocs préservés
                chunk_text = self._restore_preserved_blocks(chunk_text, preserved_blocks)
                
                # Création du chunk avec contexte
                chunk = TextChunk(
                    content=chunk_text,
                    index=i,
                    metadata=metadata or {},
                    prev_context=self._get_context(chunks, i - 1) if i > 0 else "",
                    next_context=self._get_context(chunks, i + 1) if i < len(chunks) - 1 else ""
                )
                
                processed_chunks.append(chunk)

            # Limitation du nombre de chunks
            if len(processed_chunks) > self.max_chunks:
                logger.warning(f"Nombre de chunks limité à {self.max_chunks}")
                processed_chunks = processed_chunks[:self.max_chunks]

            return processed_chunks

        except Exception as e:
            logger.error(f"Erreur découpage texte: {e}")
            return []

    def _preprocess_text(self, text: str) -> str:
        """Prétraite le texte avant découpage."""
        # Normalisation des sauts de ligne
        text = text.replace('\r\n', '\n')
        
        # Nettoyage des patterns indésirables
        for pattern in self.cleanup_patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)
        
        # Normalisation des espaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()

    def _extract_preserved_blocks(self, text: str) -> Dict[str, List[str]]:
        """Extrait les blocs à préserver."""
        preserved = {k: [] for k in self.preserve_patterns}
        
        for block_type, pattern in self.preserve_patterns.items():
            matches = re.finditer(pattern, text, re.MULTILINE)
            for i, match in enumerate(matches):
                placeholder = f"___{block_type}_{i}___"
                preserved[block_type].append((placeholder, match.group(0)))
                text = text.replace(match.group(0), placeholder)
                
        return preserved

    def _restore_preserved_blocks(
        self,
        text: str,
        preserved_blocks: Dict[str, List[str]]
    ) -> str:
        """Restaure les blocs préservés."""
        for block_type, blocks in preserved_blocks.items():
            for placeholder, content in blocks:
                text = text.replace(placeholder, content)
        return text

    def _split_into_chunks(self, text: str) -> List[str]:
        """Découpe le texte en chunks."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Si la phrase est trop longue, la découper
            if sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Découpage de la phrase longue
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)
                continue
            
            # Ajout normal au chunk courant
            if current_length + sentence_length > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Ajout du dernier chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Découpe le texte en phrases."""
        # Pattern plus sophistiqué pour la détection des phrases
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Découpe une phrase trop longue."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        words = sentence.split()
        
        for word in words:
            word_length = len(word)
            
            if current_length + word_length > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = word_length
                else:
                    # Mot trop long, le découper arbitrairement
                    chunks.append(word[:self.chunk_size])
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def _get_context(self, chunks: List[str], index: int) -> str:
        """Récupère le contexte d'un chunk."""
        if 0 <= index < len(chunks):
            context = chunks[index].strip()
            if len(context) > self.chunk_overlap:
                context = context[:self.chunk_overlap] + "..."
            return context
        return ""

    def estimate_chunks(self, text: str) -> int:
        """Estime le nombre de chunks pour un texte."""
        return len(self._split_into_chunks(text))

    def _merge_short_chunks(
        self,
        chunks: List[TextChunk],
        min_size: int
    ) -> List[TextChunk]:
        """Fusionne les chunks trop courts."""
        merged = []
        temp_chunk = None
        
        for chunk in chunks:
            if temp_chunk is None:
                temp_chunk = chunk
            else:
                combined_length = len(temp_chunk.content) + len(chunk.content)
                
                if combined_length <= self.chunk_size:
                    # Fusion des chunks
                    temp_chunk.content += f"\n{chunk.content}"
                    temp_chunk.next_context = chunk.next_context
                    temp_chunk.metadata.update(chunk.metadata)
                else:
                    merged.append(temp_chunk)
                    temp_chunk = chunk
        
        if temp_chunk:
            merged.append(temp_chunk)
            
        return merged
