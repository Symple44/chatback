# core/processing/chunking/token_chunker.py
from typing import List, Dict, Any, Optional
import re
from .base import ChunkingStrategy, Chunk, ChunkingError

class TokenChunker(ChunkingStrategy):
    """
    Stratégie de découpage basée sur les tokens avec overlap.
    Respecte les limites naturelles du texte (phrases, paragraphes).
    """

    def validate_config(self) -> None:
        """Valide la configuration du chunker."""
        required = {
            "chunk_size": (int, "Nombre de tokens par chunk"),
            "chunk_overlap": (int, "Nombre de tokens de chevauchement"),
            "min_chunk_size": (int, "Taille minimum d'un chunk"),
        }
        
        for key, (expected_type, description) in required.items():
            if key not in self.config:
                raise ChunkingError(
                    f"Configuration manquante: {key} ({description})",
                    {"required_configs": list(required.keys())}
                )
            if not isinstance(self.config[key], expected_type):
                raise ChunkingError(
                    f"Type invalide pour {key}: attendu {expected_type.__name__}",
                    {"key": key, "expected": expected_type.__name__}
                )

        # Validation des valeurs
        if self.config["chunk_size"] < self.config["min_chunk_size"]:
            raise ChunkingError(
                "chunk_size doit être supérieur à min_chunk_size",
                {"chunk_size": self.config["chunk_size"], 
                 "min_chunk_size": self.config["min_chunk_size"]}
            )
        
        if self.config["chunk_overlap"] >= self.config["chunk_size"]:
            raise ChunkingError(
                "chunk_overlap doit être inférieur à chunk_size",
                {"chunk_overlap": self.config["chunk_overlap"], 
                 "chunk_size": self.config["chunk_size"]}
            )

    def create_chunks(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Découpe le texte en chunks en respectant les limites naturelles.
        
        Args:
            text: Texte à découper
            metadata: Métadonnées à inclure dans chaque chunk
            
        Returns:
            Liste des chunks créés
        """
        try:
            # Paramètres de configuration
            chunk_size = self.config["chunk_size"]
            chunk_overlap = self.config["chunk_overlap"]
            min_chunk_size = self.config["min_chunk_size"]

            # Prétraitement du texte
            text = self._preprocess_text(text)
            
            # Découpage en paragraphes
            paragraphs = self._split_into_paragraphs(text)
            
            # Création des chunks
            chunks = []
            current_chunk = []
            current_size = 0
            current_start = 0
            
            for para in paragraphs:
                # Tokenisation du paragraphe
                para_tokens = self._tokenize(para)
                para_size = len(para_tokens)
                
                # Si le paragraphe seul dépasse la taille max, le découper
                if para_size > chunk_size:
                    if current_chunk:
                        # Sauvegarder le chunk en cours avant
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(self._create_chunk(
                            content=chunk_text,
                            index=len(chunks),
                            start_char=current_start,
                            end_char=current_start + len(chunk_text),
                            metadata=metadata
                        ))
                        current_chunk = []
                        current_size = 0
                    
                    # Découper le paragraphe
                    para_chunks = self._split_paragraph(
                        para_tokens,
                        chunk_size,
                        chunk_overlap
                    )
                    
                    # Ajouter les chunks du paragraphe
                    for p_chunk in para_chunks:
                        chunk_text = ' '.join(p_chunk)
                        chunks.append(self._create_chunk(
                            content=chunk_text,
                            index=len(chunks),
                            start_char=text.find(chunk_text),
                            end_char=text.find(chunk_text) + len(chunk_text),
                            metadata=metadata
                        ))
                        
                else:
                    # Vérifier si l'ajout du paragraphe dépasse la taille max
                    if current_size + para_size > chunk_size:
                        # Sauvegarder le chunk en cours si assez grand
                        if current_size >= min_chunk_size:
                            chunk_text = ' '.join(current_chunk)
                            chunks.append(self._create_chunk(
                                content=chunk_text,
                                index=len(chunks),
                                start_char=current_start,
                                end_char=current_start + len(chunk_text),
                                metadata=metadata
                            ))
                        
                        # Commencer un nouveau chunk avec overlap
                        if chunk_overlap > 0 and current_chunk:
                            overlap_tokens = current_chunk[-chunk_overlap:]
                            current_chunk = overlap_tokens
                            current_size = len(overlap_tokens)
                        else:
                            current_chunk = []
                            current_size = 0
                        current_start = text.find(para)
                    
                    # Ajouter le paragraphe au chunk en cours
                    current_chunk.extend(para_tokens)
                    current_size += para_size
            
            # Ajouter le dernier chunk s'il est assez grand
            if current_chunk and current_size >= min_chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(
                    content=chunk_text,
                    index=len(chunks),
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    metadata=metadata
                ))
            
            return chunks

        except Exception as e:
            raise ChunkingError(
                f"Erreur lors du découpage: {str(e)}",
                {"text_length": len(text)}
            )

    def _preprocess_text(self, text: str) -> str:
        """Prétraite le texte."""
        # Normalisation des sauts de ligne
        text = re.sub(r'\r\n', '\n', text)
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        # Suppression des espaces en début/fin
        return text.strip()

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Découpe le texte en paragraphes."""
        # Séparation sur les lignes vides
        paragraphs = re.split(r'\n\s*\n', text)
        # Filtrage des paragraphes vides
        return [p.strip() for p in paragraphs if p.strip()]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenise le texte en mots."""
        # Séparation sur les espaces et la ponctuation
        tokens = re.findall(r'\S+|\s+|[^\w\s]', text)
        return [t for t in tokens if t.strip()]

    def _split_paragraph(
        self,
        tokens: List[str],
        chunk_size: int,
        overlap: int
    ) -> List[List[str]]:
        """Découpe un paragraphe en respectant l'overlap."""
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Calculer la fin du chunk
            end = start + chunk_size
            if end > len(tokens):
                end = len(tokens)
            
            # Créer le chunk
            chunks.append(tokens[start:end])
            
            # Avancer au prochain début en tenant compte de l'overlap
            start = end - overlap
            
            # Si on est presque à la fin, prendre tout le reste
            if len(tokens) - start < chunk_size:
                if len(tokens) - start >= self.config["min_chunk_size"]:
                    chunks.append(tokens[start:])
                break
                
        return chunks

    def _create_chunk(
        self,
        content: str,
        index: int,
        start_char: int,
        end_char: int,
        metadata: Optional[Dict[str, Any]]
    ) -> Chunk:
        """Crée un objet Chunk avec les métadonnées appropriées."""
        chunk_metadata = self._create_chunk_metadata(
            metadata or {},
            {
                "chunk_index": index,
                "char_range": {"start": start_char, "end": end_char},
                "token_count": len(self._tokenize(content))
            }
        )
        
        return Chunk(
            content=content,
            index=index,
            metadata=chunk_metadata,
            start_char=start_char,
            end_char=end_char
        )