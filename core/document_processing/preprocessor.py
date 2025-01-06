# core/document_processing/preprocessor.py
from typing import List, Dict, Optional
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ProcessedSection:
    title: str
    content: str
    importance_score: float
    metadata: Dict

class DocumentPreprocessor:
    def __init__(self):
        """Initialisation du processeur de documents."""
        # Configuration spécifique au document
        self.document_prefix = 'CMA'  # Préfixe fixe
        self.document_revision_prefix = '_rev'  # Préfixe de révision

        # Marqueurs de sections
        self.section_markers = [
            r"^(?:Pour|Comment)\s+[a-zéèêë].*?$",  # Instructions directes
            r"^(?:\d+[\).]|[-•])\s+.*$",           # Points numérotés ou puces
            r"^(?:Étape|Phase|Partie)\s+.*:?$",    # Sections structurées
            r"^.*?\b(?:cliquez|sélectionnez|renseignez|faites)\b.*$"  # Actions utilisateur
        ]
        
        # Patterns de nettoyage
        self.cleanup_patterns = [
            # En-têtes et pieds de page
            (r'^2M-MANAGER\s*–\s*.*?11000\s*CARCASSONNE.*?CMA\d+_rev\d+.*?\n', ''),
            (r'=== Page \d+ ===', ''),
            
            # Informations de version et révision
            (r'Historique des révisions.*', ''),
            (r'Droit de reproduction.*', ''),
            (r'Responsabilité.*', ''),
            
            # Nettoyage général
            (r'\s{2,}', ' '),           # Espaces multiples 
            (r'^\s+', ''),              # Espaces début de ligne
            (r'\s+$', ''),              # Espaces fin de ligne
            (r'\n{3,}', '\n\n'),        # Lignes vides multiples
            
            # Suppression des logos et marqueurs
            (r'2M-MANAGER.*?\n', ''),   # En-têtes
            (r'Page \d+ sur \d+', ''),  # Numéros de page
            (r'Tél : [\d\.-]+', ''),    # Numéros de téléphone
            (r'<\|\w+\|>.*?<\/\|\w+\|>', '')  # Balises de formatage
        ]

        # Patterns spécifiques pour différents types de documents
        self.document_specific_patterns = {
            'pdf': [
                (r'Sommaire.*?Page \d+', ''),
                (r'Droit d\'auteur.*', '')
            ]
        }

    def preprocess_document(self, doc: Dict) -> Dict:
        """Prétraite un document complet."""
        try:
            # Extraction des métadonnées utiles
            metadata = self._extract_metadata(doc)
            
            # Nettoyage du contenu
            content = self._clean_content(doc.get("content", ""))
            
            # Extraction des sections
            sections = self._extract_sections(content)
            
            # Calcul des scores d'importance
            scored_sections = self._score_sections(sections)
            
            return {
                "doc_id": doc.get("doc_id", ""),
                "title": doc.get("title", ""),
                "content": content,
                "metadata": metadata,
                "sections": scored_sections,
                "processed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise RuntimeError(f"Erreur preprocessing document: {str(e)}")

    def _extract_metadata(self, doc: Dict) -> Dict:
        """Extrait et normalise les métadonnées pertinentes."""
        metadata = {}
        
        if doc.get("metadata"):
            meta = doc["metadata"]
            metadata.update({
                "author": meta.get("author", "").strip(),
                "revision": self._extract_revision(meta.get("title", "")),
                "date": meta.get("creation_date", ""),
                "source": "documentation"
            })
            
        return metadata

    def _extract_revision(self, title: str) -> str:
        """Extrait le numéro de révision du titre."""
        match = re.search(r"rev(\d+)", title.lower())
        return match.group(1) if match else "00"

    def _clean_content(self, content: str, doc_type: str = 'pdf') -> str:
        """Nettoie et normalise le contenu."""
        cleaned = content

        # Utilisation des flags communs
        flags = re.MULTILINE | re.IGNORECASE | re.DOTALL

        # Patterns généraux
        for pattern, replacement in self.cleanup_patterns:
            cleaned = re.sub(pattern, replacement, cleaned, flags=flags)

        # Nettoyage final
        cleaned_lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        cleaned = '\n'.join(cleaned_lines)

        return cleaned.strip()

    def _extract_sections(self, content: str) -> List[str]:
        """Extrait les sections logiques du contenu."""
        sections = []
        current_section = []
        
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            # Nouvelle section si ligne correspond à un marqueur
            is_new_section = any(re.match(pattern, line, re.IGNORECASE) 
                               for pattern in self.section_markers)
            
            if is_new_section and current_section:
                sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
                
        if current_section:
            sections.append("\n".join(current_section))
            
        return sections

    def _score_sections(self, sections: List[str]) -> List[ProcessedSection]:
        """Attribue des scores d'importance aux sections."""
        scored_sections = []
        
        for i, section in enumerate(sections):
            # Calcul du score basé sur plusieurs facteurs
            score = 0.0
            
            # Présence de mots-clés d'action
            action_words = len(re.findall(r'\b(cliquez|sélectionnez|renseignez|faites)\b', 
                                        section.lower()))
            score += action_words * 0.2
            
            # Présence d'étapes numérotées
            numbered_steps = len(re.findall(r'^\d+[.)]', section, re.MULTILINE))
            score += numbered_steps * 0.15
            
            # Position dans le document (les premières sections sont souvent plus importantes)
            score += max(0, 1 - (i * 0.1))
            
            # Longueur significative mais pas excessive
            text_length = len(section)
            if 100 < text_length < 1000:
                score += 0.1
                
            # Création de la section traitée
            title = self._extract_section_title(section)
            processed_section = ProcessedSection(
                title=title,
                content=section,
                importance_score=min(1.0, score),
                metadata={
                    "word_count": len(section.split()),
                    "has_steps": numbered_steps > 0,
                    "has_actions": action_words > 0
                }
            )
            
            scored_sections.append(processed_section)
            
        return sorted(scored_sections, key=lambda x: x.importance_score, reverse=True)

    def _extract_section_title(self, section: str) -> str:
        """Extrait le titre d'une section."""
        # Prend la première ligne non vide
        lines = [l for l in section.split("\n") if l.strip()]
        if not lines:
            return ""
            
        # Limite la longueur du titre
        title = lines[0][:100]
        return title.strip()

    def get_most_relevant_sections(self, 
                                 processed_docs: List[Dict],
                                 query: str,
                                 max_sections: int = 3) -> List[ProcessedSection]:
        """Retourne les sections les plus pertinentes pour une requête."""
        all_sections = []
        for doc in processed_docs:
            all_sections.extend(doc["sections"])
            
        # Calcul de la pertinence par rapport à la requête
        for section in all_sections:
            relevance = self._calculate_query_relevance(section, query)
            section.importance_score *= (1 + relevance)
            
        # Tri et sélection des meilleures sections
        return sorted(all_sections, 
                     key=lambda x: x.importance_score, 
                     reverse=True)[:max_sections]

    def _calculate_query_relevance(self, section: ProcessedSection, query: str) -> float:
        """Calcule la pertinence d'une section par rapport à la requête."""
        query_words = set(query.lower().split())
        content_words = set(section.content.lower().split())
        
        # Intersection des mots
        common_words = query_words.intersection(content_words)
        
        # Score basé sur le nombre de mots communs
        return len(common_words) / max(len(query_words), 1)
