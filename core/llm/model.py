from typing import AsyncIterator, Iterator, List, Dict, Optional, Any, Union
import os
import logging
import torch
import torch.backends.cpu
import re
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TextIteratorStreamer
)
from threading import Thread
import gc
import asyncio
import time
import psutil
from datetime import datetime
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.document_processing.extractor import DocumentExtractor
from core.llm.embeddings import EmbeddingsManager

logger = get_logger("model")

class ContextProcessor:
    def __init__(self):
        self.confidence_threshold = settings.CONTEXT_CONFIDENCE_THRESHOLD
        self.max_relevant_docs = settings.MAX_RELEVANT_DOCS
        self.doc_extractor = DocumentExtractor()

    def analyze_documents(self, query: str, documents: List[Dict]) -> Dict:
        """
        Analyse dynamiquement les documents et leurs relations.
        """
        if not documents:
            return {
                'query': query,
                'topics': [],
                'relevant_docs': [],
                'suggestions': []
            }

        try:
            # Normalisation des scores
            max_score = max(doc.get('score', 0) for doc in documents)
            normalized_docs = [
                {**doc, 'score': doc.get('score', 0) / max_score if max_score > 0 else 0}
                for doc in documents
            ]

            # Analyse des documents pour extraire les sujets communs
            topics = self._extract_topics(normalized_docs)

            # Génération de suggestions basées sur les documents trouvés
            suggestions = self._generate_suggestions(query, normalized_docs, topics)

            return {
                'query': query,
                'topics': topics,
                'relevant_docs': [d for d in normalized_docs if d['score'] > self.confidence_threshold],
                'suggestions': suggestions
            }

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des documents: {e}")
            return {
                'query': query,
                'topics': [],
                'relevant_docs': [],
                'suggestions': []
            }

    def format_response(self, analysis: Dict) -> str:
        """
        Formate une réponse basée sur l'analyse dynamique.
        """
        if not analysis['relevant_docs']:
            return "Je ne trouve pas d'information précise concernant votre demande. Pouvez-vous la reformuler ?"

        # Si nous avons des suggestions, les présenter
        if analysis['suggestions']:
            return "\n\n".join(analysis['suggestions'])

        # Si nous avons un document très pertinent
        best_doc = analysis['relevant_docs'][0]
        if best_doc['score'] > 0.8:
            # Extraire et présenter la procédure
            steps = self.extract_procedure(best_doc.get('content', ''))
            if steps:
                return self.format_procedure(steps, best_doc)

        # Si nous n'avons pas assez d'informations précises
        topics_info = ", ".join(f"'{t['keyword']}'" for t in analysis['topics'][:3])
        return (f"Votre demande pourrait concerner plusieurs sujets ({topics_info}). "
                "Pouvez-vous préciser votre demande en incluant plus de détails sur ce que vous souhaitez faire ?")
    def __init__(self):
        """Initialise le processeur de contexte."""
        # Paramètres de base
        self.confidence_threshold = settings.CONTEXT_CONFIDENCE_THRESHOLD
        self.max_relevant_docs = settings.MAX_RELEVANT_DOCS
        self.action_verbs = settings.ACTION_VERBS
        
        # Définition des types de documents et commandes
        self.command_types = {
            'achat': {
                'keywords': ['achat', 'fournisseur', 'approvisionnement', 'commander'],
                'doc_patterns': ['CMA004', 'CMA001', 'approvisionnement', 'achat'],
                'description': "commande d'achat",
                'doc_prefix': 'CMA'
            },
            'transport': {
                'keywords': ['transport', 'affrètement', 'livraison', 'expédition'],
                'doc_patterns': ['CMA005', 'transport', 'expedition'],
                'description': "commande de transport",
                'doc_prefix': 'CMA'
            },
            'client': {
                'keywords': ['client', 'vente', 'devis', 'facturation'],
                'doc_patterns': ['CM012', 'vente', 'client', 'facture'],
                'description': "commande client",
                'doc_prefix': 'CM'
            },
            'sous_traitance': {
                'keywords': ['sous-traitance', 'sous traitance', 'prestation', 'phase'],
                'doc_patterns': ['CMA006', 'CMA011', 'sous-traitance'],
                'description': "commande de sous-traitance",
                'doc_prefix': 'CMA'
            }
        }
        
        # Patterns à ignorer dans le texte
        self.ignore_patterns = [
            r"^nous allons voir",
            r"^voir",
            r"^\d+\)",
            r"^étape",
            r"^procédure",
            r"^remarque",
            r"^note :",
        ]

    def detect_command_type(self, query: str, documents: List[Dict]) -> Optional[str]:
        """
        Détecte le type de commande basé sur la requête et les documents.
        """
        query = query.lower()
        type_scores = {cmd_type: 0 for cmd_type in self.command_types}
        
        try:
            # Analyse de la requête
            for cmd_type, config in self.command_types.items():
                # Points pour les mots-clés dans la requête
                for keyword in config['keywords']:
                    if keyword in query:
                        type_scores[cmd_type] += 2
                
                # Points pour les patterns dans les documents
                for doc in documents:
                    doc_title = doc.get('title', '').lower()
                    # Points bonus pour les préfixes de document correspondants
                    if doc_title.startswith(config['doc_prefix'].lower()):
                        type_scores[cmd_type] += 2
                    
                    # Points pour les patterns spécifiques
                    for pattern in config['doc_patterns']:
                        if pattern.lower() in doc_title:
                            type_scores[cmd_type] += 3
                    
                    # Analyse du contenu
                    content = doc.get('content', '').lower()
                    keyword_count = sum(content.count(keyword) for keyword in config['keywords'])
                    type_scores[cmd_type] += min(keyword_count, 5)  # Plafonné à 5 points
            
            # Sélectionner le type avec le meilleur score
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 3:  # Seuil minimum pour considérer un type valide
                logger.info(f"Type de commande détecté: {best_type[0]} (score: {best_type[1]})")
                return best_type[0]
            
            logger.info("Aucun type de commande n'a atteint le score minimum")
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la détection du type de commande: {e}")
            return None

    def filter_relevant_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Filtre et trie les documents pertinents.
        """
        if not documents:
            return []
            
        try:
            # Normalisation des scores
            max_score = max(doc.get('score', 0) for doc in documents)
            if max_score > 0:
                for doc in documents:
                    doc['score'] = doc.get('score', 0) / max_score
            
            # Filtrage par seuil de confiance
            relevant_docs = [
                doc for doc in documents 
                if doc.get('score', 0) >= self.confidence_threshold
            ]
            
            # Tri et limitation
            relevant_docs = sorted(
                relevant_docs,
                key=lambda x: x.get('score', 0),
                reverse=True
            )[:self.max_relevant_docs]
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Erreur lors du filtrage des documents: {e}")
            return []

    def clean_step(self, step: str) -> str:
        """
        Nettoie et formate une étape.
        """
        try:
            # Suppression des numéros et puces
            step = re.sub(r'^\s*\d+[\.\)]\s*', '', step)
            step = re.sub(r'^\s*[\-\•]\s*', '', step)
            
            # Nettoyage général
            step = step.strip()
            
            # Mise en majuscule de la première lettre
            if step and len(step) > 0:
                step = step[0].upper() + step[1:]
            
            # Ajout de la ponctuation finale si nécessaire
            if step and not step.endswith(('.', '!', '?')):
                step += '.'
            
            return step
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage de l'étape: {e}")
            return step

    def extract_procedure(self, content: str) -> List[Dict[str, Any]]:
        """
        Extrait la procédure complète avec étapes et détails.
        """
        try:
            lines = content.split('\n')
            steps = []
            current_section = None
            current_step = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Détection des sections
                if any(keyword in line.lower() for keyword in settings.PROCEDURE_KEYWORDS):
                    current_section = {
                        'title': line,
                        'steps': []
                    }
                    continue
                
                # Détection des étapes
                is_new_step = bool(re.match(r'^\s*(?:\d+[\.\)]|\-|\•)\s*.+', line))
                
                if is_new_step and self.is_valid_step(line):
                    if current_step:
                        steps.append(current_step)
                    
                    current_step = {
                        'text': self.clean_step(line),
                        'details': [],
                        'substeps': []
                    }
                
                elif current_step:
                    # Détection des sous-étapes
                    if re.match(r'^\s+[\-\•]\s+.+', line):
                        current_step['substeps'].append(self.clean_step(line))
                    # Ajout des détails pertinents
                    elif not any(re.search(pattern, line.lower()) for pattern in self.ignore_patterns):
                        current_step['details'].append(line)
            
            # Ajouter la dernière étape
            if current_step and self.is_valid_step(current_step['text']):
                steps.append(current_step)
            
            return steps
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de la procédure: {e}")
            return []

    def is_valid_step(self, step: str) -> bool:
        """
        Vérifie si une étape est valide.
        """
        step = step.lower().strip()
        
        try:
            # Ignorer les lignes vides
            if not step:
                return False
            
            # Vérifier les patterns à ignorer
            if any(re.search(pattern, step) for pattern in self.ignore_patterns):
                return False
            
            # Vérifier la présence d'un verbe d'action
            return any(verb in step for verb in self.action_verbs)
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation de l'étape: {e}")
            return False

    async def prepare_context(
        self,
        query: str,
        documents: List[Dict],
        max_fragments: int = 3
    ) -> Dict[str, Any]:
        """
        Prépare le contexte complet pour la génération de réponse.

        Args:
            query: La requête de l'utilisateur
            documents: Liste des documents trouvés par la recherche
            max_fragments: Nombre maximum de fragments à inclure

        Returns:
            Dictionnaire contenant le contexte enrichi
        """
        try:
            # Détection du type de requête
            query_type = self.detect_command_type(query, documents)
            
            # Filtrage initial des documents
            if query_type:
                filtered_docs = self.filter_documents_by_type(documents, query_type)
            else:
                filtered_docs = documents
            
            # Sélection des documents pertinents
            relevant_docs = self.filter_relevant_documents(filtered_docs)
            
            # Extraire les procédures
            procedures = []
            for doc in relevant_docs:
                steps = self.extract_procedure(doc.get('content', ''))
                if steps:
                    procedures.append({
                        'source': doc.get('title', 'Document inconnu'),
                        'page': doc.get('metadata', {}).get('page', 'N/A'),
                        'steps': steps,
                        'score': doc.get('score', 0)
                    })

            # Extraction des fragments avec images
            fragments = []
            for doc in relevant_docs[:3]:  # Limiter à 3 documents pour les fragments
                try:
                    doc_fragments = await self.doc_extractor.extract_relevant_fragments(
                        doc_path=doc.get('source'),
                        keywords=query.split(),
                        max_fragments=2  # 2 fragments max par document
                    )
                    fragments.extend(doc_fragments)
                except Exception as e:
                    logger.error(f"Erreur extraction fragments pour {doc.get('title')}: {e}")

            # Analyse des sujets présents dans les documents
            topics = self._extract_topics(relevant_docs)

            # Calcul du score de confiance global
            confidence_score = max(
                (doc.get('score', 0) for doc in relevant_docs),
                default=0.0
            )

            # Construction du contexte enrichi
            context = {
                'query': query,
                'query_type': query_type,
                'procedures': procedures,
                'relevant_docs': relevant_docs,
                'fragments': fragments,
                'topics': topics,
                'confidence_score': confidence_score,
                'metadata': {
                    'total_docs': len(documents),
                    'relevant_docs': len(relevant_docs),
                    'has_procedures': bool(procedures),
                    'has_fragments': bool(fragments),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }

            # Ajouter des suggestions si nécessaire
            if not procedures or confidence_score < self.confidence_threshold:
                context['suggestions'] = self._generate_suggestions(query, topics)

            return context

        except Exception as e:
            logger.error(f"Erreur lors de la préparation du contexte: {e}")
            # Retourner un contexte minimal en cas d'erreur
            return {
                'query': query,
                'query_type': None,
                'procedures': [],
                'relevant_docs': [],
                'fragments': [],
                'topics': [],
                'confidence_score': 0,
                'metadata': {
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }

    def _generate_suggestions(
        self,
        query: str,
        topics: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Génère des suggestions basées sur les topics trouvés.
        """
        suggestions = []
        
        # Regrouper les topics similaires
        for topic in topics:
            if topic['score'] > 0.3:  # Seuil minimal de pertinence
                suggestion = {
                    'topic': topic['keyword'],
                    'documents': [d['title'] for d in topic['docs'][:2]],
                    'confidence': topic['score'],
                    'sample_query': f"{query} {topic['keyword']}"
                }
                suggestions.append(suggestion)

        return sorted(
            suggestions,
            key=lambda x: x['confidence'],
            reverse=True
        )[:3]  # Limiter à 3 suggestions

    def _extract_topics(self, documents: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extrait les topics principaux des documents.
        """
        topic_data = {}
        
        for doc in documents:
            title = doc.get('title', '').lower()
            content = doc.get('content', '').lower()
            
            # Extraire les mots-clés significatifs
            keywords = self._extract_keywords(title, content)
            
            for keyword in keywords:
                if keyword not in topic_data:
                    topic_data[keyword] = {
                        'count': 0,
                        'docs': [],
                        'score': 0
                    }
                
                topic_info = topic_data[keyword]
                topic_info['count'] += 1
                topic_info['docs'].append(doc)
                topic_info['score'] += doc.get('score', 0)

        # Convertir en liste et normaliser les scores
        topics = []
        for keyword, data in topic_data.items():
            if data['count'] > 0:
                topics.append({
                    'keyword': keyword,
                    'count': data['count'],
                    'docs': sorted(
                        data['docs'],
                        key=lambda x: x.get('score', 0),
                        reverse=True
                    ),
                    'score': data['score'] / data['count']
                })

        return sorted(topics, key=lambda x: x['score'], reverse=True)
        """
        Prépare le contexte complet pour la génération de réponse.
        """
        try:
            # Détection du type de commande
            command_type = self.detect_command_type(query, documents)
            
            # Filtrage initial des documents
            if command_type:
                filtered_docs = self.filter_documents_by_type(documents, command_type)
            else:
                filtered_docs = documents
            
            # Sélection des documents pertinents
            relevant_docs = self.filter_relevant_documents(filtered_docs)
            
            # Extraction des procédures
            procedures = []
            for doc in relevant_docs:
                steps = self.extract_procedure(doc.get('content', ''))
                if steps:
                    procedures.append({
                        'source': doc.get('title', 'Document inconnu'),
                        'page': doc.get('metadata', {}).get('page', 'N/A'),
                        'steps': steps,
                        'score': doc.get('score', 0)
                    })
            
            return {
                'query': query,
                'command_type': command_type,
                'procedures': procedures,
                'confidence_score': max((doc.get('score', 0) for doc in relevant_docs), default=0)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la préparation du contexte: {e}")
            return {
                'query': query,
                'command_type': None,
                'procedures': [],
                'confidence_score': 0
            }

    def filter_documents_by_type(self, documents: List[Dict], command_type: str) -> List[Dict]:
        """
        Filtre les documents en fonction du type de commande.
        """
        try:
            if not command_type or command_type not in self.command_types:
                return documents
            
            config = self.command_types[command_type]
            relevant_docs = []
            
            for doc in documents:
                # Score de pertinence pour ce document
                relevance_score = 0
                doc_title = doc.get('title', '').lower()
                
                # Vérification du préfixe
                if doc_title.startswith(config['doc_prefix'].lower()):
                    relevance_score += 3
                
                # Vérification des patterns dans le titre
                for pattern in config['doc_patterns']:
                    if pattern.lower() in doc_title:
                        relevance_score += 2
                
                # Vérification des mots-clés dans le contenu
                content = doc.get('content', '').lower()
                for keyword in config['keywords']:
                    if keyword in content:
                        relevance_score += 1
                
                # Si le document est suffisamment pertinent
                if relevance_score >= 3:
                    doc['type_relevance'] = relevance_score
                    relevant_docs.append(doc)
            
            return sorted(relevant_docs, key=lambda x: x.get('type_relevance', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Erreur lors du filtrage par type: {e}")
            return documents

    def format_response(self, context: Dict) -> str:
        """
        Formate la réponse finale.
        """
        try:
            if not context['procedures']:
                if context['command_type']:
                    return f"Je n'ai pas trouvé de procédure spécifique pour créer une {self.command_types[context['command_type']]['description']}."
                return "Veuillez préciser le type de commande souhaité (achat, transport, client ou sous-traitance)."
            
            response_parts = []
            
            # En-tête avec type de commande
            if context['command_type']:
                cmd_desc = self.command_types[context['command_type']]['description']
                response_parts.append(f"Pour créer une {cmd_desc}, suivez ces étapes :")
            else:
                response_parts.append("Pour créer une commande, suivez ces étapes :")
            
            # Procédure principale
            procedure = context['procedures'][0]
            for i, step in enumerate(procedure['steps'], 1):
                # Étape principale
                response_parts.append(f"\n{i}. {step['text']}")
                
                # Sous-étapes éventuelles
                for substep in step.get('substeps', []):
                    response_parts.append(f"   • {substep}")
                
                # Détails importants
                important_details = [d for d in step.get('details', []) 
                                  if d.strip() and len(d) >= 10 
                                  and not any(p.lower() in d.lower() for p in self.ignore_patterns)]
                for detail in important_details[:2]:
                    response_parts.append(f"   → {detail}")
            
            # Source du document
            response_parts.append(f"\nSource : {procedure['source']} (Page {procedure['page']})")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Erreur lors du formatage de la réponse: {e}")
            return "Une erreur est survenue lors de la génération de la réponse."
    def __init__(self):
        """Initialise le processeur de contexte."""
        self.confidence_threshold = settings.CONTEXT_CONFIDENCE_THRESHOLD
        self.max_relevant_docs = settings.MAX_RELEVANT_DOCS
        self.action_verbs = settings.ACTION_VERBS
        
    def filter_relevant_documents(self, docs: List[Dict]) -> List[Dict]:
        """
        Filtre et nettoie les documents pertinents.
        """
        # Normaliser les scores sur une échelle de 0 à 1
        max_score = max((doc.get('score', 0) for doc in docs), default=0)
        if max_score > 0:
            for doc in docs:
                doc['score'] = doc.get('score', 0) / max_score

        # Filtrer les documents pertinents
        relevant_docs = [
            doc for doc in docs 
            if doc.get('score', 0) > self.confidence_threshold
        ]
        
        # Trier par pertinence et limiter
        relevant_docs = sorted(
            relevant_docs,
            key=lambda x: x.get('score', 0),
            reverse=True
        )[:self.max_relevant_docs]
        
        return relevant_docs

    def extract_procedure(self, content: str) -> List[Dict[str, Any]]:
        """
        Extrait les procédures techniques du contenu avec contexte enrichi.
        """
        sections = []
        current_section = {
            'title': '',
            'prerequisites': [],
            'steps': [],
            'notes': []
        }
        
        lines = content.split('\n')
        in_procedure = False
        in_prereq = False
        step_count = 0
        
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Détection des titres de section
            if any(keyword in line.lower() for keyword in settings.PROCEDURE_KEYWORDS):
                if current_section['steps']:
                    sections.append(current_section)
                current_section = {
                    'title': line,
                    'prerequisites': [],
                    'steps': [],
                    'notes': []
                }
                in_procedure = True
                continue
            
            # Détection des prérequis
            if any(word in line.lower() for word in ['prérequis', 'avant de commencer', 'conditions']):
                in_prereq = True
                in_procedure = False
                continue
                
            # Détection des notes
            if line.lower().startswith(('note', 'attention', 'important')):
                current_section['notes'].append(line)
                continue
            
            # Traitement des prérequis
            if in_prereq:
                if re.match(r'^[\-\•]\s+.+', line):
                    current_section['prerequisites'].append(line.strip('- •'))
                else:
                    in_prereq = False
            
            # Traitement des étapes
            if in_procedure:
                # Détection des étapes numérotées ou avec puces
                step_match = re.match(r'^(?:\d+[\.\)]\s*|\-\s*|\•\s*)(.+)$', line)
                if step_match:
                    step_text = step_match.group(1).strip()
                    # Recherche du contexte supplémentaire
                    context_lines = self._get_step_context(lines, idx)
                    step = {
                        'text': step_text,
                        'context': context_lines,
                        'substeps': []
                    }
                    current_section['steps'].append(step)
                    step_count += 1
                    continue
                
                # Détection des étapes commençant par un verbe d'action
                for verb in self.action_verbs:
                    if line.lower().startswith(verb):
                        context_lines = self._get_step_context(lines, idx)
                        step = {
                            'text': line,
                            'context': context_lines,
                            'substeps': []
                        }
                        current_section['steps'].append(step)
                        step_count += 1
                        break
                
                # Détection des sous-étapes
                if step_count > 0 and re.match(r'^\s+[\-\•]\s+.+', line):
                    current_section['steps'][-1]['substeps'].append(line.strip())
        
        # Ajouter la dernière section
        if current_section['steps']:
            sections.append(current_section)
        
        return sections

    def prepare_context(self, query: str, documents: List[Dict]) -> Dict:
        """Prépare le contexte enrichi pour la génération de réponse."""
        relevant_docs = self.filter_relevant_documents(documents)
        
        procedures = []
        for doc in relevant_docs:
            content = doc.get('content', '').strip()
            if not content:
                continue
                
            sections = self.extract_procedure(content)
            if sections:
                procedures.append({
                    'source': doc.get('title', 'Document inconnu'),
                    'page': doc.get('metadata', {}).get('page', 'N/A'),
                    'sections': sections,
                    'score': doc.get('score', 0)
                })
        
        return {
            'query': query,
            'procedures': procedures,
            'confidence_score': max((doc.get('score', 0) for doc in relevant_docs), default=0),
            'sources': [doc.get('title') for doc in relevant_docs]
        }

    def _get_step_context(self, lines: List[str], current_idx: int, context_size: int = 2) -> List[str]:
        """
        Récupère le contexte autour d'une étape.
        """
        context = []
        start_idx = max(0, current_idx - context_size)
        end_idx = min(len(lines), current_idx + context_size + 1)
        
        for idx in range(start_idx, end_idx):
            if idx == current_idx:
                continue
            line = lines[idx].strip()
            if line and not any(line.startswith(p) for p in ['•', '-', '1.', '2.']):
                context.append(line)
        
        return context

    def format_response(self, context: Dict) -> str:
        """
        Formate la réponse avec tous les détails disponibles.
        """
        if not context['procedures']:
            return settings.NO_PROCEDURE_MESSAGE
        
        response_parts = []
        
        # Utiliser la procédure la plus pertinente
        best_procedure = context['procedures'][0]
        sections = best_procedure['sections']
        
        if not sections:
            return settings.NO_PROCEDURE_MESSAGE
        
        # Formatage de la réponse
        action = context['query']
        response_parts.append(f"Pour {action}, suivez cette procédure :")
        
        # Ajout des prérequis si présents
        prereqs = sections[0].get('prerequisites', [])
        if prereqs:
            response_parts.append("\nPrérequis :")
            for prereq in prereqs:
                response_parts.append(f"• {prereq}")
        
        # Ajout des étapes principales
        response_parts.append("\nÉtapes :")
        for i, step in enumerate(sections[0]['steps'], 1):
            response_parts.append(f"{i}. {step['text']}")
            # Ajouter les sous-étapes
            for substep in step['substeps']:
                response_parts.append(f"   • {substep}")
            # Ajouter le contexte pertinent
            for context_line in step['context']:
                if context_line.strip():
                    response_parts.append(f"   → {context_line}")
        
        # Ajout des notes importantes
        notes = sections[0].get('notes', [])
        if notes:
            response_parts.append("\nNotes importantes :")
            for note in notes:
                response_parts.append(f"• {note}")
        
        # Ajout de la source
        response_parts.append(f"\nSource : {best_procedure['source']} (Page {best_procedure['page']})")
        
        return "\n".join(response_parts)

class GenerationMonitoring:
    """Classe pour suivre les performances de génération."""
    def __init__(self):
        self.start_time = time.time()
        self.steps = {}

    def step(self, name: str):
        now = time.time()
        duration = now - self.start_time
        self.steps[name] = duration
        logger.info(f"Étape '{name}' terminée en {duration:.2f}s")
        self.start_time = now

    def summary(self):
        total = sum(self.steps.values())
        details = [f"{k}: {v:.2f}s ({(v/total)*100:.1f}%)" for k, v in self.steps.items()]
        logger.info(f"Génération complète en {total:.2f}s\n" + "\n".join(details))

class ModelInference:
    def __init__(self):
        """Initialise le modèle d'inférence."""
        try:
            # Configuration CPU/GPU
            self.device = "cpu" if settings.USE_CPU_ONLY else "cuda"
            if settings.USE_CPU_ONLY:
                torch.set_num_threads(settings.MAX_THREADS)
            
            logger.info(f"Initialisation du modèle sur {self.device}")
            
            self._setup_model()
            self._setup_tokenizer()
            self.context_processor = ContextProcessor()
            
            logger.info("ModelInference initialisé avec succès")

        except Exception as e:
            logger.error(f"Erreur initialisation modèle: {e}", exc_info=True)
            self.cleanup()
            raise

    def _setup_model(self):
        """Configure le modèle avec chargement forcé."""
        try:
            logger.info(f"Début du chargement du modèle {settings.MODEL_NAME}")
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Nettoyage préventif
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Configuration du modèle
            model_kwargs = {
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": settings.LOW_CPU_MEM_USAGE,
                "cache_dir": settings.CACHE_DIR
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_NAME,
                **model_kwargs
            )
            
            # Force le chargement complet
            self.model = self.model.to(self.device)
            
            # Force l'initialisation des poids
            with torch.no_grad():
                dummy_input = torch.zeros((1, 10), dtype=torch.long).to(self.device)
                _ = self.model(dummy_input)
            
            # Force la synchronisation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Nettoyage final
            gc.collect()
            
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            memory_delta = memory_after - memory_before
            
            # Vérification du chargement
            model_size = sum(p.nelement() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            logger.info(f"Taille du modèle: {model_size:.2f} MB")
            logger.info(f"Utilisation mémoire: {memory_delta:.2f} MB")
            
            if memory_delta < model_size/2:
                raise RuntimeError("Chargement incomplet du modèle")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise

    def _setup_tokenizer(self):
        """Configure le tokenizer."""
        try:
            logger.info("Configuration du tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                use_fast=True,
                model_max_length=settings.MAX_INPUT_LENGTH,
                cache_dir=settings.CACHE_DIR
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Tokenizer configuré avec max_length: {settings.MAX_INPUT_LENGTH}")
            
        except Exception as e:
            logger.error(f"Erreur configuration tokenizer: {e}")
            raise

    async def generate_response(
        self,
        query: str,
        context_docs: List[Dict],
        conversation_history: Optional[Dict] = None,
        language: str = "fr"
    ) -> Union[str, Dict]:
        """Génère une réponse complète."""
        monitor = GenerationMonitoring()
        try:
            logger.info(f"Génération de réponse pour: {query}")
            monitor.step("Début")

            if not context_docs:
                return settings.NO_CONTEXT_MESSAGE

            # Préparation du contexte
            context = self.context_processor.prepare_context(query, context_docs)
            monitor.step("Préparation contexte")

            # Si on a une procédure claire, on l'utilise directement
            if context['procedures'] and context['confidence_score'] >= settings.DIRECT_RESPONSE_THRESHOLD:
                response = self.context_processor.format_response(context)
                monitor.step("Formatage réponse directe")
                return response

            # Sinon, on utilise le modèle LLM
            prompt = settings.CHAT_TEMPLATE.format(
                system=settings.SYSTEM_PROMPT,
                query=query,
                context=self._format_context_for_prompt(context)
            )
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=settings.MAX_INPUT_LENGTH,
                return_token_type_ids=False
            ).to(self.device)
            
            monitor.step("Préparation LLM")

            try:
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        **settings.GENERATION_CONFIG
                    )
                
                monitor.step("Génération LLM")

                response = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                # Extraction et validation de la réponse
                if "<|assistant|>" in response:
                    response = response.split("<|assistant|>")[-1].strip()
                
                monitor.step("Post-traitement")
                monitor.summary()

                # Validation finale
                if len(response) < settings.MIN_RESPONSE_LENGTH:
                    return settings.INVALID_RESPONSE_MESSAGE

                return response

            except Exception as e:
                logger.error(f"Erreur pendant la génération LLM: {str(e)}", exc_info=True)
                raise

        except Exception as e:
            logger.error(f"Erreur génération réponse: {str(e)}", exc_info=True)
            return settings.ERROR_MESSAGE

    def _format_context_for_prompt(self, context: Dict) -> str:
        """Formate le contexte pour le prompt du modèle."""
        formatted_parts = []
        
        for proc in context['procedures']:
            formatted_parts.append(f"Document: {proc['source']} (Page {proc['page']})")
            formatted_parts.append("Étapes:")
            formatted_parts.extend([f"- {step}" for step in proc['steps']])
            formatted_parts.append("")
        
        return "\n".join(formatted_parts)
        
    async def generate_streaming_response(
        self,
        query: str,
        context_docs: List[Dict],
        conversation_history: Optional[Dict] = None,
        language: str = "fr"
    ) -> AsyncIterator[str]:
        """Génère une réponse en streaming."""
        try:
            logger.info(f"Début génération streaming pour: {query}")
            
            # Préparation comme pour la réponse normale
            context = self.context_processor.prepare_context(query, context_docs)
            
            # Si on a une procédure claire, on la retourne en streaming simulé
            if context['procedures'] and context['confidence_score'] >= settings.DIRECT_RESPONSE_THRESHOLD:
                response = self.context_processor.format_response(context)
                for line in response.split('\n'):
                    words = line.split()
                    for word in words:
                        yield word + " "
                        await asyncio.sleep(settings.STREAM_DELAY)
                    yield "\n"
                return

            # Sinon, on utilise le modèle en streaming
            prompt = settings.CHAT_TEMPLATE.format(
                system=settings.SYSTEM_PROMPT,
                query=query,
                context=self._format_context_for_prompt(context)
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=settings.MAX_INPUT_LENGTH,
                return_token_type_ids=False
            ).to(self.device)

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            generation_kwargs = {
                **inputs,
                **settings.GENERATION_CONFIG,
                "streamer": streamer
            }

            # Lancer la génération dans un thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Streaming des tokens avec gestion du contexte
            buffer = ""
            response_started = False
            async for token in self._async_tokens(streamer):
                # Détecter le début de la réponse
                if "<|assistant|>" in token:
                    response_started = True
                    token = token.split("<|assistant|>")[-1]
                
                if response_started:
                    buffer += token
                    # Envoi par mots ou phrases complètes
                    if any(c in buffer for c in ".,!?\n") or len(buffer) >= settings.STREAM_CHUNK_SIZE:
                        yield buffer
                        buffer = ""
                    
                await asyncio.sleep(settings.STREAM_DELAY)

            # Envoi du buffer restant s'il y en a
            if buffer:
                yield buffer

        except Exception as e:
            logger.error(f"Erreur dans le streaming: {str(e)}", exc_info=True)
            yield settings.STREAM_ERROR_MESSAGE

    async def _async_tokens(self, streamer: TextIteratorStreamer) -> AsyncIterator[str]:
        """
        Convertit le streamer synchrone en générateur asynchrone.
        """
        try:
            for token in streamer:
                yield token
        except Exception as e:
            logger.error(f"Erreur dans le streaming de tokens: {e}")
            raise

    def cleanup(self):
        """Nettoie les ressources."""
        try:
            if hasattr(self, 'model'):
                self.model.cpu()  # Déplacer sur CPU avant suppression
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Ressources nettoyées avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")

    def __del__(self):
        """Destructeur de la classe."""
        self.cleanup()