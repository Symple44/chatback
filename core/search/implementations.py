# core/search/implementations.py
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from datetime import datetime
import asyncio
from core.utils.logger import get_logger
from core.utils.metrics import metrics
from core.config.search_config import (
    SEARCH_STRATEGIES_CONFIG,
    SEARCH_LIMITS,
    ERROR_HANDLING
)
from .strategies import SearchStrategy, SearchResult

logger = get_logger("search_implementations")

class EnhancedRAGSearch(SearchStrategy):
    """Implémentation améliorée de la recherche RAG."""
    
    def __init__(self, components):
        super().__init__(components)
        self.config = SEARCH_STRATEGIES_CONFIG["rag"]
        self._cache = {}  # Cache simple pour les embeddings

    async def search(
        self,
        query: str,
        metadata_filter: Optional[Dict] = None,
        **kwargs
    ) -> List[SearchResult]:
        try:
            # 1. Création de l'embedding de la requête
            query_vector = await self._get_query_embedding(query)
            if not query_vector:
                return []

            # 2. Recherche vectorielle
            results = await self._vector_search(
                query=query,
                query_vector=query_vector,
                metadata_filter=metadata_filter,
                **kwargs
            )

            # 3. Post-traitement des résultats
            processed_results = await self._post_process_results(
                results=results,
                query_vector=query_vector,
                **kwargs
            )

            # 4. Mise à jour des métriques
            self._update_metrics(len(processed_results))

            return processed_results

        except Exception as e:
            logger.error(f"Erreur recherche RAG: {e}")
            if ERROR_HANDLING["fallback_to_simple"]:
                return await self._fallback_search(query, metadata_filter)
            return []

    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Récupère ou crée l'embedding de la requête."""
        cache_key = f"emb_{hash(query)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = await self.create_embedding(query)
        if embedding:
            self._cache[cache_key] = embedding
        return embedding

    async def _vector_search(
        self,
        query: str,
        query_vector: List[float],
        metadata_filter: Optional[Dict],
        **kwargs
    ) -> List[Dict]:
        """Effectue la recherche vectorielle."""
        try:
            docs = await self.es_client.search_documents(
                query=query,
                vector=query_vector,
                metadata_filter=metadata_filter,
                size=kwargs.get("max_docs", self.config["search_params"]["max_docs"]),
                min_score=kwargs.get("min_score", self.config["search_params"]["min_score"])
            )
            return docs
        except Exception as e:
            logger.error(f"Erreur recherche vectorielle: {e}")
            return []

    async def _post_process_results(
        self,
        results: List[Dict],
        query_vector: List[float],
        **kwargs
    ) -> List[SearchResult]:
        """Post-traite les résultats de recherche."""
        processed_results = []
        
        for result in results:
            try:
                # Calcul du score combiné
                vector_score = result.get("score", 0)
                semantic_score = self._calculate_semantic_score(
                    query_vector,
                    result.get("embedding", [])
                )
                
                # Score final pondéré
                final_score = (
                    self.config["search_params"]["vector_weight"] * vector_score +
                    self.config["search_params"]["semantic_weight"] * semantic_score
                )

                # Création du SearchResult
                processed_results.append(SearchResult(
                    content=result.get("content", ""),
                    score=final_score,
                    metadata={
                        **result.get("metadata", {}),
                        "vector_score": vector_score,
                        "semantic_score": semantic_score,
                        "processing_info": {
                            "method": "rag",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    },
                    source_id=result.get("_id"),
                    embedding=result.get("embedding")
                ))

            except Exception as e:
                logger.error(f"Erreur traitement résultat: {e}")
                continue

        return sorted(processed_results, key=lambda x: x.score, reverse=True)

    def _calculate_semantic_score(
        self,
        query_vector: List[float],
        doc_vector: List[float]
    ) -> float:
        """Calcule le score sémantique entre deux vecteurs."""
        if not query_vector or not doc_vector:
            return 0.0
            
        try:
            query_array = np.array(query_vector)
            doc_array = np.array(doc_vector)
            
            # Normalisation
            query_norm = query_array / np.linalg.norm(query_array)
            doc_norm = doc_array / np.linalg.norm(doc_array)
            
            # Similarité cosinus
            return float(np.dot(query_norm, doc_norm))
            
        except Exception as e:
            logger.error(f"Erreur calcul score sémantique: {e}")
            return 0.0

    async def _fallback_search(
        self,
        query: str,
        metadata_filter: Optional[Dict]
    ) -> List[SearchResult]:
        """Recherche de repli simple."""
        try:
            docs = await self.es_client.search_documents(
                query=query,
                metadata_filter=metadata_filter,
                size=5,
                min_score=0.1
            )
            
            return [
                SearchResult(
                    content=doc.get("content", ""),
                    score=doc.get("score", 0),
                    metadata=doc.get("metadata", {}),
                    source_id=doc.get("_id")
                )
                for doc in docs
            ]
            
        except Exception as e:
            logger.error(f"Erreur recherche fallback: {e}")
            return []

    def _update_metrics(self, result_count: int):
        """Met à jour les métriques de recherche."""
        metrics.increment_counter("rag_searches")
        metrics.increment_counter("documents_found", result_count)
        metrics.record_time(
            "rag_search",
            metrics.get_timer_value("search_time")
        )
class EnhancedHybridSearch(SearchStrategy):
    """Implémentation de la recherche hybride (RAG + sémantique)."""
    
    def __init__(self, components):
        super().__init__(components)
        self.config = SEARCH_STRATEGIES_CONFIG["hybrid"]
        self._cache = {}

    async def search(
        self,
        query: str,
        metadata_filter: Optional[Dict] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Effectue une recherche hybride combinant RAG et sémantique."""
        try:
            # 1. Recherches parallèles
            rag_task = self._rag_search(query, metadata_filter, **kwargs)
            semantic_task = self._semantic_search(query, metadata_filter, **kwargs)
            
            # Exécution parallèle
            rag_results, semantic_results = await asyncio.gather(rag_task, semantic_task)
            
            # 2. Fusion des résultats
            merged_results = await self._merge_results(
                rag_results=rag_results,
                semantic_results=semantic_results,
                query=query
            )
            
            # 3. Reranking des résultats
            final_results = await self._rerank_results(
                results=merged_results,
                query=query,
                **kwargs
            )
            
            # 4. Mise à jour des métriques
            self._update_metrics(len(final_results))
            
            return final_results

        except Exception as e:
            logger.error(f"Erreur recherche hybride: {e}")
            if ERROR_HANDLING["fallback_to_simple"]:
                return await self._fallback_search(query, metadata_filter)
            return []

    async def _semantic_search(
        self,
        query: str,
        metadata_filter: Optional[Dict],
        **kwargs
    ) -> List[SearchResult]:
        """Effectue la partie recherche sémantique."""
        try:
            docs = await self.es_client.search_documents(
                query=query,
                metadata_filter=metadata_filter,
                size=self.config["search_params"]["rerank_top_k"],
                min_score=self.config["search_params"]["min_score"]
            )

            return [
                SearchResult(
                    content=doc.get("content", ""),
                    score=doc.get("score", 0) * self.config["search_params"]["semantic_weight"],
                    metadata={
                        **doc.get("metadata", {}),
                        "search_type": "semantic"
                    },
                    source_id=doc.get("_id")
                )
                for doc in docs
            ]

        except Exception as e:
            logger.error(f"Erreur recherche sémantique: {e}")
            return []

    async def _merge_results(
        self,
        rag_results: List[SearchResult],
        semantic_results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """Fusionne les résultats des deux types de recherche."""
        try:
            # Création d'un dictionnaire pour dédupliquer par source_id
            merged_dict = {}
            
            # Traitement des résultats RAG
            for result in rag_results:
                if result.source_id:
                    merged_dict[result.source_id] = result
                    
            # Traitement des résultats sémantiques
            for result in semantic_results:
                if result.source_id in merged_dict:
                    # Fusion des scores si le document existe déjà
                    existing = merged_dict[result.source_id]
                    combine_method = self.config["search_params"]["combine_method"]
                    
                    if combine_method == "weighted_sum":
                        new_score = (
                            existing.score * self.config["search_params"]["rag_weight"] +
                            result.score * self.config["search_params"]["semantic_weight"]
                        )
                    elif combine_method == "max":
                        new_score = max(existing.score, result.score)
                    else:
                        new_score = (existing.score + result.score) / 2

                    # Mise à jour des métadonnées
                    merged_dict[result.source_id] = SearchResult(
                        content=existing.content,
                        score=new_score,
                        metadata={
                            **existing.metadata,
                            "search_type": "hybrid",
                            "rag_score": existing.score,
                            "semantic_score": result.score
                        },
                        source_id=result.source_id,
                        embedding=existing.embedding
                    )
                else:
                    merged_dict[result.source_id] = result

            # Conversion en liste et tri
            merged_results = list(merged_dict.values())
            return sorted(merged_results, key=lambda x: x.score, reverse=True)

        except Exception as e:
            logger.error(f"Erreur fusion résultats: {e}")
            return rag_results + semantic_results

    async def _rerank_results(
        self,
        results: List[SearchResult],
        query: str,
        **kwargs
    ) -> List[SearchResult]:
        """Réordonne les résultats en utilisant une fenêtre glissante."""
        try:
            if not self.config["search_params"]["use_sliding_window"]:
                return results[:kwargs.get("max_docs", self.config["search_params"]["max_docs"])]

            window_size = self.config["search_params"]["window_size"]
            reranked_results = []
            
            # Application de la fenêtre glissante
            for i in range(0, len(results), window_size):
                window = results[i:i + window_size]
                
                # Calcul des scores contextuels dans la fenêtre
                window_scores = await self._calculate_window_scores(window, query)
                
                # Tri de la fenêtre par score contextuel
                window = [w for w, _ in sorted(
                    zip(window, window_scores),
                    key=lambda x: x[1],
                    reverse=True
                )]
                
                reranked_results.extend(window)

            return reranked_results[:kwargs.get("max_docs", self.config["search_params"]["max_docs"])]

        except Exception as e:
            logger.error(f"Erreur reranking résultats: {e}")
            return results[:kwargs.get("max_docs", self.config["search_params"]["max_docs"])]

    async def _calculate_window_scores(
        self,
        window: List[SearchResult],
        query: str
    ) -> List[float]:
        """Calcule les scores contextuels pour une fenêtre de résultats."""
        try:
            scores = []
            query_embedding = await self.create_embedding(query)
            
            for result in window:
                # Score de base
                base_score = result.score
                
                # Score de cohérence avec la requête
                coherence_score = 0.0
                if query_embedding and result.embedding:
                    coherence_score = self._calculate_coherence(
                        query_embedding,
                        result.embedding
                    )
                
                # Score final
                final_score = (base_score * 0.7) + (coherence_score * 0.3)
                scores.append(final_score)
                
            return scores

        except Exception as e:
            logger.error(f"Erreur calcul scores fenêtre: {e}")
            return [result.score for result in window]

    def _calculate_coherence(
        self,
        query_embedding: List[float],
        doc_embedding: List[float]
    ) -> float:
        """Calcule la cohérence entre deux embeddings."""
        try:
            query_array = np.array(query_embedding)
            doc_array = np.array(doc_embedding)
            
            # Normalisation
            query_norm = query_array / np.linalg.norm(query_array)
            doc_norm = doc_array / np.linalg.norm(doc_array)
            
            # Similarité cosinus
            return float(np.dot(query_norm, doc_norm))
            
        except Exception as e:
            logger.error(f"Erreur calcul cohérence: {e}")
            return 0.0

    def _update_metrics(self, result_count: int):
        """Met à jour les métriques de recherche."""
        metrics.increment_counter("hybrid_searches")
        metrics.increment_counter("documents_found", result_count)
        metrics.record_time(
            "hybrid_search",
            metrics.get_timer_value("search_time")
        )
        
class EnhancedSemanticSearch(SearchStrategy):
    """Implémentation de la recherche purement sémantique."""
    
    def __init__(self, components):
        super().__init__(components)
        self.config = SEARCH_STRATEGIES_CONFIG["semantic"]
        self._concept_cache = {}
        self._query_expansions = {}

    async def search(
        self,
        query: str,
        metadata_filter: Optional[Dict] = None,
        **kwargs
    ) -> List[SearchResult]:
        try:
            # 1. Analyse de la requête et extraction des concepts
            query_analysis = await self._analyze_query(query)
            
            # 2. Expansion de la requête si configurée
            expanded_query = await self._expand_query(
                query=query,
                concepts=query_analysis["concepts"]
            ) if self.config["search_params"]["use_concepts"] else query

            # 3. Recherche avec la requête enrichie
            raw_results = await self.es_client.search_documents(
                query=expanded_query,
                metadata_filter=metadata_filter,
                size=kwargs.get("max_docs", self.config["search_params"]["max_docs"]) * 2,
                min_score=kwargs.get("min_score", self.config["search_params"]["min_score"])
            )

            # 4. Post-traitement sémantique
            processed_results = await self._semantic_post_process(
                raw_results=raw_results,
                query_analysis=query_analysis,
                **kwargs
            )

            # 5. Mise à jour des métriques
            self._update_metrics(len(processed_results))

            return processed_results

        except Exception as e:
            logger.error(f"Erreur recherche sémantique: {e}")
            if ERROR_HANDLING["fallback_to_simple"]:
                return await self._fallback_search(query, metadata_filter)
            return []

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyse sémantique approfondie de la requête."""
        try:
            cache_key = f"analysis_{hash(query)}"
            if cache_key in self._concept_cache:
                return self._concept_cache[cache_key]

            # Génération de l'analyse via le modèle
            analysis_prompt = f"""Analyse la requête suivante et extrait :
            - Les concepts clés
            - L'intention principale
            - Les contraintes éventuelles
            - Les relations sémantiques
            
            Requête : {query}"""

            response = await self.model.generate_response(
                query=analysis_prompt,
                max_tokens=150
            )

            # Parsing de la réponse
            analysis_text = response.get("response", "")
            
            # Extraction structurée
            analysis = {
                "concepts": self._extract_concepts(analysis_text),
                "intent": self._extract_intent(analysis_text),
                "constraints": self._extract_constraints(analysis_text),
                "semantic_relations": self._extract_relations(analysis_text),
                "timestamp": datetime.utcnow().isoformat()
            }

            # Mise en cache
            self._concept_cache[cache_key] = analysis
            return analysis

        except Exception as e:
            logger.error(f"Erreur analyse requête: {e}")
            return {
                "concepts": [],
                "intent": "unknown",
                "constraints": {},
                "semantic_relations": []
            }

    async def _expand_query(self, query: str, concepts: List[str]) -> str:
        """Enrichit la requête avec des concepts liés."""
        try:
            if not self.config["search_params"]["use_concepts"]:
                return query

            cache_key = f"expansion_{hash(query)}"
            if cache_key in self._query_expansions:
                return self._query_expansions[cache_key]

            # Limite du nombre de concepts
            max_concepts = self.config["search_params"]["max_concepts"]
            main_concepts = concepts[:max_concepts]

            # Construction de la requête enrichie
            expanded_parts = [query]
            
            # Ajout des concepts
            if main_concepts:
                concepts_str = " OR ".join(f'"{concept}"' for concept in main_concepts)
                expanded_parts.append(f"({concepts_str})")

            # Ajout de synonymes si configuré
            if self.config["search_params"]["consider_synonyms"]:
                synonyms = await self._get_synonyms(main_concepts)
                if synonyms:
                    synonyms_str = " OR ".join(f'"{syn}"' for syn in synonyms)
                    expanded_parts.append(f"({synonyms_str})")

            expanded_query = " ".join(expanded_parts)
            
            # Mise en cache
            self._query_expansions[cache_key] = expanded_query
            return expanded_query

        except Exception as e:
            logger.error(f"Erreur expansion requête: {e}")
            return query

    async def _semantic_post_process(
        self,
        raw_results: List[Dict],
        query_analysis: Dict,
        **kwargs
    ) -> List[SearchResult]:
        """Post-traitement sémantique des résultats."""
        try:
            processed_results = []
            max_docs = kwargs.get("max_docs", self.config["search_params"]["max_docs"])

            for result in raw_results:
                # Score de base
                base_score = result.get("score", 0)

                # Analyse du contenu du résultat
                content_analysis = await self._analyze_content(
                    content=result.get("content", ""),
                    query_analysis=query_analysis
                )

                # Calcul du score sémantique global
                semantic_score = self._calculate_semantic_score(
                    query_analysis=query_analysis,
                    content_analysis=content_analysis,
                    base_score=base_score
                )

                # Boost pour les correspondances exactes si configuré
                if self.config["search_params"]["boost_exact_matches"]:
                    semantic_score = self._apply_exact_match_boost(
                        score=semantic_score,
                        content=result.get("content", ""),
                        query=query_analysis["concepts"]
                    )

                # Création du SearchResult
                processed_results.append(SearchResult(
                    content=result.get("content", ""),
                    score=semantic_score,
                    metadata={
                        **result.get("metadata", {}),
                        "content_analysis": content_analysis,
                        "search_type": "semantic",
                        "base_score": base_score
                    },
                    source_id=result.get("_id")
                ))

            # Tri et limitation
            return sorted(
                processed_results,
                key=lambda x: x.score,
                reverse=True
            )[:max_docs]

        except Exception as e:
            logger.error(f"Erreur post-traitement sémantique: {e}")
            return []

    def _extract_concepts(self, analysis_text: str) -> List[str]:
        """Extrait les concepts depuis le texte d'analyse."""
        try:
            # Recherche de la section des concepts
            concept_match = re.search(
                r"concepts?:?\s*[-:]?\s*(.*?)(?=\n\n|\Z)",
                analysis_text,
                re.IGNORECASE | re.DOTALL
            )
            
            if concept_match:
                concepts_text = concept_match.group(1)
                # Extraction des concepts individuels
                concepts = [
                    concept.strip()
                    for concept in re.split(r'[,;]|\n-', concepts_text)
                    if concept.strip()
                ]
                return concepts[:self.config["search_params"]["max_concepts"]]
            
            return []
            
        except Exception as e:
            logger.error(f"Erreur extraction concepts: {e}")
            return []

    def _extract_intent(self, analysis_text: str) -> str:
        """Extrait l'intention depuis le texte d'analyse."""
        try:
            intent_match = re.search(
                r"intention:?\s*[-:]?\s*(.*?)(?=\n\n|\Z)",
                analysis_text,
                re.IGNORECASE | re.DOTALL
            )
            
            return intent_match.group(1).strip() if intent_match else "unknown"
            
        except Exception as e:
            logger.error(f"Erreur extraction intention: {e}")
            return "unknown"

    def _extract_constraints(self, analysis_text: str) -> Dict[str, Any]:
        """Extrait les contraintes depuis le texte d'analyse."""
        try:
            constraints = {}
            
            # Recherche de la section des contraintes
            constraint_match = re.search(
                r"contraintes?:?\s*[-:]?\s*(.*?)(?=\n\n|\Z)",
                analysis_text,
                re.IGNORECASE | re.DOTALL
            )
            
            if constraint_match:
                constraints_text = constraint_match.group(1)
                # Parsing des contraintes individuelles
                for line in constraints_text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        constraints[key.strip()] = value.strip()
                        
            return constraints
            
        except Exception as e:
            logger.error(f"Erreur extraction contraintes: {e}")
            return {}

    def _extract_relations(self, analysis_text: str) -> List[Dict[str, str]]:
        """Extrait les relations sémantiques depuis le texte d'analyse."""
        try:
            relations = []
            
            # Recherche de la section des relations
            relation_match = re.search(
                r"relations?:?\s*[-:]?\s*(.*?)(?=\n\n|\Z)",
                analysis_text,
                re.IGNORECASE | re.DOTALL
            )
            
            if relation_match:
                relations_text = relation_match.group(1)
                # Parsing des relations individuelles
                for line in relations_text.split('\n'):
                    if '->' in line:
                        source, target = line.split('->', 1)
                        relations.append({
                            "source": source.strip(),
                            "target": target.strip()
                        })
                        
            return relations
            
        except Exception as e:
            logger.error(f"Erreur extraction relations: {e}")
            return []

    async def _analyze_content(
        self,
        content: str,
        query_analysis: Dict
    ) -> Dict[str, Any]:
        """Analyse le contenu d'un résultat."""
        try:
            # Analyse similaire à celle de la requête
            analysis_prompt = f"""Analyse le contenu suivant et compare-le avec les concepts : {', '.join(query_analysis['concepts'])}
            
            Contenu : {content[:500]}"""  # Limitation pour éviter les tokens excessifs

            response = await self.model.generate_response(
                query=analysis_prompt,
                max_tokens=100
            )

            analysis_text = response.get("response", "")
            
            return {
                "concepts": self._extract_concepts(analysis_text),
                "relevance": self._calculate_relevance(
                    query_concepts=query_analysis["concepts"],
                    content_concepts=self._extract_concepts(analysis_text)
                ),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Erreur analyse contenu: {e}")
            return {
                "concepts": [],
                "relevance": 0.0
            }

    def _calculate_semantic_score(
        self,
        query_analysis: Dict,
        content_analysis: Dict,
        base_score: float
    ) -> float:
        """Calcule le score sémantique global."""
        try:
            # Poids des différents facteurs
            weights = {
                "base_score": 0.3,
                "concept_match": 0.4,
                "relevance": 0.3
            }

            # Score basé sur la correspondance des concepts
            concept_match = self._calculate_concept_match(
                query_concepts=query_analysis["concepts"],
                content_concepts=content_analysis.get("concepts", [])
            )

            # Score final pondéré
            final_score = (
                weights["base_score"] * base_score +
                weights["concept_match"] * concept_match +
                weights["relevance"] * content_analysis.get("relevance", 0)
            )

            return min(final_score, 1.0)  # Normalisation

        except Exception as e:
            logger.error(f"Erreur calcul score sémantique: {e}")
            return base_score

    def _calculate_concept_match(
        self,
        query_concepts: List[str],
        content_concepts: List[str]
    ) -> float:
        """Calcule le score de correspondance entre concepts."""
        try:
            if not query_concepts or not content_concepts:
                return 0.0

            # Normalisation des concepts
            query_set = {concept.lower() for concept in query_concepts}
            content_set = {concept.lower() for concept in content_concepts}

            # Intersection et union pour le coefficient de Jaccard
            intersection = len(query_set.intersection(content_set))
            union = len(query_set.union(content_set))

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.error(f"Erreur calcul correspondance concepts: {e}")
            return 0.0

    def _apply_exact_match_boost(
        self,
        score: float,
        content: str,
        query: List[str]
    ) -> float:
        """Applique un boost au score pour les correspondances exactes."""
        try:
            if not content or not query:
                return score

            content_lower = content.lower()
            exact_matches = sum(1 for q in query 
                              if q.lower() in content_lower)
            
            # Boost proportionnel au nombre de correspondances exactes
            boost_factor = 1.0 + (0.1 * exact_matches)  # 10% par correspondance
            return min(score * boost_factor, 1.0)  # Plafonné à 1.0

        except Exception as e:
            logger.error(f"Erreur application boost exact match: {e}")
            return score

    async def _get_synonyms(self, concepts: List[str]) -> List[str]:
        """Récupère les synonymes des concepts donnés."""
        try:
            if not concepts:
                return []

            # Construction du prompt pour obtenir des synonymes
            prompt = f"""Génère des synonymes ou termes connexes pour les concepts suivants :
            {', '.join(concepts)}
            Retourne uniquement les synonymes, un par ligne."""

            response = await self.model.generate_response(
                query=prompt,
                max_tokens=100
            )

            synonyms = []
            if response and response.get("response"):
                # Nettoyage et déduplication des synonymes
                synonyms = [
                    syn.strip()
                    for syn in response["response"].split("\n")
                    if syn.strip() and syn.strip() not in concepts
                ]
                
                # Limitation du nombre de synonymes
                return list(set(synonyms))[:5]  # Maximum 5 synonymes

            return synonyms

        except Exception as e:
            logger.error(f"Erreur récupération synonymes: {e}")
            return []

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calcule la distance de Levenshtein entre deux chaînes."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    async def _fallback_search(
        self,
        query: str,
        metadata_filter: Optional[Dict]
    ) -> List[SearchResult]:
        """Recherche de repli simple en cas d'erreur."""
        try:
            # Recherche simple sans analyse complexe
            docs = await self.es_client.search_documents(
                query=query,
                vector=query_vector,
                metadata_filter=metadata_filter,
                size=kwargs.get("max_docs", self.config["search_params"]["max_docs"]),
                min_score=kwargs.get("min_score", self.config["search_params"]["min_score"])
            )
            
            return [
                SearchResult(
                    content=doc.get("content", ""),
                    score=doc.get("score", 0),
                    metadata={
                        **doc.get("metadata", {}),
                        "search_type": "fallback"
                    },
                    source_id=doc.get("_id")
                )
                for doc in docs
            ]

        except Exception as e:
            logger.error(f"Erreur recherche fallback: {e}")
            return []

    def _update_metrics(self, result_count: int):
        """Met à jour les métriques de recherche."""
        metrics.increment_counter("semantic_searches")
        metrics.increment_counter("documents_found", result_count)
        metrics.record_time(
            "semantic_search",
            metrics.get_timer_value("search_time")
        )

    def _get_cache_key(self, *args) -> str:
        """Génère une clé de cache unique."""
        components = [str(arg) for arg in args]
        return f"cache_{'_'.join(components)}"