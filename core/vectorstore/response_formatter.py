# core/vectorstore/response_formatter.py
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger(__name__)

class ResponseFormatter:
    def __init__(self):
        """Initialisation du formateur de réponses."""
        self.max_content_length = 1000
        self.max_highlight_fragments = 3
        self.score_precision = 4
        
        # Configuration du scoring
        self.score_weights = {
            "bm25": 0.4,
            "vector": 0.4,
            "importance": 0.1,
            "recency": 0.1
        }

    def format_search_results(self, response: Dict) -> List[Dict]:
        """
        Formate les résultats de recherche Elasticsearch.
        
        Args:
            response: Réponse brute d'Elasticsearch
            
        Returns:
            Liste des résultats formatés
        """
        try:
            results = []
            hits = response.get("hits", {})
            total = hits.get("total", {})
            aggregations = response.get("aggregations", {})
            
            # Métriques de recherche
            metrics.set_gauge(
                "search_total_hits", 
                total.get("value", 0) if isinstance(total, dict) else total
            )
            metrics.set_gauge("search_max_score", hits.get("max_score", 0))
            
            # Traitement des hits
            max_score = hits.get("max_score", 1.0) or 1.0
            for hit in hits.get("hits", []):
                try:
                    result = self._format_hit(hit, max_score)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Erreur formatage hit: {e}")
                    continue
                    
            # Ajout des agrégations si présentes
            if aggregations:
                results = self._add_aggregation_data(results, aggregations)
            
            # Tri final
            results.sort(key=lambda x: x["final_score"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur formatage résultats: {e}")
            return []

    def _format_hit(self, hit: Dict, max_score: float) -> Optional[Dict]:
        """
        Formate un hit individuel avec normalisation des scores.
        
        Args:
            hit: Hit Elasticsearch
            max_score: Score maximum pour normalisation
        """
        try:
            source = hit.get("_source", {})
            
            # Extraction et normalisation des différents scores
            base_score = hit.get("_score", 0)
            importance_score = source.get("importance_score", 0.5)
            
            # Calcul du score final pondéré
            final_score = self._compute_weighted_score(
                base_score=base_score,
                max_score=max_score,
                importance=importance_score,
                processed_date=source.get("processed_date")
            )
            
            # Construction du résultat
            result = {
                "id": hit.get("_id"),
                "title": source.get("title", ""),
                "content": self._truncate_content(source.get("content", "")),
                "metadata": self._enrich_metadata(source.get("metadata", {}), hit),
                "scores": {
                    "base": round(base_score / max_score, self.score_precision),
                    "importance": round(importance_score, self.score_precision),
                    "final": round(final_score, self.score_precision)
                },
                "final_score": final_score,
                "highlights": self._format_highlights(hit.get("highlight", {})),
                "processed_date": source.get("processed_date")
            }
            
            # Ajout des champs optionnels
            if "inner_hits" in hit:
                result["inner_hits"] = self._format_inner_hits(hit["inner_hits"])
                
            if "fields" in hit:
                result["fields"] = hit["fields"]
                
            # Extraction des suggestions si présentes
            if "suggest" in hit:
                result["suggestions"] = self._format_suggestions(hit["suggest"])
                
            return result
            
        except Exception as e:
            logger.error(f"Erreur formatage hit individuel: {e}")
            return None

    def _compute_weighted_score(
        self,
        base_score: float,
        max_score: float,
        importance: float,
        processed_date: Optional[str] = None
    ) -> float:
        """
        Calcule un score final pondéré.
        
        Args:
            base_score: Score de base ES
            max_score: Score maximum pour normalisation
            importance: Score d'importance du document
            processed_date: Date de traitement pour facteur fraîcheur
        """
        try:
            # Normalisation des scores
            normalized_base = base_score / max_score
            
            # Calcul du facteur de fraîcheur
            recency_factor = 1.0
            if processed_date:
                try:
                    processed_dt = datetime.fromisoformat(processed_date.replace('Z', '+00:00'))
                    age_days = (datetime.utcnow() - processed_dt).days
                    recency_factor = max(0.5, 1.0 - (age_days / 365))  # Décroissance sur 1 an
                except ValueError:
                    pass

            # Calcul du score final pondéré
            final_score = (
                self.score_weights["bm25"] * normalized_base +
                self.score_weights["importance"] * importance +
                self.score_weights["recency"] * recency_factor
            )
            
            return min(final_score, 1.0)  # Plafonnement à 1.0
            
        except Exception as e:
            logger.error(f"Erreur calcul score: {e}")
            return normalized_base

    def _enrich_metadata(self, metadata: Dict, hit: Dict) -> Dict:
        """Enrichit les métadonnées avec des informations additionnelles."""
        metadata.update({
            "index": hit.get("_index"),
            "shard": {
                "index": hit.get("_index"),
                "node": hit.get("_node"),
                "shard": hit.get("_shard")
            } if "_shard" in hit else None,
            "last_updated": datetime.utcnow().isoformat()
        })
        return metadata

    def _format_highlights(self, highlights: Dict) -> Dict[str, List[str]]:
        """
        Formate les fragments de highlighting.
        
        Args:
            highlights: Highlights bruts d'ES
        """
        formatted = {}
        for field, fragments in highlights.items():
            if isinstance(fragments, list):
                # Limitation du nombre de fragments
                formatted[field] = fragments[:self.max_highlight_fragments]
            else:
                formatted[field] = [str(fragments)]
        return formatted

    def _truncate_content(self, content: str) -> str:
        """Tronque le contenu à la longueur maximale."""
        if len(content) > self.max_content_length:
            return f"{content[:self.max_content_length]}..."
        return content

    def _add_aggregation_data(self, results: List[Dict], aggregations: Dict) -> List[Dict]:
        """
        Ajoute les données d'agrégation aux résultats.
        
        Args:
            results: Résultats de recherche
            aggregations: Agrégations ES
        """
        for result in results:
            result["aggregations"] = {
                "tags": self._format_bucket_aggregation(
                    aggregations.get("metadata_tags", {}).get("buckets", [])
                ),
                "applications": self._format_bucket_aggregation(
                    aggregations.get("applications", {}).get("buckets", [])
                ),
                "importance": aggregations.get("importance_ranges", {}).get("buckets", [])
            }
        return results

    def _format_bucket_aggregation(self, buckets: List[Dict]) -> List[Dict]:
        """
        Formate les buckets d'agrégation.
        
        Args:
            buckets: Buckets d'agrégation ES
        """
        return [
            {
                "key": bucket["key"],
                "doc_count": bucket["doc_count"],
                "score": round(
                    bucket.get("score", {}).get("value", 0), 
                    self.score_precision
                )
            }
            for bucket in buckets
        ]

    def _format_suggestions(self, suggest_data: Dict) -> List[Dict]:
        """
        Formate les suggestions de recherche.
        
        Args:
            suggest_data: Données de suggestion ES
        """
        suggestions = []
        for suggestion_type, values in suggest_data.items():
            for value in values:
                if "options" in value:
                    suggestions.extend([{
                        "text": option.get("text"),
                        "score": round(
                            option.get("score", 0), 
                            self.score_precision
                        ),
                        "freq": option.get("freq"),
                        "type": suggestion_type
                    } for option in value["options"]])
        return suggestions
