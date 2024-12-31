# core/vectorstore/response_formatter.py
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from core.utils.logger import get_logger
from core.utils.metrics import metrics

logger = get_logger(__name__)

class ResponseFormatter:
    def __init__(self):
        self.max_content_length = 1000
        self.max_highlight_fragments = 3
        self.score_precision = 4

    def format_search_results(self, response: Dict) -> List[Dict]:
        """Formate les résultats de recherche."""
        try:
            results = []
            hits = response.get("hits", {})
            total = hits.get("total", {})
            
            # Métriques de recherche
            metrics.set_gauge("search_total_hits", 
                total.get("value", 0) if isinstance(total, dict) else total)
            metrics.set_gauge("search_max_score", hits.get("max_score", 0))
            
            for hit in hits.get("hits", []):
                try:
                    result = self._format_hit(hit)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Erreur formatage hit: {e}")
                    continue
                    
            return results
            
        except Exception as e:
            logger.error(f"Erreur formatage résultats: {e}")
            return []

    def _format_hit(self, hit: Dict) -> Optional[Dict]:
        """Formate un résultat individuel."""
        try:
            source = hit.get("_source", {})
            
            result = {
                "id": hit.get("_id"),
                "title": source.get("title", ""),
                "content": self._truncate_content(source.get("content", "")),
                "score": round(hit.get("_score", 0), self.score_precision),
                "metadata": self._enrich_metadata(source.get("metadata", {}), hit),
                "timestamp": source.get("timestamp"),
                "highlights": self._format_highlights(hit.get("highlight", {})),
                "vectors": self._extract_vectors(source)
            }
            
            # Ajout optionnel des champs spéciaux
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

    def _truncate_content(self, content: str) -> str:
        """Tronque le contenu à la longueur maximale."""
        if len(content) > self.max_content_length:
            return f"{content[:self.max_content_length]}..."
        return content

    def _enrich_metadata(self, metadata: Dict, hit: Dict) -> Dict:
        """Enrichit les métadonnées avec des informations additionnelles."""
        metadata.update({
            "index": hit.get("_index"),
            "type": hit.get("_type", "_doc"),
            "matched_queries": hit.get("matched_queries", []),
            "sort_values": hit.get("sort", []),
            "shard": {
                "index": hit.get("_index"),
                "node": hit.get("_node"),
                "shard": hit.get("_shard")
            } if "_shard" in hit else None
        })
        return metadata

    def _format_highlights(self, highlights: Dict) -> Dict[str, List[str]]:
        """Formate les fragments de highlight."""
        formatted = {}
        for field, fragments in highlights.items():
            if isinstance(fragments, list):
                formatted[field] = fragments[:self.max_highlight_fragments]
            else:
                formatted[field] = [str(fragments)]
        return formatted

    def _extract_vectors(self, source: Dict) -> Optional[Dict[str, List[float]]]:
        """Extrait les vecteurs du document."""
        vectors = {}
        for key, value in source.items():
            if key.endswith("_vector") and isinstance(value, list):
                vectors[key] = value
        return vectors if vectors else None

    def _format_inner_hits(self, inner_hits: Dict) -> Dict[str, List[Dict]]:
        """Formate les inner_hits."""
        formatted = {}
        for name, data in inner_hits.items():
            hits = data.get("hits", {}).get("hits", [])
            formatted[name] = [self._format_hit(hit) for hit in hits]
        return formatted

    def _format_suggestions(self, suggest: Dict) -> List[Dict]:
        """Formate les suggestions de recherche."""
        suggestions = []
        for suggestion_type, values in suggest.items():
            for value in values:
                if "options" in value:
                    suggestions.extend([{
                        "text": option.get("text"),
                        "score": option.get("score"),
                        "freq": option.get("freq"),
                        "type": suggestion_type
                    } for option in value["options"]])
        return suggestions

    def format_aggregation_results(self, response: Dict) -> Dict[str, Any]:
        """Formate les résultats d'agrégation."""
        try:
            return self._format_aggs(response.get("aggregations", {}))
        except Exception as e:
            logger.error(f"Erreur formatage agrégations: {e}")
            return {}

    def _format_aggs(self, aggs: Dict) -> Dict[str, Any]:
        """Formate récursivement les agrégations."""
        result = {}
        for name, data in aggs.items():
            if "buckets" in data:
                result[name] = self._format_buckets(data["buckets"])
            elif "value" in data:
                result[name] = data["value"]
            elif "values" in data:
                result[name] = data["values"]
            else:
                sub_aggs = {k: v for k, v in data.items() 
                          if isinstance(v, dict)}
                if sub_aggs:
                    result[name] = self._format_aggs(sub_aggs)
        return result

    def _format_buckets(self, buckets: Union[List[Dict], Dict]) -> List[Dict]:
        """Formate les buckets d'agrégation."""
        if isinstance(buckets, dict):
            return [{
                "key": key,
                "doc_count": data.get("doc_count", 0),
                "aggs": self._format_aggs(data) if isinstance(data, dict) else data
            } for key, data in buckets.items()]
            
        return [{
            "key": bucket.get("key"),
            "doc_count": bucket.get("doc_count", 0),
            "aggs": self._format_aggs(bucket) if isinstance(bucket, dict) else bucket
        } for bucket in buckets]
