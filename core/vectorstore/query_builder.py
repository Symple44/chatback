# core/vectorstore/query_builder.py
from typing import List, Dict, Optional, Any
from datetime import datetime
from core.config import settings

class QueryBuilder:
    def __init__(self):
        self.embedding_dim = settings.ELASTICSEARCH_EMBEDDING_DIM
        self.index_prefix = settings.ELASTICSEARCH_INDEX_PREFIX

    def build_search_query(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        metadata_filter: Optional[Dict] = None,
        size: int = 5,
        min_score: float = 0.1,
        highlight: bool = True
    ) -> Dict[str, Any]:
        """Construit la requête de recherche."""
        search_body = {
            "size": size,
            "min_score": min_score,
            "_source": ["title", "content", "metadata"],
            "timeout": "30s"
        }

        # Construction de la requête principale
        main_query = {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "content"],
                "type": "best_fields",
                "operator": "or",
                "tie_breaker": 0.3,
                "fuzziness": "AUTO",
                "prefix_length": 2
            }
        }

        # Si un vecteur est fourni, utiliser function_score
        if vector and len(vector) == self.embedding_dim:
            final_query = {
                "function_score": {
                    "query": main_query,
                    "functions": [{
                        "script_score": {
                            "script": {
                                "source": """
                                if (!doc.containsKey('embedding') || doc['embedding'].empty) { 
                                    return 0.0; 
                                }
                                double cosine = cosineSimilarity(params.query_vector, 'embedding');
                                return cosine;
                                """,
                                "params": {
                                    "query_vector": vector
                                }
                            }
                        },
                        "weight": 0.5
                    }],
                    "score_mode": "sum",
                    "boost_mode": "multiply"
                }
            }
        else:
            final_query = main_query

        # Ajout des filtres de métadonnées si présents
        if metadata_filter:
            filter_clauses = []
            for key, value in metadata_filter.items():
                filter_field = f"metadata.{key}"
                if isinstance(value, (list, tuple)):
                    filter_clauses.append({
                        "terms": {f"{filter_field}.keyword": value}
                    })
                elif isinstance(value, dict):
                    range_filter = {}
                    if "gte" in value:
                        range_filter["gte"] = value["gte"]
                    if "lte" in value:
                        range_filter["lte"] = value["lte"]
                    if range_filter:
                        filter_clauses.append({
                            "range": {filter_field: range_filter}
                        })
                else:
                    filter_clauses.append({
                        "term": {f"{filter_field}.keyword": str(value)}
                    })

            if filter_clauses:
                final_query = {
                    "bool": {
                        "must": [final_query],
                        "filter": filter_clauses
                    }
                }

        search_body["query"] = final_query

        # Configuration du highlighting
        if highlight:
            search_body["highlight"] = self._build_highlight_config()

        return search_body

    def _build_highlight_config(self) -> Dict:
        """Construit la configuration du highlighting."""
        return {
            "fields": {
                "content": {
                    "type": "unified",
                    "fragment_size": 150,
                    "number_of_fragments": 3,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                },
                "title": {
                    "number_of_fragments": 0
                }
            },
            "require_field_match": False,
            "max_analyzed_offset": 1000000
        }
