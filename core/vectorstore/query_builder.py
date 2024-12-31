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

        # Construction de la requête bool
        bool_query = {
            "must": [],
            "should": [],
            "filter": [],
            "minimum_should_match": 1
        }

        # Ajout de la recherche textuelle
        bool_query["must"].append({
            "multi_match": {
                "query": query,
                "fields": ["title^2", "content"],
                "type": "best_fields",
                "operator": "or",
                "tie_breaker": 0.3,
                "fuzziness": "AUTO",
                "prefix_length": 2
            }
        })

        # Ajout de la recherche vectorielle si un vecteur est fourni
        if vector and len(vector) == self.embedding_dim:
            vector_query = {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": [{
                                "exists": {
                                    "field": "embedding"
                                }
                            }]
                        }
                    },
                    "script": {
                        "lang": "painless",
                        "source": """
                        if (doc['embedding'].size() == 0) { return 0; }
                        return cosineSimilarity(params.query_vector, doc['embedding']) + 1.0;
                        """,
                        "params": {
                            "query_vector": vector
                        }
                    },
                    "boost_mode": "sum"
                }
            }
            bool_query["should"].append(vector_query)

        # Ajout des filtres de métadonnées
        if metadata_filter:
            for key, value in metadata_filter.items():
                filter_field = f"metadata.{key}"
                if isinstance(value, (list, tuple)):
                    bool_query["filter"].append({
                        "terms": {f"{filter_field}.keyword": value}
                    })
                elif isinstance(value, dict):
                    range_filter = {}
                    if "gte" in value:
                        range_filter["gte"] = value["gte"]
                    if "lte" in value:
                        range_filter["lte"] = value["lte"]
                    if range_filter:
                        bool_query["filter"].append({
                            "range": {filter_field: range_filter}
                        })
                else:
                    bool_query["filter"].append({
                        "term": {f"{filter_field}.keyword": str(value)}
                    })

        search_body["query"] = {"bool": bool_query}

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
