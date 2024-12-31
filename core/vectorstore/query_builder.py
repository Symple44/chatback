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
        search_body = {
            "size": size,
            "min_score": min_score,
            "_source": ["title", "content", "application", "metadata"],
            "timeout": "30s",
            "query": {
                "bool": {
                    "must": self._build_text_query(query),
                    "should": self._build_vector_query(vector) if vector else [],
                    "filter": self._build_filters(metadata_filter),
                    "minimum_should_match": 1
                }
            }
        }

        if highlight:
            search_body["highlight"] = self._build_highlight_config()

        return search_body

    def _build_text_query(self, query: str) -> List[Dict]:
        return [{
            "multi_match": {
                "query": query,
                "fields": ["title^2", "content", "metadata.*"],
                "type": "best_fields",
                "operator": "or",
                "minimum_should_match": "75%",
                "fuzziness": "AUTO",
                "tie_breaker": 0.3
            }
        }]

    def _build_vector_query(self, vector: List[float]) -> List[Dict]:
        if not vector or len(vector) != self.embedding_dim:
            return []

        return [{
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": vector}
                },
                "min_score": 0.5
            }
        }]

    def _build_filters(self, metadata_filter: Optional[Dict]) -> List[Dict]:
        filters = []
        
        if metadata_filter:
            for key, value in metadata_filter.items():
                if isinstance(value, (list, tuple)):
                    filters.append({
                        "terms": {f"metadata.{key}.keyword": value}
                    })
                elif isinstance(value, (int, float)):
                    filters.append({
                        "range": {
                            f"metadata.{key}": {"gte": value}
                        }
                    })
                elif isinstance(value, dict):
                    range_filter = {}
                    if "gte" in value:
                        range_filter["gte"] = value["gte"]
                    if "lte" in value:
                        range_filter["lte"] = value["lte"]
                    if range_filter:
                        filters.append({
                            "range": {f"metadata.{key}": range_filter}
                        })
                else:
                    filters.append({
                        "term": {f"metadata.{key}.keyword": value}
                    })

        # Ajout de filtres temporels si nécessaire
        filters.append({
            "range": {
                "timestamp": {"lte": "now"}
            }
        })

        return filters

    def _build_highlight_config(self) -> Dict:
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

    def build_bulk_query(self, documents: List[Dict]) -> List[Dict]:
        actions = []
        for doc in documents:
            action = {
                "_index": f"{self.index_prefix}_documents",
                "_source": {
                    "title": doc["title"],
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "embedding": doc.get("vector"),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            if "id" in doc:
                action["_id"] = doc["id"]
            actions.append(action)
        return actions

    def build_aggregation_query(
        self,
        aggs: Dict[str, Any],
        query: Optional[Dict] = None
    ) -> Dict:
        body = {"size": 0, "aggs": aggs}
        if query:
            body["query"] = query
        return body
