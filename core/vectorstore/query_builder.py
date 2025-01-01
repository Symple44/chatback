# core/vectorstore/query_builder.py
from typing import List, Dict, Optional, Any
from datetime import datetime
from core.config import settings

class QueryBuilder:
    def __init__(self):
        self.embedding_dim = settings.ELASTICSEARCH_EMBEDDING_DIM
        self.index_prefix = settings.ELASTICSEARCH_INDEX_PREFIX
        self.default_min_score = 0.1

    def build_search_query(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        metadata_filter: Optional[Dict] = None,
        size: int = 5,
        min_score: float = 0.1,
        highlight: bool = True
    ) -> Dict[str, Any]:
        """
        Construit une requête de recherche hybride.
        
        Args:
            query: Texte de la requête
            vector: Vecteur d'embedding optionnel
            metadata_filter: Filtres sur les métadonnées
            size: Nombre de résultats souhaités
            min_score: Score minimum pour les résultats
            highlight: Activer le highlighting
        """
        search_body = {
            "size": size,
            "min_score": min_score,
            "_source": {
                "includes": ["title", "content", "metadata", "importance_score", "processed_date"]
            },
            "query": {
                "bool": {
                    "must": [],
                    "should": [],
                    "filter": []
                }
            }
        }

        # 1. Recherche textuelle BM25 améliorée
        if query:
            search_body["query"]["bool"]["must"].append({
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "content^2", "metadata.*"],
                    "type": "most_fields",
                    "operator": "or",
                    "fuzziness": "AUTO",
                    "prefix_length": 2,
                    "minimum_should_match": "75%"
                }
            })

            # Boosters additionnels pour la pertinence
            search_body["query"]["bool"]["should"].extend([
                # Match exact dans le titre
                {
                    "match_phrase": {
                        "title": {
                            "query": query,
                            "boost": 2.0
                        }
                    }
                },
                # Termes dans l'ordre pour le contenu
                {
                    "match_phrase": {
                        "content": {
                            "query": query,
                            "boost": 1.5,
                            "slop": 2
                        }
                    }
                }
            ])

        # 2. Recherche vectorielle si vecteur fourni
        if vector and len(vector) == self.embedding_dim:
            script_score = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": """
                        double score = cosineSimilarity(params.query_vector, 'embedding') + 1.0;
                        // Bonus basé sur l'importance du document
                        if (doc['importance_score'].size() > 0) {
                            score *= (1.0 + doc['importance_score'].value * 0.2);
                        }
                        // Pénalité pour les documents anciens
                        if (doc['processed_date'].size() > 0) {
                            long age = Math.abs(ChronoUnit.DAYS.between(
                                doc['processed_date'].value.toInstant(),
                                params.now
                            ));
                            score *= Math.exp(-age / 365.0);
                        }
                        return score;
                        """,
                        "params": {
                            "query_vector": vector,
                            "now": datetime.utcnow().isoformat()
                        }
                    }
                }
            }
            search_body["query"]["bool"]["should"].append(script_score)

        # 3. Filtres sur les métadonnées
        if metadata_filter:
            for key, value in metadata_filter.items():
                if isinstance(value, (list, tuple)):
                    search_body["query"]["bool"]["filter"].append({
                        "terms": {f"metadata.{key}": value}
                    })
                elif isinstance(value, dict):
                    # Gestion des filtres de plage
                    range_filter = {}
                    if "gte" in value:
                        range_filter["gte"] = value["gte"]
                    if "lte" in value:
                        range_filter["lte"] = value["lte"]
                    if range_filter:
                        search_body["query"]["bool"]["filter"].append({
                            "range": {f"metadata.{key}": range_filter}
                        })
                else:
                    search_body["query"]["bool"]["filter"].append({
                        "term": {f"metadata.{key}": str(value)}
                    })

        # 4. Highlighting configuration
        if highlight:
            search_body["highlight"] = self._build_highlight_config()

        # 5. Agrégations pour les facettes
        search_body["aggs"] = self._build_aggregations()

        return search_body

    def _build_highlight_config(self) -> Dict:
        """Configuration du highlighting."""
        return {
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
            "fields": {
                "title": {
                    "number_of_fragments": 0,
                    "type": "unified"
                },
                "content": {
                    "fragment_size": 150,
                    "number_of_fragments": 3,
                    "type": "unified",
                    "phrase_limit": 512
                }
            },
            "require_field_match": False,
            "max_analyzed_offset": 1000000
        }

    def _build_aggregations(self) -> Dict:
        """Construction des agrégations pour les facettes."""
        return {
            "metadata_tags": {
                "terms": {
                    "field": "metadata.tags",
                    "size": 20
                }
            },
            "applications": {
                "terms": {
                    "field": "metadata.application",
                    "size": 20
                }
            },
            "importance_ranges": {
                "range": {
                    "field": "importance_score",
                    "ranges": [
                        {"to": 0.33},
                        {"from": 0.33, "to": 0.66},
                        {"from": 0.66}
                    ]
                }
            }
        }

    def build_suggestion_query(self, prefix: str, size: int = 5) -> Dict:
        """Construit une requête de suggestion."""
        return {
            "suggest": {
                "title_suggest": {
                    "prefix": prefix,
                    "completion": {
                        "field": "title.suggest",
                        "size": size,
                        "skip_duplicates": True,
                        "fuzzy": {
                            "fuzziness": "AUTO"
                        }
                    }
                }
            }
        }

    def build_vector_query(
        self,
        vector: List[float],
        metadata_filter: Optional[Dict] = None,
        min_score: float = 0.7,
        size: int = 5
    ) -> Dict:
        """Construit une requête purement vectorielle."""
        return self.build_search_query(
            query="",
            vector=vector,
            metadata_filter=metadata_filter,
            min_score=min_score,
            size=size,
            highlight=False
        )
