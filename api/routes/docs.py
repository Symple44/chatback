# apo/routes/docs.py
"""
Documentation des endpoints API avec des exemples.
À importer dans main.py pour enrichir la documentation FastAPI.
"""

# Descriptions des tags pour grouper les endpoints
tags_metadata = [
    {
        "name": "chat",
        "description": "Opérations liées au chat, incluant les conversations standard et streaming",
    },
    {
        "name": "history",
        "description": "Gestion de l'historique des conversations",
    },
    {
        "name": "monitoring",
        "description": "Endpoints de monitoring et statistiques système",
    }
]

# Descriptions détaillées pour chaque endpoint
chat_description = """
Envoie une requête au chatbot et reçoit une réponse.

La réponse peut être :
* Standard (réponse complète en une fois)
* Streaming (réponse progressive)

Le contexte de conversation peut être fourni pour maintenir la cohérence des échanges.
"""

streaming_description = """
Version streaming du chat qui retourne la réponse progressivement.

Utilise Server-Sent Events (SSE) pour envoyer les tokens au fur et à mesure.
Chaque événement contient un token de la réponse.
"""

history_description = """
Récupère l'historique des conversations pour un utilisateur donné.

Peut être filtré par :
* ID de session
* Limite de résultats
"""

health_description = """
Vérifie l'état de santé des différents composants du système :
* Base de données
* Elasticsearch
* Redis
* Modèle LLM
"""

# Exemples de réponses pour la documentation
example_responses = {
    "chat": {
        "response": "Bonjour! Comment puis-je vous aider?",
        "session_id": "abc123",
        "conversation_id": "xyz789",
        "documents_used": [
            {
                "title": "Guide d'introduction",
                "page": "Page 1",
                "relevance": 0.95
            }
        ],
        "confidence_score": 0.85,
        "processing_time": 1.23
    },
    "history": [
        {
            "query": "Bonjour",
            "response": "Bonjour! Comment puis-je vous aider?",
            "timestamp": "2024-12-07T10:30:00",
            "session_id": "abc123"
        }
    ],
    "health": {
        "status": "healthy",
        "components": {
            "database": True,
            "elasticsearch": True,
            "redis": True,
            "model": True
        },
        "memory": {
            "used_percent": 65.5,
            "available_gb": 8.2
        },
        "uptime_seconds": 3600
    }
}
