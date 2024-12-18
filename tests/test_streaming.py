# tests/test_streaming.py
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import pytest
import json
from main import app
import asyncio
from typing import AsyncIterator

# Mock du générateur de streaming
async def mock_stream_generator() -> AsyncIterator[str]:
    responses = [
        "Bonjour",
        " comment",
        " allez",
        "-vous",
        " ?"]
    for token in responses:
        yield token
        await asyncio.sleep(0.1)

# Mock du ModelInference pour les tests
class MockModelInference:
    async def generate_streaming_response(self, *args, **kwargs):
        async for token in mock_stream_generator():
            yield token

# Configuration des tests
@pytest.fixture
def test_client():
    # Patch des composants nécessaires
    with patch("api.routes.chat.get_components") as mock_components:
        # Configuration du mock des composants
        mock_components.return_value.model = MockModelInference()
        mock_components.return_value.es_client.search_documents = AsyncMock(
            return_value=[
                {
                    "title": "Test Doc",
                    "metadata": {"page": 1},
                    "score": 0.95
                }
            ]
        )
        
        # Création du client de test
        with TestClient(app) as client:
            yield client

@pytest.mark.asyncio
async def test_chat_stream_endpoint(test_client):
    """Test de l'endpoint de streaming."""
    
    # Données de la requête
    request_data = {
        "user_id": "test_user",
        "query": "Comment allez-vous ?",
        "language": "fr"
    }

    # Envoi de la requête
    response = test_client.post(
        "/chat/stream",
        json=request_data,
        stream=True
    )

    # Vérification du statut de la réponse
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"

    # Lecture et vérification du stream
    collected_response = ""
    events = []

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data: "):
                event_data = json.loads(decoded_line[6:])
                events.append(event_data)
                if "token" in event_data:
                    collected_response += event_data["token"]

    # Vérifications des événements reçus
    assert any(e.get("status") == "started" for e in events)
    assert any(e.get("status") == "completed" for e in events)
    assert collected_response.strip() == "Bonjour comment allez-vous ?"

@pytest.mark.asyncio
async def test_chat_stream_error_handling(test_client):
    """Test de la gestion des erreurs dans le streaming."""
    
    # Configuration du mock pour simuler une erreur
    with patch("api.routes.chat.get_components") as mock_components:
        mock_components.return_value.model.generate_streaming_response = AsyncMock(
            side_effect=Exception("Test error")
        )

        # Données de la requête
        request_data = {
            "user_id": "test_user",
            "query": "Test error",
            "language": "fr"
        }

        # Envoi de la requête
        response = test_client.post(
            "/chat/stream",
            json=request_data,
            stream=True
        )

        # Vérification de la réponse d'erreur
        assert response.status_code == 200  # Le stream commence toujours avec 200

        events = []
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    events.append(json.loads(decoded_line[6:]))

        # Vérification qu'un événement d'erreur a été envoyé
        assert any("error" in e for e in events)

@pytest.mark.asyncio
async def test_chat_stream_validation(test_client):
    """Test de la validation des données d'entrée."""
    
    # Test avec des données invalides
    invalid_request = {
        "user_id": "",  # ID utilisateur vide
        "query": "",    # Question vide
    }

    response = test_client.post(
        "/chat/stream",
        json=invalid_request
    )

    # Vérification que la validation échoue
    assert response.status_code == 422  # Unprocessable Entity
    
    # Vérification du message d'erreur
    error_detail = response.json()["detail"]
    assert any("user_id" in error["loc"] for error in error_detail)
    assert any("query" in error["loc"] for error in error_detail)