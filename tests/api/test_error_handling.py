# tests/api/test_error_handling.py
@pytest.mark.asyncio
class TestErrorHandling:
    async def test_invalid_user_data(self, async_client, test_db):
        """Test la validation des données utilisateur."""
        invalid_cases = [
            ({}, "Données manquantes"),
            ({"email": "invalid"}, "Email invalide"),
            ({"email": "test@example.com", "username": "a"}, "Username trop court"),
        ]
        
        for data, case in invalid_cases:
            response = await async_client.post("/api/users/new", json=data)
            assert response.status_code == 422, f"Cas '{case}' devrait échouer"

    async def test_invalid_session_data(self, async_client, test_db):
        """Test la validation des données de session."""
        invalid_uuid = "not-a-uuid"
        response = await async_client.post("/api/sessions/new", json={"user_id": invalid_uuid})
        assert response.status_code == 422, "UUID invalide devrait être rejeté"

    async def test_invalid_chat_data(self, async_client, test_db):
        """Test la validation des données de chat."""
        response = await async_client.post("/api/chat/", json={
            "session_id": "invalid",
            "query": ""
        })
        assert response.status_code == 422, "Message vide devrait être rejeté"
