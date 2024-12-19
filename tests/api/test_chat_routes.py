# tests/api/test_chat_routes.py
@pytest.mark.asyncio
class TestChatAPI:
    async def test_send_message(self, async_client, test_db):
        """Test l'envoi d'un message."""
        session_id = await TestSessionAPI().test_create_session(async_client, test_db)
        
        chat_data = {
            "session_id": session_id,
            "query": "Comment créer une commande ?",
            "chat_metadata": {
                "language": "fr",
                "source": "test"
            }
        }
        response = await async_client.post("/api/chat/", json=chat_data)
        assert response.status_code == 200, f"Échec envoi message: {response.json()}"
        data = response.json()
        assert all(k in data for k in ["response", "session_id", "confidence_score"])
        return data["id"]

    async def test_get_chat_history(self, async_client, test_db):
        """Test la récupération de l'historique des chats."""
        session_id = await TestSessionAPI().test_create_session(async_client, test_db)
        await self.test_send_message(async_client, test_db)
        
        response = await async_client.get(f"/api/chat/history/{session_id}")
        assert response.status_code == 200, "Échec récupération historique"
        data = response.json()
        assert isinstance(data, list), "Devrait retourner une liste"
        assert len(data) > 0, "L'historique ne devrait pas être vide"

    async def test_stream_chat(self, async_client, test_db):
        """Test le streaming de chat."""
        session_id = await TestSessionAPI().test_create_session(async_client, test_db)
        
        chat_data = {
            "session_id": session_id,
            "query": "Explique le processus de création de commande",
            "chat_metadata": {"stream": True}
        }
        async with async_client.stream("POST", "/api/chat/stream", json=chat_data) as response:
            assert response.status_code == 200, "Échec initialisation stream"
            received_data = False
            async for line in response.aiter_lines():
                if line.strip():
                    received_data = True
                    assert "data:" in line, "Format SSE incorrect"
            assert received_data, "Aucune donnée reçue dans le stream"
