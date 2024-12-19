# tests/api/test_integration.py
@pytest.mark.asyncio
class TestIntegrationFlows:
    async def test_complete_user_flow(self, async_client, test_db):
        """Test le flux complet d'un utilisateur."""
        # 1. Création utilisateur
        user_id = await TestUserAPI().test_create_user(async_client, test_db)
        
        # 2. Création session
        session_data = {
            "user_id": user_id,
            "session_metadata": {"source": "integration_test"}
        }
        session_response = await async_client.post("/api/sessions/new", json=session_data)
        session_id = session_response.json()["session_id"]
        
        # 3. Envoi plusieurs messages
        messages = [
            "Comment créer une commande ?",
            "Quelles sont les étapes ?",
            "Merci pour les informations"
        ]
        
        for msg in messages:
            chat_data = {
                "session_id": session_id,
                "query": msg,
                "chat_metadata": {"test_flow": True}
            }
            response = await async_client.post("/api/chat/", json=chat_data)
            assert response.status_code == 200, f"Message échoué: {msg}"
        
        # 4. Vérification historique
        history_response = await async_client.get(f"/api/chat/history/{session_id}")
        history = history_response.json()
        assert len(history) == len(messages), "Nombre de messages incorrect"
        
        # 5. Suppression session
        delete_response = await async_client.delete(f"/api/sessions/del/{session_id}")
        assert delete_response.status_code == 200, "Échec suppression session"
