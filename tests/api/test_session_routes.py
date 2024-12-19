# tests/api/test_session_routes.py
@pytest.mark.asyncio
class TestSessionAPI:
    async def test_create_session(self, async_client, test_db):
        """Test la création d'une session."""
        # Créer un utilisateur d'abord
        user_data = {
            "email": "session@test.com",
            "username": "sessionuser",
            "full_name": "Session User",
            "user_metadata": {}
        }
        user_response = await async_client.post("/api/users/new", json=user_data)
        user_id = user_response.json()["id"]

        session_data = {
            "user_id": user_id,
            "session_metadata": {"source": "web"}
        }
        response = await async_client.post("/api/sessions/new", json=session_data)
        assert response.status_code == 200, f"Échec création session: {response.json()}"
        data = response.json()
        assert all(k in data for k in ["session_id", "user_id", "session_metadata"])
        return data["session_id"]

    async def test_get_user_sessions(self, async_client, test_db):
        """Test la récupération des sessions d'un utilisateur."""
        user_id = await TestUserAPI().test_create_user(async_client, test_db)
        await self.test_create_session(async_client, test_db)
        
        response = await async_client.get(f"/api/sessions/user/{user_id}")
        assert response.status_code == 200, "Échec récupération sessions"
        assert isinstance(response.json(), list), "Devrait retourner une liste"

    async def test_delete_session(self, async_client, test_db):
        """Test la suppression d'une session."""
        session_id = await self.test_create_session(async_client, test_db)
        response = await async_client.delete(f"/api/sessions/del/{session_id}")
        assert response.status_code == 200, "Échec suppression session"
