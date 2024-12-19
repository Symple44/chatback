# tests/api/test_user_routes.py
@pytest.mark.asyncio
class TestUserAPI:
    async def test_create_user(self, async_client, test_db):
        """Test la création d'un utilisateur."""
        test_data = {
            "email": "test@example.com",
            "username": "testuser",
            "full_name": "Test User",
            "user_metadata": {"preferences": {"theme": "dark"}}
        }
        response = await async_client.post("/api/users/new", json=test_data)
        assert response.status_code == 200, f"Échec création utilisateur: {response.json()}"
        data = response.json()
        for field in ["id", "email", "username", "full_name", "user_metadata"]:
            assert field in data, f"Champ manquant: {field}"
        return data["id"]

    async def test_get_user(self, async_client, test_db):
        """Test la récupération d'un utilisateur."""
        user_id = await self.test_create_user(async_client, test_db)
        response = await async_client.get(f"/api/users/{user_id}")
        assert response.status_code == 200, f"Échec récupération utilisateur: {response.json()}"
        data = response.json()
        assert data["id"] == user_id, "ID utilisateur incorrect"

    async def test_get_nonexistent_user(self, async_client, test_db):
        """Test la récupération d'un utilisateur inexistant."""
        fake_id = str(uuid.uuid4())
        response = await async_client.get(f"/api/users/{fake_id}")
        assert response.status_code == 404, "Devrait retourner 404 pour un utilisateur inexistant"
