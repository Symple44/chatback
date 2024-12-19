# tests/conftest.py
import pytest
from httpx import AsyncClient
from main import app
from core.database.base import get_session_manager
from core.config import settings

@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
async def test_db():
    test_session_manager = get_session_manager(settings.DATABASE_URL + "_test")
    await test_session_manager.create_all()
    yield test_session_manager
    await test_session_manager.drop_all()
