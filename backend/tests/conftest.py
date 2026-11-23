"""Test setup: an isolated users file and secret, so tests never touch backend/users.json."""
import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="fdt-tests-"))
os.environ["USERS_FILE"] = str(_tmp / "users.json")
os.environ["JWT_SECRET"] = "test-secret-not-for-production-use-0123456789"
os.environ["MAX_FAILED_LOGINS"] = "3"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services import user_store  # noqa: E402
from app.utils.security import hash_password  # noqa: E402

TEST_USER = {"username": "tester", "password": "Test@12345", "role": "dispatcher", "name": "Test User"}
user_store.save_user({"username": TEST_USER["username"], "name": TEST_USER["name"], "role": TEST_USER["role"], "password_hash": hash_password(TEST_USER["password"])})


@pytest.fixture(scope="session")
def anon_client():
    """Client without a login token."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def client(anon_client):
    """Client that is logged in as the test user."""
    with TestClient(app) as c:
        token = c.post("/api/auth/login", json={"username": TEST_USER["username"], "password": TEST_USER["password"]}).json()["access_token"]
        c.headers.update({"Authorization": f"Bearer {token}"})
        yield c


@pytest.fixture(autouse=True)
def no_real_weather_calls(monkeypatch):
    """Tests never call Open-Meteo: both live sources behave as 'offline' unless a test installs a fake."""
    from app.services import weather_source

    def offline(*args, **kwargs):
        raise OSError("network disabled in tests")
    monkeypatch.setattr(weather_source, "fetch_open_meteo", offline)
    monkeypatch.setattr(weather_source, "fetch_previous_runs", offline)
    monkeypatch.setattr(weather_source, "_live_down_until", 0.0)
    monkeypatch.setattr(weather_source, "_fc_down_until", 0.0)
    weather_source._live_cache.clear()
    yield
    weather_source._live_cache.clear()


@pytest.fixture(autouse=True)
def no_real_traffic_calls(monkeypatch, tmp_path):
    """Tests never call TomTom (even if backend/.env has a key) and never write to the real traffic log."""
    from app import config
    from app.services import traffic_live

    monkeypatch.setattr(config, "TOMTOM_API_KEY", "")
    monkeypatch.setattr(config, "TRAFFIC_LOG_FILE", tmp_path / "traffic_observations.csv")

    def offline(*args, **kwargs):
        raise traffic_live.TrafficAPIError("network disabled in tests")
    monkeypatch.setattr(traffic_live, "fetch_flow", offline)
    traffic_live.clear()
    monkeypatch.setitem(traffic_live._usage, "requests", 0)
    yield
    traffic_live.clear()


@pytest.fixture(autouse=True)
def no_backup_ai_from_env(monkeypatch):
    """A backup AI provider in backend/.env must not change the tests; tests that need one set it themselves."""
    from app import config
    monkeypatch.setattr(config, "LLM_FALLBACK_PROVIDER", "none")
    monkeypatch.setattr(config, "LLM_FALLBACK_API_KEY", "")
