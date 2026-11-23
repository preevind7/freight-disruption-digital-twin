"""Login / protection tests."""
import pytest

from tests.conftest import TEST_USER

PROTECTED = [
    ("post", "/api/predict/rainfall"),
    ("post", "/api/predict/accident"),
    ("post", "/api/predict/traffic"),
    ("get", "/api/models"),
    ("get", "/api/models/rainfall"),
    ("get", "/api/rainfall/history?city=Mumbai&forecast_date=2025-07-20"),
    ("post", "/api/models/reload"),
    ("get", "/api/auth/me"),
]


def test_health_is_public(anon_client):
    assert anon_client.get("/api/health").status_code == 200


@pytest.mark.parametrize("method,path", PROTECTED)
def test_protected_without_token(anon_client, method, path):
    r = getattr(anon_client, method)(path, json={}) if method == "post" else anon_client.get(path)
    assert r.status_code == 401, path
    assert r.headers.get("www-authenticate") == "Bearer"
    assert "Login required" in r.json()["detail"]


def test_login_success_and_me(anon_client):
    r = anon_client.post("/api/auth/login", json={"username": "TESTER", "password": TEST_USER["password"]})  # username case-insensitive
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer" and body["access_token"]
    assert body["user"] == {"username": "tester", "name": "Test User", "role": "dispatcher", "role_title": "Freight Dispatcher"}
    me = anon_client.get("/api/auth/me", headers={"Authorization": "Bearer " + body["access_token"]})
    assert me.json()["username"] == "tester"
    assert "password_hash" not in me.text


def test_swagger_form_login(anon_client):
    r = anon_client.post("/api/auth/token", data={"username": "tester", "password": TEST_USER["password"]})
    assert r.status_code == 200 and r.json()["access_token"]


def test_wrong_password_and_unknown_user(anon_client):
    for payload in ({"username": "tester", "password": "nope"}, {"username": "ghost", "password": "whatever"}):
        r = anon_client.post("/api/auth/login", json=payload)
        assert r.status_code == 401
        assert r.json()["detail"] == "Incorrect username or password."  # same message: no username probing


def test_invalid_and_tampered_tokens(anon_client, client):
    assert anon_client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-token"}).status_code == 401
    good = client.headers["Authorization"]
    tampered = good[:-3] + ("aaa" if not good.endswith("aaa") else "bbb")
    assert anon_client.get("/api/auth/me", headers={"Authorization": tampered}).status_code == 401


def test_expired_token(anon_client, monkeypatch):
    from app import config
    from app.utils import security
    monkeypatch.setattr(config, "TOKEN_EXPIRE_MINUTES", -1)
    token = security.create_access_token("tester", "dispatcher")["access_token"]
    r = anon_client.get("/api/auth/me", headers={"Authorization": "Bearer " + token})
    assert r.status_code == 401 and "expired" in r.json()["detail"]


def test_lockout_after_repeated_failures(anon_client):
    from app.services import user_store
    from app.utils.security import hash_password
    user_store.save_user({"username": "lockme", "name": "Lock Me", "role": "planner", "password_hash": hash_password("Right@1234")})
    for _ in range(3):  # MAX_FAILED_LOGINS=3 in conftest
        assert anon_client.post("/api/auth/login", json={"username": "lockme", "password": "wrong"}).status_code == 401
    r = anon_client.post("/api/auth/login", json={"username": "lockme", "password": "Right@1234"})
    assert r.status_code == 429  # locked even with the right password


def test_logged_in_prediction_works(client):
    r = client.post("/api/predict/accident", json={
        "city": "Pune", "road_type": "urban", "lanes": 2, "traffic_signal": True, "weather": "clear", "visibility": "high",
        "temperature": 30, "traffic_density": "low", "hour": 3, "day_of_week": "Sunday", "festival": "None"})
    assert r.status_code == 200


def test_password_hash_roundtrip():
    from app.utils.security import hash_password, verify_password
    h = hash_password("S3cret!pass")
    assert verify_password("S3cret!pass", h) and not verify_password("wrong", h)
    assert "S3cret" not in h


# ---------------- sign-up ----------------
NEW_USER = {"name": "Ravi  Kumar", "username": "Ravi.K", "password": "Ravi@2026x", "role": "operator"}


def test_register_then_login(anon_client):
    r = anon_client.post("/api/auth/register", json=NEW_USER)
    assert r.status_code == 201, r.text
    assert r.json() == {"username": "ravi.k", "name": "Ravi Kumar", "role": "operator", "role_title": "Fleet Operator"}
    assert "password" not in r.text
    login = anon_client.post("/api/auth/login", json={"username": "ravi.k", "password": NEW_USER["password"]})
    assert login.status_code == 200
    assert login.json()["user"]["name"] == "Ravi Kumar"


def test_register_duplicate_username(anon_client):
    anon_client.post("/api/auth/register", json={**NEW_USER, "username": "dupe_user"})
    r = anon_client.post("/api/auth/register", json={**NEW_USER, "username": "DUPE_USER"})
    assert r.status_code == 422
    assert "already taken" in r.json()["detail"]


@pytest.mark.parametrize(
    "change,field",
    [
        ({"password": "short1"}, "password"),           # too short
        ({"password": "onlyletters"}, "password"),      # no number
        ({"password": "12345678"}, "password"),         # no letter
        ({"username": "has space"}, "username"),
        ({"username": "ab"}, "username"),
        ({"name": " "}, "name"),
        ({"role": "admin"}, "role"),
    ],
)
def test_register_invalid(anon_client, change, field):
    r = anon_client.post("/api/auth/register", json={**NEW_USER, "username": "valid_name1", **change})
    assert r.status_code == 422
    assert any(e["field"] == field for e in r.json()["errors"]), r.json()


def test_register_password_containing_username(anon_client):
    r = anon_client.post("/api/auth/register", json={**NEW_USER, "username": "meena", "password": "meena1234"})
    assert r.status_code == 422 and r.json()["errors"][0]["field"] == "password"
