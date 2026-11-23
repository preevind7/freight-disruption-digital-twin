"""Prediction API tests (logged in). Run from the backend folder:  python -m pytest -v

They need the trained model files (python -m training.train_all).
"""
from __future__ import annotations

import shutil

import pytest

from app import config

RAIN_INPUT = {"forecast_date": "2025-07-15", "rainfall_last_7_days_mm": [42.0, 18.5, 60.2, 12.0, 5.0, 0.0, 22.4]}
ACC_OK = {
    "city": "Mumbai", "road_type": "highway", "lanes": 4, "traffic_signal": False, "weather": "rain",
    "visibility": "low", "temperature": 27, "traffic_density": "high", "hour": 18, "day_of_week": "Friday", "festival": "None",
}
TRAFFIC_OK = {"city": "Bangalore", "date": "2026-11-08", "hour": 18, "road_type": "urban", "lanes": 3, "traffic_signal": True, "weather": "clear"}


# The logged-in `client` fixture comes from tests/conftest.py.


# ---------------- health & metadata ----------------
def test_health_all_models_loaded(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert all(m["loaded"] for m in body["models"].values())


def test_models_overview_has_metrics(client):
    body = client.get("/api/models").json()
    for name in ("rainfall", "accident", "traffic"):
        assert body[name]["metrics"]["test"], name
        assert body[name]["input_options"], name


def test_cors_preflight(client):
    origin = config.CORS_ORIGINS[0]
    r = client.options("/api/predict/rainfall", headers={"Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type,authorization"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == origin


# ---------------- rainfall ----------------
# The rainfall model is trained either on the uploaded 20-city file ("Mumbai") or on the real
# per-segment Open-Meteo file ("S01 Mumbai – Panvel"); the tests use whichever is loaded.
@pytest.fixture(scope="module")
def rain_city(client):
    locations = client.get("/api/models/rainfall").json()["input_options"]["location"]
    return next(loc for loc in locations if "Mumbai" in loc)


@pytest.fixture
def rain_ok(rain_city):
    return {"city": rain_city, **RAIN_INPUT}


def test_rainfall_valid(client, rain_ok):
    r = client.post("/api/predict/rainfall", json=rain_ok)
    assert r.status_code == 200, r.text
    body = r.json()
    assert abs(sum(c["probability"] for c in body["category_probabilities"]) - 1) < 0.01
    assert body["flood_trigger_risk_level"] in {"Low", "Moderate", "High"}
    assert body["model"]["name"] == "rainfall"


def test_rainfall_is_deterministic_and_input_sensitive(client, rain_ok):
    a = client.post("/api/predict/rainfall", json=rain_ok).json()
    b = client.post("/api/predict/rainfall", json=rain_ok).json()
    dry = client.post("/api/predict/rainfall", json={**rain_ok, "rainfall_last_7_days_mm": [0] * 7}).json()
    assert a == b  # same input -> same output (no randomness)
    assert dry["category_probabilities"] != a["category_probabilities"]  # output depends on input


def test_rainfall_city_case_insensitive(client, rain_ok, rain_city):
    assert client.post("/api/predict/rainfall", json={**rain_ok, "city": rain_city.lower()}).json()["city"] == rain_city


@pytest.mark.parametrize(
    "change",
    [
        {"city": "Atlantis"},
        {"rainfall_last_7_days_mm": [1, 2, 3]},
        {"rainfall_last_7_days_mm": [-1, 0, 0, 0, 0, 0, 0]},
        {"forecast_date": "15-07-2025"},
        {"city": None},
    ],
)
def test_rainfall_invalid(client, rain_ok, change):
    payload = {k: v for k, v in {**rain_ok, **change}.items() if v is not None}
    r = client.post("/api/predict/rainfall", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]


def test_rainfall_history(client, rain_city):
    body = client.get("/api/rainfall/history", params={"city": rain_city, "forecast_date": "2025-07-20"}).json()
    assert len(body["rainfall_last_7_days_mm"]) == 7
    assert body["actual_rainfall_mm"] is not None


def test_rainfall_history_out_of_range(client, rain_city):
    r = client.get("/api/rainfall/history", params={"city": rain_city, "forecast_date": "2030-01-01"})
    assert r.status_code == 404


# ---------------- accident ----------------
def test_accident_valid_matches_dataset_rule(client):
    body = client.post("/api/predict/accident", json=ACC_OK).json()
    # 0.10 base + 0.35 (rain, low visibility) + 0.20 (high density) + 0.15 (peak hour)
    assert body["risk_score"] == pytest.approx(0.80, abs=0.01)
    assert body["risk_level"] == "High"
    assert body["drivers"]


def test_accident_low_risk(client):
    calm = {**ACC_OK, "weather": "clear", "visibility": "high", "traffic_density": "low", "hour": 3}
    body = client.post("/api/predict/accident", json=calm).json()
    assert body["risk_level"] == "Low"


@pytest.mark.parametrize(
    "payload",
    [
        {**ACC_OK, "weather": "clear", "visibility": "low"},  # never seen together
        {**ACC_OK, "hour": 24},
        {**ACC_OK, "lanes": 0},
        {**ACC_OK, "road_type": "motorway"},
        {**ACC_OK, "day_of_week": "Funday"},
    ],
)
def test_accident_invalid(client, payload):
    assert client.post("/api/predict/accident", json=payload).status_code == 422


# ---------------- traffic ----------------
def test_traffic_detects_festival(client):
    body = client.post("/api/predict/traffic", json=TRAFFIC_OK).json()
    assert body["festival_event"] == "Diwali"
    assert body["congestion_level"] in {"low", "medium", "high"}
    assert abs(sum(body["probabilities"].values()) - 1) < 0.01


def test_traffic_normal_day(client):
    body = client.post("/api/predict/traffic", json={**TRAFFIC_OK, "date": "2026-11-04"}).json()
    assert body["festival_event"] == "None"
    assert body["calendar_event"] is None


def test_traffic_invalid(client):
    assert client.post("/api/predict/traffic", json={**TRAFFIC_OK, "festival": "Carnival"}).status_code == 422
    assert client.post("/api/predict/traffic", json={**TRAFFIC_OK, "city": "Paris"}).status_code == 422


# ---------------- missing model file ----------------
def test_missing_model_returns_503(client, tmp_path):
    backup = tmp_path / "traffic_model.joblib"
    shutil.move(config.TRAFFIC_MODEL_PATH, backup)
    try:
        client.post("/api/models/reload")
        r = client.post("/api/predict/traffic", json=TRAFFIC_OK)
        assert r.status_code == 503
        assert "train_traffic" in r.json()["detail"]
        # the other models keep working
        assert client.post("/api/predict/accident", json=ACC_OK).status_code == 200
        assert client.get("/api/health").json()["status"] == "degraded"
    finally:
        shutil.move(backup, config.TRAFFIC_MODEL_PATH)
        client.post("/api/models/reload")
    assert client.post("/api/predict/traffic", json=TRAFFIC_OK).status_code == 200
