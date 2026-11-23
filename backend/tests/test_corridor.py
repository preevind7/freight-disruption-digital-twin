"""Corridor forecast tests (all three models together)."""
from datetime import timedelta

import pytest

from app import config
from app.services import corridor_service, rainfall_service, weather_source

URL = "/api/corridor/forecast"


@pytest.fixture(autouse=True)
def fresh_state():
    corridor_service.clear_cache()
    weather_source._live_cache.clear()
    weather_source._live_down_until = 0.0
    yield
    corridor_service.clear_cache()


def test_requires_login(anon_client):
    assert anon_client.get(URL).status_code == 401


def test_dataset_date_heavy_rain_day(client):
    body = client.get(URL, params={"corridor_id": "mumbai-bengaluru", "date": "2025-06-19"}).json()
    assert body["rainfall_sources"]["dataset"] == 25
    assert len(body["segments"]) == 25
    s01 = next(s for s in body["segments"] if s["id"] == "S01")
    if rainfall_service.is_corridor_dataset():
        # real per-segment rainfall: every segment uses its own location
        assert all(s["cities"]["rainfall"]["km"] == 0.0 and s["cities"]["rainfall"]["name"].startswith(s["id"]) for s in body["segments"])
    else:
        assert s01["cities"]["rainfall"]["name"] == "Mumbai"
        assert s01["rainfall"]["heavy_alert"] is True          # 107.8 mm was observed that day in the city file
    for seg in body["segments"]:
        assert seg["rainfall"]["heavy_alert"] == any(a["type"] == "heavy_rain" and a["segment_id"] == seg["id"] for a in body["alerts"])
    for seg in body["segments"]:
        for key in ("risk", "accident_risk", "p_high_congestion", "congestion_level"):
            assert len(seg["hourly"][key]) == 24
        assert all(0 <= r <= 1 for r in seg["hourly"]["risk"])
        assert abs(sum(seg["rainfall"]["probabilities"].values()) - 1) < 0.01


def test_risk_follows_stated_formula(client):
    body = client.get(URL, params={"date": "2025-06-19"}).json()
    w = body["method"]["weights"]
    seg = body["segments"][0]
    for hour in (3, 9, 18):
        expected = 1 - (1 - w["rain"] * seg["rainfall"]["rain_component"]) * (1 - w["accident"] * seg["hourly"]["accident_risk"][hour]) * (1 - w["congestion"] * seg["hourly"]["p_high_congestion"][hour])
        assert seg["hourly"]["risk"][hour] == pytest.approx(expected, abs=0.002)


def test_deterministic(client):
    a = client.get(URL, params={"date": "2024-08-01"}).json()
    corridor_service.clear_cache()
    b = client.get(URL, params={"date": "2024-08-01"}).json()
    assert a["segments"] == b["segments"]


def test_festival_detected(client):
    body = client.get(URL, params={"corridor_id": "chennai-bengaluru", "date": "2026-11-08"}).json()
    assert body["event"]["festival"] == "Diwali"
    assert any(a["type"] == "festival" for a in body["alerts"])


def _fake_open_meteo(mm):
    def fake(lat, lon, start, end):
        days = (end - start).days + 1
        return {(start + timedelta(days=i)).isoformat(): mm for i in range(days)}
    return fake


def test_live_weather_used_for_today(client, monkeypatch):
    monkeypatch.setattr(config, "LIVE_WEATHER", True)
    monkeypatch.setattr(weather_source, "fetch_open_meteo", _fake_open_meteo(0.0))
    dry = client.get(URL, params={"date": config.today_ist().isoformat()}).json()
    assert dry["rainfall_sources"]["live"] == 25
    corridor_service.clear_cache(); weather_source._live_cache.clear()
    monkeypatch.setattr(weather_source, "fetch_open_meteo", _fake_open_meteo(90.0))
    wet = client.get(URL, params={"date": config.today_ist().isoformat()}).json()
    assert wet["segments"][0]["rainfall"]["last_7_days_mm"] == [90.0] * 7
    assert wet["segments"][0]["rainfall"]["probabilities"] != dry["segments"][0]["rainfall"]["probabilities"]  # input matters
    assert wet["segments"][0]["rainfall"]["p_rain"] > dry["segments"][0]["rainfall"]["p_rain"]


def test_replay_when_offline(client, monkeypatch):
    def broken(*args):
        raise OSError("no internet")
    monkeypatch.setattr(config, "LIVE_WEATHER", True)
    monkeypatch.setattr(weather_source, "fetch_open_meteo", broken)
    body = client.get(URL, params={"date": config.today_ist().isoformat()}).json()
    assert body["rainfall_sources"]["replay"] == 25
    assert "replaying" in body["segments"][0]["rainfall"]["source_detail"]


@pytest.mark.parametrize("params", [{"corridor_id": "delhi-agra"}, {"date": "2010-01-01"}, {"date": "2099-01-01"}])
def test_invalid_requests(client, params):
    assert client.get(URL, params=params).status_code == 422
