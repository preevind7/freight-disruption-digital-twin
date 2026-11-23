"""TomTom live traffic: measurement, use in the corridor forecast, caching, budget, errors, no key leaks.
TomTom is replaced by a fake; no network is used."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from app import config
from app.services import corridor_service, traffic_live

URL = "/api/corridor/forecast"
KEY = "test-tomtom-key-SECRET123"
REAL_FETCH = traffic_live.fetch_flow      # saved before the test fixtures replace it


def test_measure_speed_ratio_levels_and_closure():
    free = traffic_live.measure({"currentSpeed": 70, "freeFlowSpeed": 70, "confidence": 1})
    assert (free["speed_ratio"], free["congestion"], free["level"]) == (1.0, 0.0, "low")
    slow = traffic_live.measure({"currentSpeed": 42, "freeFlowSpeed": 70, "confidence": 0.9})
    assert slow["level"] == "medium" and slow["congestion"] == pytest.approx(0.8)
    jam = traffic_live.measure({"currentSpeed": 20, "freeFlowSpeed": 70, "confidence": 0.9})
    assert jam["level"] == "high" and jam["congestion"] == 1.0
    closed = traffic_live.measure({"currentSpeed": 60, "freeFlowSpeed": 70, "roadClosure": True})
    assert closed["level"] == "high" and closed["road_closure"] and closed["congestion"] == 1.0


def test_without_key_the_model_is_used(client):
    st = client.get("/api/traffic/live/status").json()
    assert st["enabled"] is False and "TOMTOM_API_KEY" in st["reason"]
    body = client.get(URL, params={"date": config.today_ist().isoformat()}).json()
    assert body["traffic_sources"] == {"live": 0, "model": 25}
    assert all(s["live_traffic"] is None for s in body["segments"])


@pytest.fixture
def tomtom(monkeypatch):
    calls = []

    def fake(lat, lon):
        calls.append((lat, lon))
        if abs(lat - 18.7915) < 1e-3:                       # S03 Bhor Ghat: closed
            return {"currentSpeed": 5, "freeFlowSpeed": 60, "roadClosure": True, "confidence": 0.95, "frc": "FRC1"}
        if abs(lat - 19.0330) < 0.2 and lon < 73.2:        # around Mumbai: jam
            return {"currentSpeed": 18, "freeFlowSpeed": 65, "confidence": 0.9, "frc": "FRC0"}
        return {"currentSpeed": 62, "freeFlowSpeed": 70, "confidence": 0.8, "frc": "FRC0",
                "currentTravelTime": 300, "freeFlowTravelTime": 266}
    monkeypatch.setattr(config, "TOMTOM_API_KEY", KEY)
    monkeypatch.setattr(traffic_live, "fetch_flow", fake)
    corridor_service.clear_cache()
    yield calls
    corridor_service.clear_cache()


def test_live_traffic_replaces_current_hour_and_is_logged(client, tomtom):
    body = client.get(URL, params={"date": config.today_ist().isoformat()}).json()
    hour = datetime.now(config.IST).hour
    assert body["traffic_sources"] == {"live": 25, "model": 0} and len(tomtom) == 25
    for seg in body["segments"]:
        obs = seg["live_traffic"]
        assert obs["source"] == "TomTom"
        assert seg["hourly"]["congestion_level"][hour] == obs["level"]
        assert seg["hourly"]["p_high_congestion"][hour] == pytest.approx(obs["congestion"], abs=1e-3)
    s03 = next(s for s in body["segments"] if s["id"] == "S03")
    assert s03["live_traffic"]["road_closure"] is True
    assert any(a["type"] == "traffic" and a["segment_id"] == "S03" and a["severity"] == "High" for a in body["alerts"])
    rows = config.TRAFFIC_LOG_FILE.read_text(encoding="utf-8").splitlines()
    assert rows[0].split(",")[:3] == ["observed_at_ist", "date", "hour"] and len(rows) == 26
    assert KEY not in json.dumps(body) and KEY not in config.TRAFFIC_LOG_FILE.read_text(encoding="utf-8")
    st = client.get("/api/traffic/live/status").json()
    assert st["enabled"] and st["requests_today"] == 25 and st["observations_logged"] == 25 and KEY not in json.dumps(st)


def test_cache_avoids_repeat_calls_and_other_dates_never_call(client, tomtom):
    client.get(URL, params={"date": config.today_ist().isoformat()})
    corridor_service.clear_cache()
    client.get(URL, params={"date": config.today_ist().isoformat(), "refresh": "true"})
    assert len(tomtom) == 25                                    # cached readings reused
    past = client.get(URL, params={"date": (config.today_ist() - timedelta(days=3)).isoformat()}).json()
    assert len(tomtom) == 25 and past["traffic_sources"]["live"] == 0


def test_daily_budget_is_respected(client, tomtom, monkeypatch):
    monkeypatch.setattr(config, "TRAFFIC_DAILY_BUDGET", 10)
    body = client.get(URL, params={"date": config.today_ist().isoformat()}).json()
    assert len(tomtom) == 10 and body["traffic_sources"] == {"live": 10, "model": 15}


def test_rejected_key_pauses_and_message_has_no_key(client, monkeypatch):
    import io
    import urllib.error

    def forbidden(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO(b"{}"))
    monkeypatch.setattr(config, "TOMTOM_API_KEY", KEY)
    monkeypatch.setattr(traffic_live, "fetch_flow", REAL_FETCH)          # the real request code ...
    monkeypatch.setattr(traffic_live.urllib.request, "urlopen", forbidden)  # ... with TomTom answering 403
    corridor_service.clear_cache()
    body = client.get(URL, params={"date": config.today_ist().isoformat()}).json()
    assert body["traffic_sources"]["live"] == 0
    assert "rejected the key" in body["live_traffic_status"] and KEY not in body["live_traffic_status"]
    st = client.get("/api/traffic/live/status").json()
    assert st["paused"] and KEY not in json.dumps(st)
    corridor_service.clear_cache()
