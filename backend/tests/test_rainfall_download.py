"""Tests for the real-rainfall download (training/download_rainfall.py) and the per-segment model path.

No internet is used: Open-Meteo is replaced by a fake that returns seeded, made-up rainfall.
The fake data only checks the plumbing (download -> CSV -> training -> serving); it is never
shipped and never used to report accuracy.
"""
from __future__ import annotations

import io
import json
import urllib.error
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import pytest

from app import config
from app.ml_models import rainfall_model as rm
from app.ml_models.registry import registry
from app.services import corridor_service, rainfall_service, weather_source
from training import download_rainfall as dl


class FakeClock:
    def __init__(self):
        self.now = 1_000_000.0
        self.slept = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def fake_archive(calls=None, missing=None, trailing_none=0):
    """Seeded monsoon-shaped rain per location; optional gaps to test validation."""
    def fetch(lat, lon, start, end):
        if calls is not None:
            calls.append((lat, lon, start, end))
        rng = np.random.default_rng(int(abs(lat * 1000 + lon * 10)))
        out = {}
        day = start
        while day <= end:
            monsoon = 1.0 if 6 <= day.month <= 9 else 0.15
            wet = rng.random() < 0.55 * monsoon
            out[day.isoformat()] = round(float(rng.gamma(0.9, 28.0)) if wet else 0.0, 1)
            day += timedelta(days=1)
        for d in missing or []:
            out[d] = None
        for i in range(trailing_none):
            out[(end - timedelta(days=i)).isoformat()] = None
        return out
    return fetch


def pacer(clock, hourly=4000.0, daily=9000.0):
    return dl.Pacer(hourly, daily, clock=clock.time, sleep=clock.sleep, say=lambda *_: None)


# ---------------- download script ----------------
def test_call_cost_matches_open_meteo_rule():
    assert dl.call_cost(date(2024, 1, 1), date(2024, 1, 14)) == 1.0
    assert dl.call_cost(date(2024, 1, 1), date(2024, 1, 3)) == 1.0
    assert dl.call_cost(date(2020, 1, 1), date(2020, 12, 30)) == pytest.approx(365 / 14)


def test_pacer_waits_for_hourly_budget():
    clock = FakeClock()
    p = pacer(clock, hourly=100)
    p.wait_for(60); p.record(60)
    p.wait_for(30); p.record(30)
    assert clock.slept == []
    p.wait_for(30)                      # 60 + 30 + 30 > 100 -> must wait until the first call is an hour old
    assert clock.slept and clock.slept[0] > 3500


def test_pacer_stops_at_daily_budget():
    clock = FakeClock()
    p = pacer(clock, hourly=1000, daily=150)
    p.wait_for(100); p.record(100)
    with pytest.raises(dl.DailyLimitReached):
        p.wait_for(100)


def test_clean_series_trims_unfilled_days_and_rejects_gaps():
    start, end = date(2024, 1, 1), date(2024, 1, 31)
    s = dl.clean_series(fake_archive(trailing_none=3)(19, 73, start, end), start, end)
    assert s.index.max().date() == date(2024, 1, 28) and not s.isna().any()
    with pytest.raises(dl.RainfallDownloadError):
        dl.clean_series(fake_archive(missing=["2024-01-10"])(19, 73, start, end), start, end)


def _http_429(reason):
    return urllib.error.HTTPError(dl.ARCHIVE_URL, 429, "Too Many Requests", {}, io.BytesIO(json.dumps({"error": True, "reason": reason}).encode()))


def test_rate_limit_minutely_retries_and_daily_stops():
    clock = FakeClock()
    attempts = []

    def flaky(lat, lon, start, end):
        attempts.append(1)
        if len(attempts) == 1:
            raise _http_429("Minutely API request limit exceeded. Please try again in one minute.")
        return fake_archive()(lat, lon, start, end)

    values = dl.fetch_with_retry(flaky, pacer(clock), 19, 73, date(2024, 1, 1), date(2024, 1, 5))
    assert len(values) == 5 and clock.slept == [65]

    def daily(*_):
        raise _http_429("Daily API request limit exceeded. Please try again tomorrow.")
    with pytest.raises(dl.DailyLimitReached):
        dl.fetch_with_retry(daily, pacer(FakeClock()), 19, 73, date(2024, 1, 1), date(2024, 1, 5))


def test_run_builds_dataset_and_resumes_with_only_new_days(tmp_path):
    out, cache = tmp_path / "corridor.csv", tmp_path / "cache"
    calls = []
    start, end = date(2023, 1, 1), date(2023, 3, 31)
    df = dl.run(start, end, out, cache, pacer(FakeClock()), fetch=fake_archive(calls), say=lambda *_: None)
    segments, _ = dl.load_segments()
    assert len(calls) == len(segments) == df["Location"].nunique() == 35
    assert set(["Date", "Location", "Latitude", "Longitude", "Rainfall_mm", "Segment_ID", "Source"]) <= set(df.columns)
    s03 = df[df["Segment_ID"] == "S03"].iloc[0]
    assert s03["Location"] == "S03 Khalapur – Lonavala (Bhor Ghat)"
    assert (s03["Latitude"], s03["Longitude"]) == (18.7915, 73.3405)
    rm.build_training_frame(df)                     # consecutive days, usable for lag features

    calls.clear()                                   # second run: nothing new to fetch
    dl.run(start, end, out, cache, pacer(FakeClock()), fetch=fake_archive(calls), say=lambda *_: None)
    assert calls == []
    calls.clear()                                   # later run: only the missing tail is fetched
    dl.run(start, date(2023, 4, 10), out, cache, pacer(FakeClock()), fetch=fake_archive(calls), say=lambda *_: None)
    assert len(calls) == 35 and all(c[2] == date(2023, 4, 1) and c[3] == date(2023, 4, 10) for c in calls)
    assert pd.read_csv(out)["Date"].max() == "2023-04-10"


def test_run_stops_cleanly_at_daily_limit(tmp_path):
    clock = FakeClock()
    out = tmp_path / "corridor.csv"
    result = dl.run(date(2020, 1, 1), date(2023, 12, 31), out, tmp_path / "cache", pacer(clock, hourly=10**6, daily=500),
                    fetch=fake_archive(), say=lambda *_: None)
    assert result is None and not out.exists()
    assert 0 < len(list((tmp_path / "cache").glob("*.csv"))) < 35   # partial progress kept for the next run


# ---------------- training + serving on the per-segment file ----------------
@pytest.fixture(scope="module")
def corridor_bundle(tmp_path_factory):
    """Download (fake) -> train -> bundle, all inside a temporary folder."""
    from training import common, train_rainfall

    tmp = tmp_path_factory.mktemp("corridor-rain")
    datasets, models, reports = tmp / "datasets", tmp / "models", tmp / "reports"
    csv = datasets / config.RAINFALL_CORRIDOR_DATASET.name
    dl.run(date(2021, 1, 1), date(2024, 6, 30), csv, datasets / "cache", pacer(FakeClock(), hourly=10**6, daily=10**6),
           fetch=fake_archive(), say=lambda *_: None)
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "RAINFALL_DATASET", csv)
    mp.setattr(train_rainfall, "RAINFALL_MODEL_PATH", models / "rainfall_model.joblib")
    mp.setattr(common, "REPORTS_DIR", reports)
    try:
        train_rainfall.main()
    finally:
        mp.undo()
    report = json.loads((reports / "rainfall_report.json").read_text(encoding="utf-8"))
    return joblib.load(models / "rainfall_model.joblib"), report, datasets


def test_training_uses_segments_and_recent_test_period(corridor_bundle):
    bundle, report, _ = corridor_bundle
    assert bundle["dataset"] == "corridor_rainfall_open_meteo.csv"
    assert len(bundle["input_options"]["location"]) == 35
    assert report["split"]["validation"]["from"] == "2022-01-01"
    assert report["split"]["test"]["from"] == "2023-01-01" and report["split"]["test"]["to"] == "2024-06-30"
    assert "Open-Meteo" in report["dataset_description"] and any("ERA5" in x for x in report["limitations"])


@pytest.fixture
def use_corridor_model(corridor_bundle, monkeypatch):
    bundle, _, datasets = corridor_bundle
    monkeypatch.setattr(config, "DATASETS_DIR", datasets)
    monkeypatch.setitem(registry._bundles, "rainfall", bundle)
    corridor_service.clear_cache()
    weather_source._live_cache.clear()
    yield bundle
    corridor_service.clear_cache()


def test_corridor_forecast_uses_each_segments_own_rainfall(client, use_corridor_model):
    body = client.get("/api/corridor/forecast", params={"corridor_id": "mumbai-bengaluru", "date": "2024-06-20"}).json()
    assert body["rainfall_sources"]["dataset"] == 25
    for seg in body["segments"]:
        assert seg["cities"]["rainfall"]["name"].startswith(seg["id"] + " ")
        assert seg["cities"]["rainfall"]["km"] == 0.0
        assert "corridor_rainfall_open_meteo.csv" in seg["rainfall"]["source_detail"]
    s03 = next(s for s in body["segments"] if s["id"] == "S03")
    expected = rainfall_service._dataset().loc["S03 Khalapur – Lonavala (Bhor Ghat)"]["Rainfall_mm"]
    assert s03["rainfall"]["last_7_days_mm"] == [round(float(expected[date(2024, 6, 20) - timedelta(days=i)]), 1) for i in range(1, 8)]


def test_predict_and_history_with_segment_location(client, use_corridor_model):
    loc = "C06 Ambur – Vaniyambadi"
    hist = client.get("/api/rainfall/history", params={"city": loc, "forecast_date": "2024-06-15"}).json()
    assert hist["source"] == "corridor_rainfall_open_meteo.csv" and len(hist["rainfall_last_7_days_mm"]) == 7
    r = client.post("/api/predict/rainfall", json={"city": loc.lower(), "forecast_date": "2024-06-15", "rainfall_last_7_days_mm": hist["rainfall_last_7_days_mm"]})
    assert r.status_code == 200, r.text
    assert r.json()["city"] == loc and "Open-Meteo" in r.json()["note"]
    assert client.post("/api/predict/rainfall", json={"city": "Mumbai", "forecast_date": "2024-06-15", "rainfall_last_7_days_mm": [0] * 7}).status_code == 422


def test_old_model_keeps_old_dataset_even_if_new_file_exists(monkeypatch):
    """Downloading the file must not change anything until the model is retrained on it."""
    monkeypatch.setattr(config, "RAINFALL_DATASET", config.RAINFALL_CORRIDOR_DATASET)   # as if downloaded
    assert rainfall_service.dataset_path().name == registry.get("rainfall")["dataset"]


# ---------------- heavy-rain alert rule (model / persistence / both) ----------------
def test_heavy_alert_rule_function():
    assert rm.heavy_alert("model", 0.20, 0.09, 0.0) == (True, "20% chance of >= 64.5 mm")
    assert rm.heavy_alert("model", 0.05, 0.09, 120.0)[0] is False
    on, why = rm.heavy_alert("persistence", 0.01, 0.09, 80.0)
    assert on and "80 mm fell the day before" in why
    assert rm.heavy_alert("persistence", 0.50, 0.09, 10.0)[0] is False
    assert rm.heavy_alert("model_or_persistence", 0.01, 0.09, 70.0)[0] and rm.heavy_alert("model_or_persistence", 0.2, 0.09, 0.0)[0]


def test_choose_alert_rule_picks_persistence_when_it_wins_on_validation():
    from training import train_rainfall as tr
    rng = np.random.default_rng(0)
    n = 4000
    heavy = rng.random(n) < 0.02
    y = np.where(heavy, rm.HEAVY_CLASS, 0)
    lag_1 = np.where(heavy & (rng.random(n) < 0.8), 90.0, 1.0)      # yesterday heavy -> today heavy, 80% of the time
    p_heavy = np.clip(rng.random(n) * 0.2, 0, 1)                    # model probabilities carry no signal
    rule, thresholds, metrics = tr.choose_alert_rule(y, p_heavy, lag_1)
    assert rule == "persistence" and thresholds["persistence"] is None
    assert metrics["persistence"]["f1"] >= metrics["model"]["f1"]


def test_choose_alert_rule_picks_model_when_it_wins_on_validation():
    from training import train_rainfall as tr
    rng = np.random.default_rng(1)
    n = 4000
    heavy = rng.random(n) < 0.02
    y = np.where(heavy, rm.HEAVY_CLASS, 0)
    lag_1 = rng.random(n) * 20                                       # yesterday never heavy
    p_heavy = np.where(heavy, 0.6, 0.02) + rng.random(n) * 0.01      # model separates heavy days well
    rule, thresholds, _ = tr.choose_alert_rule(y, p_heavy, lag_1)
    assert rule == "model" and 0.03 < thresholds["model"] <= 0.62


def test_persistence_rule_is_used_by_prediction_and_corridor(client, monkeypatch):
    bundle = dict(registry.get("rainfall"), heavy_alert_rule="persistence")
    if "fallback" in bundle:                     # model trained with forecasts: dates without one use the fallback
        bundle["fallback"] = dict(bundle["fallback"], heavy_alert_rule="persistence")
    monkeypatch.setitem(registry._bundles, "rainfall", bundle)
    corridor_service.clear_cache()
    city = next(loc for loc in bundle["input_options"]["location"] if "Mumbai" in loc)
    wet = client.post("/api/predict/rainfall", json={"city": city, "forecast_date": "2025-07-15", "rainfall_last_7_days_mm": [80, 0, 0, 0, 0, 0, 0]}).json()
    dry = client.post("/api/predict/rainfall", json={"city": city, "forecast_date": "2025-07-15", "rainfall_last_7_days_mm": [60, 60, 60, 60, 60, 60, 60]}).json()
    assert wet["heavy_rain_alert"] is True and wet["alert_rule"] == "persistence" and "80 mm" in wet["alert_reason"]
    assert wet["flood_trigger_risk_level"] == "High"
    assert dry["heavy_rain_alert"] is False and dry["alert_reason"] == ""
    body = client.get("/api/corridor/forecast", params={"corridor_id": "mumbai-bengaluru", "date": "2025-07-20"}).json()
    assert body["heavy_alert_rule"] == "persistence"
    for seg in body["segments"]:
        assert seg["rainfall"]["heavy_alert"] == (seg["rainfall"]["last_7_days_mm"][0] >= 64.5)
        assert (seg["rainfall"]["rain_component"] == 1.0) == seg["rainfall"]["heavy_alert"]
    corridor_service.clear_cache()
