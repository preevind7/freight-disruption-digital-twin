"""Tests for the optional "rain forecast made the day before" input.

No internet: Open-Meteo is replaced by fakes. The fake rainfall and fake forecasts only check the
plumbing (download -> CSV -> training -> serving); they are never shipped or used to report accuracy.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import pytest

from app import config
from app.ml_models import rainfall_model as rm
from app.ml_models.registry import registry
from app.services import corridor_service, rainfall_service, weather_source
from training import download_forecasts as dfc
from training import download_rainfall as dl
from tests.test_rainfall_download import FakeClock, fake_archive, pacer


def fake_previous_runs(daily_by_location: dict, noise=0.25, calls=None, drop_hours=()):
    """Hourly UTC values whose IST-day totals are the true daily rain x (1 +- noise): an informative forecast."""
    coords = {}

    def fetch(lat, lon, start, end):
        if calls is not None:
            calls.append((lat, lon, start, end))
        series = daily_by_location[(round(lat, 4), round(lon, 4))]
        rng = np.random.default_rng(int(abs(lat * 100 + lon)))
        out = {}
        t = datetime(start.year, start.month, start.day) - timedelta(days=1)
        stop = datetime(end.year, end.month, end.day, 23)
        while t <= stop:
            ist_day = (t + timedelta(hours=5, minutes=30)).date()
            total = series.get(ist_day)
            key = t.strftime("%Y-%m-%dT%H:%M")
            out[key] = None if total is None or key in drop_hours else round(total * (1 + noise * (rng.random() * 2 - 1)) / 24.0, 3)
            t += timedelta(hours=1)
        return out
    return fetch


def daily_lookup(rain_csv):
    df = pd.read_csv(rain_csv)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return {(round(lat, 4), round(lon, 4)): dict(zip(g["Date"], g["Rainfall_mm"])) for (lat, lon), g in df.groupby(["Latitude", "Longitude"])}


# ---------------- IST day totals ----------------
def test_daily_ist_totals_sums_utc_hours_into_indian_days():
    hourly = {}
    t = datetime(2024, 6, 30, 0)
    while t <= datetime(2024, 7, 2, 23):
        hourly[t.strftime("%Y-%m-%dT%H:%M")] = 1.0
        t += timedelta(hours=1)
    totals = weather_source.daily_ist_totals(hourly, date(2024, 7, 1), date(2024, 7, 2))
    assert totals.tolist() == [24.0, 24.0]          # 18:30 UTC (previous day) ... 18:30 UTC = one IST day
    hourly["2024-07-01T03:00"] = None               # one missing hour -> that day is left empty
    totals = weather_source.daily_ist_totals(hourly, date(2024, 7, 1), date(2024, 7, 2))
    assert np.isnan(totals.iloc[0]) and totals.iloc[1] == 24.0


def test_forecast_rule_and_request_frame():
    assert rm.heavy_alert("forecast", 0.0, 0.1, 0.0, 70.0)[0] is True
    assert rm.heavy_alert("forecast", 0.9, 0.1, 99.0, None)[0] is False
    frame = rm.build_request_frame("X", date(2024, 7, 1), [0] * 7)
    assert list(frame.columns) == rm.FEATURES_FORECAST and np.isnan(frame.loc[0, rm.FORECAST_FEATURE])
    assert rm.build_request_frame("X", date(2024, 7, 1), [0] * 7, 12.5).loc[0, rm.FORECAST_FEATURE] == 12.5


# ---------------- download -> train -> serve ----------------
@pytest.fixture(scope="module")
def forecast_setup(tmp_path_factory):
    from training import common, train_rainfall

    tmp = tmp_path_factory.mktemp("rain-forecast")
    datasets, models, reports = tmp / "datasets", tmp / "models", tmp / "reports"
    rain_csv = datasets / config.RAINFALL_CORRIDOR_DATASET.name
    fc_csv = datasets / config.RAIN_FORECAST_DATASET.name
    dl.run(date(2022, 1, 1), date(2026, 6, 30), rain_csv, datasets / "cache", pacer(FakeClock(), 10**7, 10**7), fetch=fake_archive(), say=lambda *_: None)
    calls = []
    dfc.run(date(2024, 1, 1), date(2026, 6, 30), fc_csv, datasets / "fc-cache", pacer(FakeClock(), 10**7, 10**7),
            fetch=fake_previous_runs(daily_lookup(rain_csv), calls=calls), say=lambda *_: None)
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "RAINFALL_DATASET", rain_csv)
    mp.setattr(config, "RAIN_FORECAST_DATASET", fc_csv)
    mp.setattr(train_rainfall, "RAINFALL_MODEL_PATH", models / "rainfall_model.joblib")
    mp.setattr(common, "REPORTS_DIR", reports)
    try:
        train_rainfall.main()
    finally:
        mp.undo()
    report = json.loads((reports / "rainfall_report.json").read_text(encoding="utf-8"))
    return {"bundle": joblib.load(models / "rainfall_model.joblib"), "report": report, "datasets": datasets,
            "fc_csv": fc_csv, "calls": calls, "md": (reports / "rainfall_report.md").read_text(encoding="utf-8")}


def test_forecast_download_chunks_and_output(forecast_setup):
    df = pd.read_csv(forecast_setup["fc_csv"])
    assert df["Location"].nunique() == 35 and df["Forecast_mm"].notna().mean() > 0.99
    assert df["Date"].min() == "2024-01-01" and df["Date"].max() == "2026-06-30"
    per_segment = len(forecast_setup["calls"]) / 35
    assert per_segment == 8                                       # 912 days in chunks of 120 days
    assert all(start - timedelta(days=0) <= end for _, _, start, end in forecast_setup["calls"])


def test_forecast_download_resumes_with_only_new_days(forecast_setup, tmp_path):
    rain = daily_lookup(forecast_setup["datasets"] / config.RAINFALL_CORRIDOR_DATASET.name)
    cache, out = tmp_path / "c", tmp_path / "f.csv"
    dfc.run(date(2024, 1, 1), date(2024, 3, 31), out, cache, pacer(FakeClock()), fetch=fake_previous_runs(rain), say=lambda *_: None)
    calls = []
    dfc.run(date(2024, 1, 1), date(2024, 4, 10), out, cache, pacer(FakeClock()), fetch=fake_previous_runs(rain, calls=calls), say=lambda *_: None)
    assert len(calls) == 35 and all(c[2] == date(2024, 4, 1) for c in calls)


def test_training_compares_with_and_without_forecast_on_same_days(forecast_setup):
    report, bundle = forecast_setup["report"], forecast_setup["bundle"]
    assert report["split"]["validation"]["from"] == "2025-01-01" and report["split"]["test"]["from"] == "2026-01-01"
    keys = list(report["validation_comparison"])
    assert any(k.endswith("· past rain") for k in keys) and any(k.endswith("· past rain + forecast") for k in keys)
    assert "forecast alone (category of the forecast made the day before)" in report["test_baselines"]
    # the fake forecast is informative, so the forecast model must win and keep a past-rain fallback
    assert report["uses_forecast"] is True and bundle["uses_forecast"] is True
    assert bundle["features"] == rm.FEATURES_FORECAST and bundle["fallback"]["features"] == rm.FEATURES
    assert bundle["forecast_dataset"] == config.RAIN_FORECAST_DATASET.name
    assert "## Fallback when no forecast exists for a date" in forecast_setup["md"]
    assert report["test_metrics"]["log_loss"] < report["fallback"]["test_metrics_same_days"]["log_loss"]


def test_training_ignores_forecast_file_without_enough_rows(forecast_setup, tmp_path, monkeypatch):
    from training import common, train_rainfall
    short = tmp_path / "short.csv"
    df = pd.read_csv(forecast_setup["fc_csv"])
    df[df["Date"] >= "2025-12-01"].to_csv(short, index=False)                 # nothing in the training years
    monkeypatch.setattr(config, "RAINFALL_DATASET", forecast_setup["datasets"] / config.RAINFALL_CORRIDOR_DATASET.name)
    monkeypatch.setattr(config, "RAIN_FORECAST_DATASET", short)
    monkeypatch.setattr(train_rainfall, "RAINFALL_MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(common, "REPORTS_DIR", tmp_path / "r")
    train_rainfall.main()
    bundle = joblib.load(tmp_path / "m.joblib")
    assert bundle["uses_forecast"] is False and "fallback" not in bundle and bundle["features"] == rm.FEATURES


@pytest.fixture
def use_forecast_model(forecast_setup, monkeypatch):
    monkeypatch.setattr(config, "DATASETS_DIR", forecast_setup["datasets"])
    monkeypatch.setattr(config, "LIVE_WEATHER", False)
    monkeypatch.setitem(registry._bundles, "rainfall", forecast_setup["bundle"])
    corridor_service.clear_cache()
    weather_source._live_cache.clear()
    yield forecast_setup["bundle"]
    corridor_service.clear_cache()


def test_corridor_uses_forecast_model_when_forecast_exists(client, use_forecast_model):
    body = client.get("/api/corridor/forecast", params={"corridor_id": "mumbai-bengaluru", "date": "2026-06-20"}).json()
    seg = next(s for s in body["segments"] if s["id"] == "S03")
    expected = pd.read_csv(config.DATASETS_DIR / config.RAIN_FORECAST_DATASET.name).query("Segment_ID == 'S03' and Date == '2026-06-20'")["Forecast_mm"].iloc[0]
    assert seg["rainfall"]["forecast_source"] == "dataset" and seg["rainfall"]["forecast_mm"] == round(float(expected), 1)
    assert seg["rainfall"]["model_used"] == use_forecast_model["model_name"]


def test_corridor_falls_back_without_forecast(client, use_forecast_model):
    # 2023: rainfall exists, forecasts (archived from 2024) do not -> past-rain-only fallback, said so
    body = client.get("/api/corridor/forecast", params={"corridor_id": "mumbai-bengaluru", "date": "2023-07-10"}).json()
    for seg in body["segments"]:
        assert seg["rainfall"]["forecast_mm"] is None and seg["rainfall"]["forecast_source"] == "none"
        assert seg["rainfall"]["model_used"] == use_forecast_model["fallback"]["model_name"]
        assert seg["rainfall"]["alert_rule"] == use_forecast_model["fallback"]["heavy_alert_rule"]


def test_predict_with_and_without_forecast_and_history(client, use_forecast_model):
    loc = "S03 Khalapur – Lonavala (Bhor Ghat)"
    hist = client.get("/api/rainfall/history", params={"city": loc, "forecast_date": "2026-06-20"}).json()
    assert hist["forecast_rain_mm"] is not None
    base = {"city": loc, "forecast_date": "2026-06-20", "rainfall_last_7_days_mm": hist["rainfall_last_7_days_mm"]}
    with_fc = client.post("/api/predict/rainfall", json={**base, "forecast_rain_mm": hist["forecast_rain_mm"]}).json()
    without = client.post("/api/predict/rainfall", json=base).json()
    assert with_fc["used_forecast"] is True and with_fc["model_used"] == use_forecast_model["model_name"]
    assert without["used_forecast"] is False and without["model_used"] == use_forecast_model["fallback"]["model_name"]
    assert "past-rain-only model" in without["explanation"]
    heavy_fc = client.post("/api/predict/rainfall", json={**base, "forecast_rain_mm": 150}).json()
    dry_fc = client.post("/api/predict/rainfall", json={**base, "forecast_rain_mm": 0}).json()
    assert heavy_fc["heavy_rain_probability"] > dry_fc["heavy_rain_probability"]     # the forecast input matters
    assert client.post("/api/predict/rainfall", json={**base, "forecast_rain_mm": -1}).status_code == 422


def test_live_forecast_used_for_today_when_not_in_file(client, use_forecast_model, monkeypatch):
    today = config.today_ist()
    monkeypatch.setattr(config, "LIVE_WEATHER", True)
    monkeypatch.setattr(weather_source, "fetch_open_meteo", lambda lat, lon, s, e: {(s + timedelta(days=i)).isoformat(): 1.0 for i in range((e - s).days + 1)})
    hours = {}
    t = datetime(today.year, today.month, today.day) - timedelta(days=1)
    while t <= datetime(today.year, today.month, today.day, 23):
        hours[t.strftime("%Y-%m-%dT%H:%M")] = 0.5
        t += timedelta(hours=1)
    monkeypatch.setattr(weather_source, "fetch_previous_runs", lambda lat, lon, s, e, timeout=None: hours)
    value, source, _ = weather_source.forecast_for("S03 Khalapur – Lonavala (Bhor Ghat)", 18.79, 73.34, today, "live")
    assert (value, source) == (12.0, "live")
    assert weather_source.forecast_for("S03 Khalapur – Lonavala (Bhor Ghat)", 18.79, 73.34, today + timedelta(days=1), "live")[1] == "none"
