"""ML corridor forecast: runs the three trained models for every highway segment and hour.

For each segment (NH48 Mumbai-Bengaluru and Chennai-Bengaluru):
  * Rainfall model  - (optionally with the rain forecast made the day before, when the model was
                      trained with it and a forecast exists; otherwise its past-rain-only fallback)
                      the segment's own location when the model was trained on the per-segment
                      Open-Meteo file, otherwise the nearest of the 20 rainfall cities; fed the last
                      7 days of rain (dataset / Open-Meteo live / replay - see weather_source.py)
  * Traffic model   - nearest of the 8 road-dataset cities, for each hour 0-23, festival from the calendar,
                      weather taken from the rainfall prediction
  * Accident model  - same city, each hour, with the predicted weather and predicted traffic density

The three outputs are combined into one 0-1 "disruption risk" per hour with a transparent rule.
There are no disruption / delay records in the uploaded data, so this combination is NOT learned:
its weights are stated assumptions and are returned in the response.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from functools import lru_cache
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app import config
from app.ml_models import accident_model as am
from app.ml_models import rainfall_model as rm
from app.ml_models import traffic_model as tm
from app.ml_models.registry import registry
from app.services import rainfall_service, traffic_live, weather_source
from app.utils.validation import InvalidInputError

WEIGHTS = {"rain": 0.7, "accident": 0.5, "congestion": 0.3}
ROAD = {"road_type": "highway", "lanes": 4, "traffic_signal": False, "temperature": 28.0}
FORMULA = "risk = 1 - (1 - 0.7 x rain) x (1 - 0.5 x accident) x (1 - 0.3 x congestion)"
ASSUMPTIONS = [
    "No disruption or delay records exist in the uploaded data, so the combination rule above is an assumption, not a trained model.",
    "congestion, current hour today: measured by TomTom live traffic when a TomTom key is set: min(1, (1 - current speed / free-flow speed) / 0.5); other hours use the traffic model.",
    "rain = 1.0 when the heavy-rain alert is on (alert rule chosen on validation data); otherwise heavy-rain probability divided by the alert threshold, capped at 0.99.",
    "accident = accident model risk score; congestion = traffic model probability of 'high' congestion.",
    "Rainfall: each segment uses its own location when the per-segment Open-Meteo data is installed, otherwise the nearest rainfall city. Traffic and accident: nearest road-dataset city. Distances are shown per segment.",
    "Segments are treated as 4-lane highway without signals (lane data is not available for the corridor).",
    "Weather for the traffic and accident models is derived from the rainfall forecast: rain if P(>= 2.5 mm) >= 50%. Fog cannot be forecast from the data.",
    "Rainfall is a daily forecast, so its effect is the same for every hour of the day; hour-to-hour changes come from traffic and accident risk.",
]
_cache: Dict[Tuple[str, str], Tuple[float, dict]] = {}


@lru_cache(maxsize=1)
def corridors() -> dict:
    with open(config.CORRIDORS_FILE, encoding="utf-8") as handle:
        return json.load(handle)


def _km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _traffic_status_text() -> str:
    st = traffic_live.status()
    return st["last_error"] if st["paused"] and st["last_error"] else st["reason"]


def rain_cities() -> Dict[str, Tuple[float, float]]:
    return _rain_locations(str(rainfall_service.dataset_path()))


@lru_cache(maxsize=2)
def _rain_locations(path: str) -> Dict[str, Tuple[float, float]]:
    df = pd.read_csv(path, usecols=["Location", "Latitude", "Longitude"]).drop_duplicates("Location")
    return {r.Location: (float(r.Latitude), float(r.Longitude)) for r in df.itertuples()}


@lru_cache(maxsize=1)
def road_cities() -> Dict[str, Tuple[float, float]]:
    df = pd.read_csv(config.ACCIDENT_DATASET, usecols=["city", "latitude", "longitude"]).groupby("city").mean()
    return {city: (float(r.latitude), float(r.longitude)) for city, r in df.iterrows()}


def category_mean_mm() -> List[float]:
    return _category_mean_mm(str(rainfall_service.dataset_path()))


@lru_cache(maxsize=2)
def _category_mean_mm(path: str) -> List[float]:
    """Average rainfall of each IMD category in the dataset, used to turn probabilities into expected mm."""
    rain = pd.read_csv(path, usecols=["Rainfall_mm"])["Rainfall_mm"]
    means = rain.groupby(rm.rainfall_to_category(rain)).mean()
    return [float(means.get(i, 0.0)) for i in range(len(rm.CATEGORY_LABELS))]


def nearest(cities: Dict[str, Tuple[float, float]], lat: float, lon: float) -> Tuple[str, float]:
    name = min(cities, key=lambda c: _km(lat, lon, *cities[c]))
    return name, round(_km(lat, lon, *cities[name]), 1)


def _r(values, digits=3):
    return [round(float(v), digits) for v in values]


def forecast(corridor_id: str, day: date, use_cache: bool = True) -> dict:
    data = corridors()
    corridor = next((c for c in data["corridors"] if c["id"] == corridor_id), None)
    if corridor is None:
        raise InvalidInputError(f"Unknown corridor '{corridor_id}'. Use one of: {', '.join(c['id'] for c in data['corridors'])}.", "corridor_id")
    cache_key = (corridor_id, day.isoformat())
    cached = _cache.get(cache_key)
    if use_cache and cached and time.time() - cached[0] < config.FORECAST_CACHE_MINUTES * 60:
        return cached[1]

    rain_b, acc_b, tr_b = registry.get("rainfall"), registry.get("accident"), registry.get("traffic")
    segments = [s for s in data["segments"] if s["corridor_id"] == corridor_id]
    rcities, ccities = rain_cities(), road_cities()
    acc_cities = set(acc_b["input_options"]["city"])
    tr_cities = set(tr_b["input_options"]["city"])

    # ---- 1. rainfall model (one row per segment) ----
    rain_city = [nearest(rcities, s["mid"]["lat"], s["mid"]["lon"]) for s in segments]
    with ThreadPoolExecutor(max_workers=8) as pool:
        histories = list(pool.map(lambda pair: weather_source.last_7_days(pair[1][0], pair[0]["mid"]["lat"], pair[0]["mid"]["lon"], day), zip(segments, rain_city)))
        if rainfall_service.uses_forecast(rain_b):
            forecasts = list(pool.map(lambda t: weather_source.forecast_for(t[1][0], t[0]["mid"]["lat"], t[0]["mid"]["lon"], day, t[2][1]),
                                      zip(segments, rain_city, histories)))
        else:
            forecasts = [(None, "unused", "") for _ in segments]
    # Main model where a forecast made the day before exists, past-rain-only fallback elsewhere.
    profiles = [rainfall_service.profile(rain_b, fc[0] is not None) for fc in forecasts]
    rain_frames = [rm.build_request_frame(rc[0], day, h[0], fc[0]) for rc, h, fc in zip(rain_city, histories, forecasts)]
    rain_proba = np.zeros((len(segments), len(rm.CATEGORY_LABELS)))
    for with_fc in sorted({p["with_forecast"] for p in profiles}):
        idx = [i for i, p in enumerate(profiles) if p["with_forecast"] == with_fc]
        prof = profiles[idx[0]]
        batch = pd.concat([rain_frames[i] for i in idx], ignore_index=True)[prof["features"]]
        rain_proba[idx] = rainfall_service.class_probabilities(prof["pipeline"], batch, len(rm.CATEGORY_LABELS))
    threshold = float(rain_b["heavy_alert_threshold"])
    rule = rainfall_service.alert_rule(rain_b)
    means = np.array(category_mean_mm())

    # ---- 2. festival / event for the date ----
    festival_event, calendar_text = tm.resolve_event(day, None, config.SIGNIFICANT_DATES_DATASET)
    acc_festival = festival_event if festival_event in acc_b["input_options"]["festival"] else am.NO_FESTIVAL
    day_name = day.strftime("%A")

    # ---- 3. traffic + accident models (24 rows per segment) ----
    road_city, weather, traffic_rows = [], [], []
    for seg, probs in zip(segments, rain_proba):
        city, dist = nearest({c: v for c, v in ccities.items() if c in acc_cities and c in tr_cities}, seg["mid"]["lat"], seg["mid"]["lon"])
        road_city.append((city, dist))
        p_rain = 1 - probs[0]
        wet = p_rain >= 0.5
        w = ("rain", "low" if probs[2] + probs[3] >= 0.5 else "medium") if wet else ("clear", "high")
        weather.append(w)
        traffic_rows.append(tm.build_hourly_frame(city, day, ROAD["road_type"], ROAD["lanes"], ROAD["traffic_signal"], w[0], festival_event))
    traffic_frame = pd.concat(traffic_rows, ignore_index=True)
    tr_proba = tr_b["pipeline"].predict_proba(traffic_frame)
    tr_classes = list(tr_b["pipeline"].classes_)
    levels = np.asarray(tr_classes)[tr_proba.argmax(axis=1)]
    p_high = tr_proba[:, tr_classes.index("high")]
    p_high = p_high.astype(float).copy()
    levels = levels.astype("<U6").copy()

    # ---- 3b. live traffic (TomTom) replaces the model's estimate for the current hour, today only ----
    live_traffic: Dict[str, Optional[dict]] = {seg["id"]: None for seg in segments}
    now_hour = datetime.now(config.IST).hour
    if day == config.today_ist():
        live_traffic = traffic_live.current(segments, festival_event)
        for i, seg in enumerate(segments):
            obs = live_traffic.get(seg["id"])
            if obs is not None:
                levels[i * 24 + now_hour] = obs["level"]
                p_high[i * 24 + now_hour] = obs["congestion"]

    acc_rows = []
    for i, ((city, _), w) in enumerate(zip(road_city, weather)):
        acc_rows.append(am.build_hourly_frame(city, ROAD["road_type"], ROAD["lanes"], ROAD["traffic_signal"], w[0], w[1],
                                              ROAD["temperature"], [str(x) for x in levels[i * 24:(i + 1) * 24]], day_name, acc_festival))
    acc_risk = np.clip(acc_b["pipeline"].predict(pd.concat(acc_rows, ignore_index=True)), 0, 1)

    # ---- 4. combine + alerts ----
    out_segments, alerts = [], []
    sources = {"dataset": 0, "live": 0, "replay": 0}
    for i, seg in enumerate(segments):
        probs = rain_proba[i]
        p_heavy = float(probs[3])
        hist, source, source_detail = histories[i]
        prof = profiles[i]
        fc_mm, fc_source, fc_detail = forecasts[i]
        heavy_alert, alert_reason = rm.heavy_alert(prof["rule"], p_heavy, prof["threshold"], float(hist[0]), fc_mm)
        rain_c = 1.0 if heavy_alert else (min(0.99, p_heavy / prof["threshold"]) if prof["threshold"] > 0 else 0.0)
        acc = acc_risk[i * 24:(i + 1) * 24]
        cong = p_high[i * 24:(i + 1) * 24]
        risk = 1 - (1 - WEIGHTS["rain"] * rain_c) * (1 - WEIGHTS["accident"] * acc) * (1 - WEIGHTS["congestion"] * cong)
        sources[source] += 1
        peak_hour = int(np.argmax(risk))
        out_segments.append({
            "id": seg["id"], "name": seg["name"], "km": seg["km"],
            "cities": {"rainfall": {"name": rain_city[i][0], "km": rain_city[i][1]}, "road": {"name": road_city[i][0], "km": road_city[i][1]}},
            "rainfall": {
                "source": source, "source_detail": source_detail, "last_7_days_mm": _r(hist, 1),
                "probabilities": dict(zip(rm.CATEGORY_LABELS, _r(probs, 4))),
                "predicted_category": rm.CATEGORY_LABELS[int(np.argmax(probs))],
                "p_rain": round(float(1 - probs[0]), 4), "p_heavy": round(p_heavy, 4),
                "heavy_alert": bool(heavy_alert), "alert_reason": alert_reason, "expected_mm": round(float(probs @ means), 1),
                "rain_component": round(rain_c, 3),
                "forecast_mm": None if fc_mm is None else round(fc_mm, 1), "forecast_source": fc_source, "forecast_detail": fc_detail,
                "model_used": prof["model_name"], "alert_threshold": round(prof["threshold"], 4), "alert_rule": prof["rule"],
            },
            "live_traffic": live_traffic.get(seg["id"]),
            "conditions": {"weather": weather[i][0], "visibility": weather[i][1]},
            "hourly": {
                "risk": _r(risk), "accident_risk": _r(acc), "p_high_congestion": _r(cong),
                "congestion_level": [str(x) for x in levels[i * 24:(i + 1) * 24]],
            },
            "peak": {"hour": peak_hour, "risk": round(float(risk[peak_hour]), 3)},
        })
        if heavy_alert:
            alerts.append({"id": f"ML-{day}-{seg['id']}-rain", "severity": "High", "type": "heavy_rain", "segment_id": seg["id"], "hour": None,
                           "title": f"Heavy-rain alert on {seg['id']} {seg['name']}: {alert_reason}"})
        obs = live_traffic.get(seg["id"])
        if obs is not None and obs["road_closure"]:
            alerts.append({"id": f"ML-{day}-{seg['id']}-closed", "severity": "High", "type": "traffic", "segment_id": seg["id"], "hour": now_hour,
                           "title": f"Road closure reported on {seg['id']} {seg['name']} (TomTom live traffic)"})
        elif obs is not None and obs["level"] == "high" and obs["confidence"] >= 0.5:
            alerts.append({"id": f"ML-{day}-{seg['id']}-jam", "severity": "Moderate", "type": "traffic", "segment_id": seg["id"], "hour": now_hour,
                           "title": f"Slow traffic now on {seg['id']} {seg['name']}: {obs['current_speed_kmh']} km/h vs {obs['free_flow_speed_kmh']} km/h free flow (TomTom)"})
        if risk[peak_hour] >= 0.6:
            alerts.append({"id": f"ML-{day}-{seg['id']}-high", "severity": "High", "type": "disruption", "segment_id": seg["id"], "hour": peak_hour,
                           "title": f"High disruption risk on {seg['id']} {seg['name']} at {peak_hour:02d}:00 (risk {risk[peak_hour]:.2f})"})
        elif risk[peak_hour] >= 0.5:
            alerts.append({"id": f"ML-{day}-{seg['id']}-mod", "severity": "Moderate", "type": "disruption", "segment_id": seg["id"], "hour": peak_hour,
                           "title": f"Risk building on {seg['id']} {seg['name']} around {peak_hour:02d}:00 (risk {risk[peak_hour]:.2f})"})
    if festival_event != "None":
        busiest = max(out_segments, key=lambda s: max(s["hourly"]["p_high_congestion"]))
        alerts.append({"id": f"ML-{day}-event", "severity": "Moderate", "type": "festival", "segment_id": busiest["id"], "hour": None,
                       "title": f"Festival / event today: {calendar_text or festival_event} - congestion model adjusted"})
    alerts.append({"id": f"ML-{day}-refresh", "severity": "Info", "type": "info", "segment_id": segments[0]["id"], "hour": None,
                   "title": "ML forecast refreshed · rainfall input: " + ", ".join(f"{n} {k}" for k, n in sources.items() if n)})

    result = {
        "corridor_id": corridor_id, "corridor_name": corridor["name"], "date": day.isoformat(),
        "generated_at": datetime.now(config.IST).replace(microsecond=0).isoformat(),
        "event": {"festival": festival_event, "calendar_text": calendar_text},
        "rainfall_sources": sources, "heavy_alert_threshold": round(threshold, 4),
        "heavy_alert_rule": rule, "heavy_alert_rule_text": rm.ALERT_RULE_TEXT[rule],
        "traffic_sources": {"live": sum(1 for v in live_traffic.values() if v), "model": sum(1 for v in live_traffic.values() if not v)},
        "live_traffic_status": _traffic_status_text() if day == config.today_ist() else "Live traffic is only used for today.",
        "models": {"rainfall": rain_b["model_name"], "accident": acc_b["model_name"], "traffic": tr_b["model_name"]},
        "method": {"weights": WEIGHTS, "formula": FORMULA, "assumptions": ASSUMPTIONS, "road_assumptions": ROAD},
        "segments": out_segments, "alerts": alerts,
    }
    _cache[cache_key] = (time.time(), result)
    return result


def clear_cache() -> None:
    _cache.clear()
