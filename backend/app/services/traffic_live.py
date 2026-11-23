"""Live traffic on each corridor segment from the TomTom Traffic Flow API (free key, no card).

For a point, TomTom's Flow Segment Data returns the road segment nearest to it with the current
average speed, the free-flow speed (speed with no traffic), travel times, a confidence value and a
road-closure flag. From that we compute the measured congestion of the segment *now*:

    speed_ratio = currentSpeed / freeFlowSpeed            (1.0 = free flow)
    congestion  = min(1, (1 - speed_ratio) / 0.5)          (traffic at half the free-flow speed = 1.0)
    level       = low (ratio >= 0.8) / medium (0.5 - 0.8) / high (< 0.5, or road closed)

The corridor forecast uses this measured value for the current hour instead of the traffic model's
estimate (the model's training data has no real traffic counts). Every fresh reading is also
appended to datasets/traffic_observations.csv, so a traffic model can later be trained on real data.

The API key is read from backend/.env (TOMTOM_API_KEY), used only here, and never logged or returned.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

from app import config

logger = logging.getLogger(__name__)

FLOW_URL = os.getenv("TOMTOM_FLOW_URL", "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json")
FREE_RATIO, HIGH_RATIO = 0.8, 0.5          # speed-ratio boundaries for low / medium / high
LOG_FIELDS = ["observed_at_ist", "date", "hour", "weekday", "segment_id", "lat", "lon", "frc", "current_speed_kmh",
              "free_flow_speed_kmh", "current_travel_time_s", "free_flow_travel_time_s", "confidence", "road_closure",
              "speed_ratio", "congestion", "level", "festival_event"]

_cache: Dict[str, tuple] = {}             # segment id -> (time fetched, observation)
_lock = threading.Lock()
_usage = {"day": None, "requests": 0}
_state = {"down_until": 0.0, "last_error": ""}


class TrafficAPIError(Exception):
    def __init__(self, message: str, stop_for_minutes: int = 10):
        super().__init__(message)
        self.stop_for_minutes = stop_for_minutes


def enabled() -> tuple:
    """(on?, reason shown to the user)."""
    if not config.LIVE_TRAFFIC:
        return False, "Live traffic is switched off (LIVE_TRAFFIC=false in backend/.env)."
    if not config.TOMTOM_API_KEY:
        return False, "No TomTom key: add TOMTOM_API_KEY=... to backend/.env and restart the backend."
    return True, "TomTom Traffic Flow API"


def fetch_flow(lat: float, lon: float) -> dict:
    """One TomTom request. Separate function so tests can replace it. Raises TrafficAPIError."""
    query = urllib.parse.urlencode({"key": config.TOMTOM_API_KEY, "point": f"{lat:.5f},{lon:.5f}", "unit": "KMPH"})
    request = urllib.request.Request(f"{FLOW_URL}?{query}", headers={"User-Agent": "freight-twin-ml/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=config.LIVE_TRAFFIC_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:   # messages never include the URL (it contains the key)
        if error.code in (401, 403):
            raise TrafficAPIError("TomTom rejected the key (HTTP %d). Check TOMTOM_API_KEY in backend/.env." % error.code, 60) from None
        if error.code == 429:
            raise TrafficAPIError("TomTom request limit reached (HTTP 429); using the traffic model for now.", 30) from None
        if error.code == 400:
            raise TrafficAPIError("TomTom found no road near this point (HTTP 400).", 0) from None
        raise TrafficAPIError("TomTom returned HTTP %d." % error.code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise TrafficAPIError("Could not reach TomTom (%s)." % getattr(error, "reason", error)) from None
    data = payload.get("flowSegmentData")
    if not data or "currentSpeed" not in data or "freeFlowSpeed" not in data:
        raise TrafficAPIError("TomTom returned no flow data for this point.", 0)
    return data


def measure(data: dict) -> dict:
    current, free = float(data["currentSpeed"]), float(data["freeFlowSpeed"])
    closed = bool(data.get("roadClosure", False))
    ratio = 0.0 if closed else (min(1.5, current / free) if free > 0 else 1.0)
    congestion = 1.0 if closed else max(0.0, min(1.0, (1.0 - ratio) / (1.0 - HIGH_RATIO)))
    level = "high" if closed or ratio < HIGH_RATIO else ("medium" if ratio < FREE_RATIO else "low")
    return {
        "source": "TomTom", "current_speed_kmh": round(current), "free_flow_speed_kmh": round(free),
        "current_travel_time_s": data.get("currentTravelTime"), "free_flow_travel_time_s": data.get("freeFlowTravelTime"),
        "speed_ratio": round(ratio, 3), "congestion": round(congestion, 3), "level": level,
        "confidence": round(float(data.get("confidence", 0.0)), 2), "road_closure": closed, "frc": data.get("frc", ""),
    }


def _take_budget(n: int) -> int:
    """How many of n requests may be made today without passing the daily budget."""
    today = config.today_ist().isoformat()
    with _lock:
        if _usage["day"] != today:
            _usage["day"], _usage["requests"] = today, 0
        allowed = max(0, min(n, config.TRAFFIC_DAILY_BUDGET - _usage["requests"]))
        _usage["requests"] += allowed
        return allowed


def _log(rows: List[dict]) -> None:
    if not rows:
        return
    path = config.TRAFFIC_LOG_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            new_file = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                writer.writerows(rows)
    except OSError as error:
        logger.warning("Could not save traffic observations (%s).", error)


def current(segments: List[dict], festival_event: str = "None") -> Dict[str, Optional[dict]]:
    """Live observation per segment id (None where unavailable). Cached for TRAFFIC_CACHE_MINUTES."""
    on, _ = enabled()
    result: Dict[str, Optional[dict]] = {s["id"]: None for s in segments}
    if not on or time.time() < _state["down_until"]:
        return result
    now = time.time()
    fresh_after = now - 60 * config.TRAFFIC_CACHE_MINUTES
    todo = []
    for seg in segments:
        cached = _cache.get(seg["id"])
        if cached and cached[0] >= fresh_after:
            result[seg["id"]] = cached[1]
        else:
            todo.append(seg)
    todo = todo[:_take_budget(len(todo))]
    if not todo:
        return result

    def one(seg):
        try:
            return seg, measure(fetch_flow(seg["mid"]["lat"], seg["mid"]["lon"])), None
        except TrafficAPIError as error:
            return seg, None, error

    stamp = datetime.now(config.IST).replace(microsecond=0)
    log_rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for seg, obs, error in pool.map(one, todo):
            if error is not None:
                if error.stop_for_minutes:
                    with _lock:
                        if time.time() >= _state["down_until"]:
                            logger.warning("Live traffic unavailable: %s", error)
                        _state["down_until"] = time.time() + 60 * error.stop_for_minutes
                _state["last_error"] = str(error)
                continue
            obs["observed_at"] = stamp.isoformat()
            _cache[seg["id"]] = (now, obs)
            result[seg["id"]] = obs
            log_rows.append({**obs, "observed_at_ist": stamp.isoformat(), "date": stamp.date().isoformat(), "hour": stamp.hour,
                             "weekday": stamp.strftime("%A"), "segment_id": seg["id"], "lat": seg["mid"]["lat"], "lon": seg["mid"]["lon"],
                             "festival_event": festival_event})
    _log(log_rows)
    return result


def status() -> dict:
    on, reason = enabled()
    path = config.TRAFFIC_LOG_FILE
    logged = 0
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            logged = max(0, sum(1 for _ in handle) - 1)
    today = config.today_ist().isoformat()
    return {
        "enabled": on, "provider": "TomTom Traffic Flow API" if on else None, "reason": reason,
        "paused": time.time() < _state["down_until"], "last_error": _state["last_error"] or None,
        "requests_today": _usage["requests"] if _usage["day"] == today else 0, "daily_budget": config.TRAFFIC_DAILY_BUDGET,
        "cache_minutes": config.TRAFFIC_CACHE_MINUTES, "observations_logged": logged, "log_file": path.name,
    }


def clear() -> None:
    _cache.clear()
    _state["down_until"], _state["last_error"] = 0.0, ""
