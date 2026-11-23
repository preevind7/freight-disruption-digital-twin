"""Where the rainfall model's input (the 7 days of rain before a date) comes from.

Order of preference:
  1. "dataset" - observed values in the rainfall file the model was trained on
                 (corridor_rainfall_open_meteo.csv per segment, or the uploaded city file)
  2. "live"    - Open-Meteo (free, no API key) daily precipitation at the segment's location,
                 used for recent / current dates when the server has internet access
  3. "replay"  - the same calendar date from the latest year in the dataset, clearly labelled,
                 so the app still works offline

The optional forecast input (rain forecast for the day, made the day before) follows the same idea:
the downloaded forecast file, else Open-Meteo's Previous Runs API live, else none (the model then
uses its past-rain-only fallback) - see forecast_for().
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app import config
from app.services import rainfall_service

logger = logging.getLogger(__name__)
LAG_DAYS = 7
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_VARIABLE = "precipitation_previous_day1"   # value predicted 24 h before each hour
IST_OFFSET = np.timedelta64(330, "m")   # IST = UTC + 5:30

_live_cache: Dict[Tuple[float, float, str], List[float]] = {}
_live_down_until = 0.0  # simple circuit breaker after a failed call (7-day history)
_fc_down_until = 0.0    # separate breaker for the forecast API, so one failing does not switch off the other
_live_lock = threading.Lock()


_table_lock = threading.Lock()   # tables are built once; lookups are plain dict reads (safe from many threads)


@lru_cache(maxsize=2)
def _rain_table(path: str) -> Dict[str, Dict[date, float]]:
    df = pd.read_csv(path, usecols=["Date", "Location", "Rainfall_mm"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return {loc: dict(zip(g["Date"], g["Rainfall_mm"].astype(float))) for loc, g in df.groupby("Location")}


def _rain_series(city: str) -> Optional[Dict[date, float]]:
    path = rainfall_service.dataset_path()
    if not path.exists():
        return None
    with _table_lock:
        table = _rain_table(str(path))
    return table.get(city)


def _dataset_window(city: str, day: date) -> Optional[List[float]]:
    series = _rain_series(city)
    if not series:
        return None
    days = [day - timedelta(days=i) for i in range(1, LAG_DAYS + 1)]
    if all(d in series for d in days):
        return [float(series[d]) for d in days]
    return None


def fetch_open_meteo(lat: float, lon: float, start: date, end: date) -> Dict[str, Optional[float]]:
    """Daily precipitation (mm) per date from Open-Meteo. Separate function so tests can replace it."""
    query = urllib.parse.urlencode({
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata", "start_date": start.isoformat(), "end_date": end.isoformat(),
    })
    request = urllib.request.Request(f"{OPEN_METEO_URL}?{query}", headers={"User-Agent": "freight-twin-ml/1.0"})
    with urllib.request.urlopen(request, timeout=config.LIVE_WEATHER_TIMEOUT) as response:
        payload = json.loads(response.read().decode("utf-8"))
    daily = payload.get("daily") or {}
    return dict(zip(daily.get("time", []), daily.get("precipitation_sum", [])))


def _live_window(lat: float, lon: float, day: date) -> Optional[List[float]]:
    global _live_down_until
    if not config.LIVE_WEATHER or time.time() < _live_down_until:
        return None
    today = config.today_ist()
    # Open-Meteo's forecast API covers roughly the last 3 months and the next 2 weeks.
    if not (today - timedelta(days=85) <= day <= today + timedelta(days=14)):
        return None
    key = (round(lat, 2), round(lon, 2), day.isoformat())
    if key in _live_cache:
        return _live_cache[key]
    days = [day - timedelta(days=i) for i in range(1, LAG_DAYS + 1)]
    try:
        values = fetch_open_meteo(lat, lon, days[-1], days[0])
        window = [values.get(d.isoformat()) for d in days]
        if any(v is None for v in window):
            return None
        window = [max(0.0, float(v)) for v in window]
    except Exception as exc:  # no internet, firewall, timeout, bad response ...
        with _live_lock:
            if time.time() >= _live_down_until:
                logger.warning("Live weather unavailable (%s); using dataset replay for 10 minutes.", exc)
            _live_down_until = time.time() + 600
        return None
    _live_cache[key] = window
    return window


def _replay_window(city: str, day: date) -> Optional[Tuple[List[float], date]]:
    series = _rain_series(city)
    if not series:
        return None
    last_year = max(series).year
    for year in range(last_year, last_year - 12, -1):
        try:
            replay_day = day.replace(year=year)
        except ValueError:  # 29 February
            replay_day = day.replace(year=year, day=28)
        window = _dataset_window(city, replay_day)
        if window is not None:
            return window, replay_day
    return None


def last_7_days(city: str, lat: float, lon: float, day: date) -> Tuple[List[float], str, str]:
    """Returns (rain for day-1 ... day-7 in mm, source, human-readable description)."""
    window = _dataset_window(city, day)
    if window is not None:
        return window, "dataset", f"Observed rainfall for {city} from {rainfall_service.dataset_path().name}"
    window = _live_window(lat, lon, day)
    if window is not None:
        return window, "live", "Open-Meteo daily precipitation at this segment's location (last 7 days)"
    replay = _replay_window(city, day)
    if replay is not None:
        window, replay_day = replay
        return window, "replay", (
            f"No live weather available: replaying observed {city} rainfall from the 7 days before "
            f"{replay_day.isoformat()} ({rainfall_service.dataset_path().name})"
        )
    raise RuntimeError(f"No rainfall history available for {city}.")


# ---------------- optional input: rain forecast made the day before ----------------
def fetch_previous_runs(lat: float, lon: float, start: date, end: date, timeout: Optional[float] = None) -> Dict[str, Optional[float]]:
    """Hourly precipitation predicted 24 h ahead (UTC hours) covering the IST days start..end."""
    query = urllib.parse.urlencode({
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "hourly": FORECAST_VARIABLE,
        "start_date": (start - timedelta(days=1)).isoformat(), "end_date": end.isoformat(),   # an IST day starts 18:30 UTC the day before
    })
    request = urllib.request.Request(f"{PREVIOUS_RUNS_URL}?{query}", headers={"User-Agent": "freight-twin-ml/1.0"})
    with urllib.request.urlopen(request, timeout=timeout or config.LIVE_WEATHER_TIMEOUT) as response:
        payload = json.loads(response.read().decode("utf-8"))
    hourly = payload.get("hourly") or {}
    times, values = hourly.get("time", []), hourly.get(FORECAST_VARIABLE, [])
    if len(times) != len(values):
        raise ValueError(f"Open-Meteo returned a malformed {FORECAST_VARIABLE} series.")
    return dict(zip(times, values))


def daily_ist_totals(hourly: Dict[str, Optional[float]], start: date, end: date) -> pd.Series:
    """Sum UTC hourly values into IST days; a day needs all 24 hours, otherwise it stays empty (NaN)."""
    index = pd.date_range(start, end, freq="D")
    if not hourly:
        return pd.Series(np.nan, index=index, dtype="float64")
    s = pd.Series(list(hourly.values()), index=pd.to_datetime(list(hourly.keys())), dtype="float64")
    grouped = s.groupby((s.index + IST_OFFSET).normalize())
    totals = grouped.sum(min_count=1).where(grouped.count() == 24)
    return totals.reindex(index).clip(lower=0.0).round(2)


@lru_cache(maxsize=2)
def _forecast_table(path: str) -> Dict[Tuple[str, date], float]:
    df = pd.read_csv(path, usecols=["Date", "Location", "Forecast_mm"]).dropna(subset=["Forecast_mm"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return {(loc, d): float(v) for loc, d, v in zip(df["Location"], df["Date"], df["Forecast_mm"])}


def _forecast_dataset_value(city: str, day: date) -> Optional[float]:
    name = rainfall_service.forecast_dataset_name()
    path = config.DATASETS_DIR / name if name else None
    if path is None or not path.exists():
        return None
    with _table_lock:
        table = _forecast_table(str(path))
    return table.get((city, day))


def _live_forecast(lat: float, lon: float, day: date) -> Optional[float]:
    global _fc_down_until
    if not config.LIVE_WEATHER or time.time() < _fc_down_until:
        return None
    today = config.today_ist()
    if not (today - timedelta(days=85) <= day <= today):   # a full day forecast made 24 h ahead exists up to today
        return None
    key = (round(lat, 2), round(lon, 2), "fc-" + day.isoformat())
    if key in _live_cache:
        return _live_cache[key]
    try:
        value = daily_ist_totals(fetch_previous_runs(lat, lon, day, day), day, day).iloc[0]
    except Exception as exc:
        with _live_lock:
            if time.time() >= _fc_down_until:
                logger.warning("Live forecast unavailable (%s); using the past-rain-only model for 10 minutes.", exc)
            _fc_down_until = time.time() + 600
        return None
    value = None if pd.isna(value) else float(value)
    _live_cache[key] = value
    return value


def forecast_for(city: str, lat: float, lon: float, day: date, history_source: str) -> Tuple[Optional[float], str, str]:
    """Rain forecast for `day` made the day before: (mm or None, source, description).
    When the 7-day history is a replay, the forecast of the same replayed date is used, so inputs match."""
    if history_source == "replay":
        replay = _replay_window(city, day)
        if replay is not None:
            value = _forecast_dataset_value(city, replay[1])
            if value is not None:
                return value, "replay", f"Forecast made the day before {replay[1].isoformat()} (replayed, from the forecast file)"
        return None, "none", "No forecast for the replayed date; past-rain-only model used"
    value = _forecast_dataset_value(city, day)
    if value is not None:
        return value, "dataset", f"Forecast made the day before, from {rainfall_service.forecast_dataset_name()}"
    value = _live_forecast(lat, lon, day)
    if value is not None:
        return value, "live", "Open-Meteo forecast for this segment made 24 h ahead (Previous Runs API)"
    return None, "none", "No forecast made the day before is available for this date; past-rain-only model used"
