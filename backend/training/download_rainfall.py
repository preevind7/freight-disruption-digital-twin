"""Download REAL daily rainfall for every corridor segment from Open-Meteo.

Run from the backend folder (needs internet, no API key):
    python -m training.download_rainfall

Source: Open-Meteo Historical Weather API (https://open-meteo.com/en/docs/historical-weather-api),
which serves ERA5 / ERA5-Land reanalysis. Weather data by Open-Meteo.com, licensed CC BY 4.0;
the free API is for non-commercial use.

Free limits: 600 calls/minute, 5,000/hour, 10,000/day. A request longer than 2 weeks for
one location counts as several calls (about 1 per 14 days), so ~7 years x 35 segments is
~6,300 calls. The script paces itself below the hourly limit, saves each segment as soon as
it is downloaded, and simply continues where it stopped if you run it again.

Output: datasets/corridor_rainfall_open_meteo.csv with the columns the training code uses
(Date, Location, Latitude, Longitude, Rainfall_mm) plus Segment_ID, Corridor and Source.
Running it again later only downloads the new days.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import pandas as pd

from app import config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
SOURCE = "Open-Meteo Historical Weather API (ERA5 reanalysis), CC BY 4.0"
DEFAULT_START = date(2020, 1, 1)
ARCHIVE_DELAY_DAYS = 6          # the archive is usually complete up to ~5 days ago
HOURLY_BUDGET = 4000            # stay well under the 5,000 calls/hour free limit
DAILY_BUDGET = 9000             # leave room for the app's live weather calls
HOUR = 3600.0


class DailyLimitReached(Exception):
    pass


class RainfallDownloadError(Exception):
    pass


def call_cost(start: date, end: date) -> float:
    """Open-Meteo counts one call per 14 days of data for one location (fractional)."""
    days = (end - start).days + 1
    return max(1.0, days / 14.0)


def fetch_archive(lat: float, lon: float, start: date, end: date, timeout: float = 60.0) -> Dict[str, Optional[float]]:
    """One HTTP request. Returns {"YYYY-MM-DD": mm or None}. Raises urllib errors unchanged."""
    query = urllib.parse.urlencode({
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata", "start_date": start.isoformat(), "end_date": end.isoformat(),
    })
    request = urllib.request.Request(f"{ARCHIVE_URL}?{query}", headers={"User-Agent": "freight-twin-ml/1.0 (training data download)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    daily = payload.get("daily") or {}
    times, values = daily.get("time", []), daily.get("precipitation_sum", [])
    if not times or len(times) != len(values):
        raise RainfallDownloadError("Open-Meteo returned no daily precipitation for this location.")
    return dict(zip(times, values))


class Pacer:
    """Keeps the weighted calls within the hourly and daily budgets (sliding windows)."""

    def __init__(self, hourly: float, daily: float, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, say: Callable[[str], None] = print):
        self.hourly, self.daily = hourly, daily
        self.clock, self.sleep, self.say = clock, sleep, say
        self.history: Deque[Tuple[float, float]] = deque()

    def _used(self, window: float) -> float:
        now = self.clock()
        return sum(cost for t, cost in self.history if now - t < window)

    def wait_for(self, cost: float) -> None:
        while self.history and self.clock() - self.history[0][0] >= 24 * HOUR:
            self.history.popleft()
        if self._used(24 * HOUR) + cost > self.daily:
            raise DailyLimitReached()
        while self._used(HOUR) > 0 and self._used(HOUR) + cost > self.hourly:
            now = self.clock()
            oldest = next(t for t, _ in self.history if now - t < HOUR)
            wait = oldest + HOUR - now + 5
            self.say(f"  Pausing {wait / 60:.0f} min to stay under Open-Meteo's free hourly limit "
                     f"(progress is saved; you can also stop with Ctrl+C and run again later).")
            self.sleep(wait)

    def record(self, cost: float) -> None:
        self.history.append((self.clock(), cost))


def _rate_limit_reason(error: urllib.error.HTTPError) -> str:
    try:
        return json.loads(error.read().decode("utf-8")).get("reason", "")
    except Exception:
        return ""


def fetch_with_retry(fetch, pacer: Pacer, lat: float, lon: float, start: date, end: date) -> Dict[str, Optional[float]]:
    cost = call_cost(start, end)
    for attempt in range(6):
        pacer.wait_for(cost)
        try:
            values = fetch(lat, lon, start, end)
            pacer.record(cost)
            return values
        except urllib.error.HTTPError as error:
            pacer.record(cost)
            if error.code != 429:
                raise RainfallDownloadError(f"Open-Meteo returned HTTP {error.code}: {_rate_limit_reason(error) or error.reason}") from error
            reason = _rate_limit_reason(error).lower()
            if "daily" in reason:
                raise DailyLimitReached() from error
            if "hourly" in reason:
                now = datetime.now()
                wait = ((60 - now.minute) * 60 - now.second) + 60
                pacer.say(f"  Open-Meteo hourly limit reached; waiting {wait / 60:.0f} min (progress is saved).")
            else:
                wait = 65
                pacer.say("  Open-Meteo per-minute limit reached; waiting about a minute.")
            pacer.sleep(wait)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt >= 3:
                raise RainfallDownloadError(f"Could not reach Open-Meteo ({error}). Check the internet connection and run again.") from error
            pacer.say(f"  Network problem ({error}); retrying in {10 * (attempt + 1)} s.")
            pacer.sleep(10 * (attempt + 1))
    raise RainfallDownloadError("Open-Meteo kept refusing the request; run the script again later.")


def clean_series(values: Dict[str, Optional[float]], start: date, end: date) -> pd.Series:
    """Daily mm for start..end. Trailing days that the archive has not filled yet are dropped;
    a gap in the middle is an error (lag features need consecutive days)."""
    days = pd.date_range(start, end, freq="D")
    series = pd.Series([values.get(d.strftime("%Y-%m-%d")) for d in days], index=days, dtype="float64")
    last_valid = series.last_valid_index()
    if last_valid is None:
        raise RainfallDownloadError(f"No rainfall values between {start} and {end}.")
    series = series.loc[:last_valid]
    if series.isna().any():
        gap = series[series.isna()].index[0].date()
        raise RainfallDownloadError(f"Open-Meteo has a missing day ({gap}); run again later.")
    return series.clip(lower=0.0).round(2)


def _cache_file(cache_dir: Path, segment_id: str) -> Path:
    return cache_dir / f"{segment_id}.csv"


def _read_cache(path: Path) -> Optional[pd.Series]:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    return pd.Series(df["Rainfall_mm"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["Date"]))


def _write_cache(path: Path, series: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame({"Date": series.index.strftime("%Y-%m-%d"), "Rainfall_mm": series.to_numpy()}).to_csv(tmp, index=False)
    tmp.replace(path)


def download_segment(segment: dict, start: date, end: date, cache_dir: Path, pacer: Pacer, fetch=fetch_archive) -> Tuple[pd.Series, str]:
    """Returns (daily series, what happened). Uses the cache and downloads only missing days."""
    path = _cache_file(cache_dir, segment["id"])
    cached = _read_cache(path)
    lat, lon = segment["mid"]["lat"], segment["mid"]["lon"]
    if cached is not None and len(cached) and cached.index.min().date() <= start:
        cached = cached.loc[pd.Timestamp(start):]
        have_until = cached.index.max().date()
        if have_until >= end:
            return cached.loc[:pd.Timestamp(end)], "already downloaded"
        fresh = clean_series(fetch_with_retry(fetch, pacer, lat, lon, have_until + timedelta(days=1), end), have_until + timedelta(days=1), end)
        series = pd.concat([cached, fresh])
        _write_cache(path, series)
        return series, f"added {len(fresh)} new days"
    series = clean_series(fetch_with_retry(fetch, pacer, lat, lon, start, end), start, end)
    _write_cache(path, series)
    return series, f"downloaded {len(series)} days"


def build_dataset(segments: List[dict], series_by_id: Dict[str, pd.Series], corridor_names: Dict[str, str]) -> pd.DataFrame:
    """One table, every segment cut to the same last day so all series are consecutive and equal length."""
    common_end = min(s.index.max() for s in series_by_id.values())
    frames = []
    for seg in segments:
        s = series_by_id[seg["id"]].loc[:common_end]
        frames.append(pd.DataFrame({
            "Date": s.index.strftime("%Y-%m-%d"),
            "Location": f"{seg['id']} {seg['name']}",
            "Segment_ID": seg["id"],
            "Corridor": corridor_names.get(seg["corridor_id"], seg["corridor_id"]),
            "Latitude": seg["mid"]["lat"],
            "Longitude": seg["mid"]["lon"],
            "Rainfall_mm": s.to_numpy(),
            "Source": SOURCE,
        }))
    return pd.concat(frames, ignore_index=True)


def load_segments() -> Tuple[List[dict], Dict[str, str]]:
    with open(config.CORRIDORS_FILE, encoding="utf-8") as handle:
        data = json.load(handle)
    return data["segments"], {c["id"]: c["name"] for c in data["corridors"]}


def run(start: date, end: date, out: Path, cache_dir: Path, pacer: Pacer, fetch=fetch_archive, say=print) -> Optional[pd.DataFrame]:
    segments, corridor_names = load_segments()
    total_cost = sum(call_cost(start, end) for _ in segments)
    say(f"Real rainfall for {len(segments)} corridor segments, {start} to {end} (Open-Meteo, free, no key).")
    say(f"About {math.ceil(total_cost):,} Open-Meteo calls if nothing is saved yet; saved segments are skipped.\n")
    series_by_id: Dict[str, pd.Series] = {}
    try:
        for n, seg in enumerate(segments, 1):
            series, what = download_segment(seg, start, end, cache_dir, pacer, fetch)
            series_by_id[seg["id"]] = series
            say(f"[{n:>2}/{len(segments)}] {seg['id']} {seg['name']}: {what} "
                f"(wettest day {series.max():.1f} mm, heavy-rain days {(series >= 64.5).sum()})")
    except DailyLimitReached:
        say(f"\nStopped: Open-Meteo's free daily limit is used up. {len(series_by_id)} of {len(segments)} segments are saved.")
        say("Run the same command again tomorrow; it continues where it stopped.")
        return None
    except KeyboardInterrupt:
        say(f"\nStopped by you. {len(series_by_id)} of {len(segments)} segments are saved; run again to continue.")
        return None

    dataset = build_dataset(segments, series_by_id, corridor_names)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    dataset.to_csv(tmp, index=False)
    tmp.replace(out)
    heavy = int((dataset["Rainfall_mm"] >= 64.5).sum())
    say(f"\nSaved {out.name}: {len(dataset):,} rows, {dataset['Location'].nunique()} segments, "
        f"{dataset['Date'].min()} to {dataset['Date'].max()}, {heavy} heavy-rain segment-days (>= 64.5 mm).")
    say("Next: python -m training.train_rainfall")
    return dataset


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START, help="first day (YYYY-MM-DD), default 2020-01-01")
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="last day (default: 6 days ago)")
    parser.add_argument("--hourly-budget", type=float, default=HOURLY_BUDGET)
    args = parser.parse_args(argv)
    end = args.end or (config.today_ist() - timedelta(days=ARCHIVE_DELAY_DAYS))
    if args.start >= end - timedelta(days=3 * 365):
        parser.error("Use at least 3 years of data (training, validation and test years).")
    try:
        result = run(args.start, end, config.RAINFALL_CORRIDOR_DATASET, config.OPEN_METEO_CACHE_DIR,
                     Pacer(args.hourly_budget, DAILY_BUDGET))
    except RainfallDownloadError as error:
        print(f"\nDownload stopped: {error}\nSegments downloaded so far are saved; run the command again to continue.")
        return 1
    return 0 if result is not None else 2


if __name__ == "__main__":
    sys.exit(main())
