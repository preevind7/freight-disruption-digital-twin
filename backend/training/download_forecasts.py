"""Download the rain FORECASTS that were made one day earlier, for every corridor segment.

Run from the backend folder after training/download_rainfall.py (needs internet, no API key):
    python -m training.download_forecasts

Source: Open-Meteo Previous Runs API (https://open-meteo.com/en/docs/previous-runs-api).
`precipitation_previous_day1` is, for each hour, the value that was predicted 24 hours before
that hour. Summed over an Indian day (IST) it is "the rain forecast for day D made on day D-1",
so the model only gets information that was available before the day started (no look-ahead).
Most models are archived from January 2024. Weather data by Open-Meteo.com (CC BY 4.0).

Output: datasets/corridor_rain_forecast_open_meteo.csv (Date, Location, Segment_ID, Forecast_mm).
A day without all 24 hourly values is left empty (the model then uses past rain only).
Same pacing / resume behaviour as download_rainfall.py; running again only fetches new days.
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app import config
from app.services import weather_source
from training import download_rainfall as dl

VARIABLE = weather_source.FORECAST_VARIABLE
SOURCE = "Open-Meteo Previous Runs API (forecast made 24 h ahead), CC BY 4.0"
DEFAULT_START = date(2024, 1, 1)
CHUNK_DAYS = 120


def fetch_previous_runs(lat: float, lon: float, start: date, end: date, timeout: float = 60.0) -> Dict[str, Optional[float]]:
    """Hourly forecasts (UTC hours) covering the IST days start..end. Returns {"YYYY-MM-DDTHH:MM": mm or None}."""
    values = weather_source.fetch_previous_runs(lat, lon, start, end, timeout=timeout)
    if not values:
        raise dl.RainfallDownloadError(f"Open-Meteo returned no {VARIABLE} for this location.")
    return values


daily_ist_totals = weather_source.daily_ist_totals


def fetch_days(fetch, pacer: dl.Pacer, lat: float, lon: float, start: date, end: date) -> pd.Series:
    parts = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(end, chunk_start + timedelta(days=CHUNK_DAYS - 1))
        hourly = dl.fetch_with_retry(fetch, pacer, lat, lon, chunk_start, chunk_end)
        parts.append(daily_ist_totals(hourly, chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return pd.concat(parts)


def download_segment(segment: dict, start: date, end: date, cache_dir: Path, pacer: dl.Pacer, fetch=fetch_previous_runs):
    path = cache_dir / f"{segment['id']}.csv"
    lat, lon = segment["mid"]["lat"], segment["mid"]["lon"]
    cached = None
    if path.exists():
        df = pd.read_csv(path, parse_dates=["Date"])
        cached = pd.Series(df["Forecast_mm"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["Date"]))
    if cached is not None and len(cached) and cached.index.min().date() <= start:
        cached = cached.loc[pd.Timestamp(start):]
        have_until = cached.index.max().date()
        if have_until >= end:
            return cached.loc[:pd.Timestamp(end)], "already downloaded"
        series = pd.concat([cached, fetch_days(fetch, pacer, lat, lon, have_until + timedelta(days=1), end)])
        what = "added new days"
    else:
        series = fetch_days(fetch, pacer, lat, lon, start, end)
        what = f"downloaded {series.notna().sum()} forecast days"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame({"Date": series.index.strftime("%Y-%m-%d"), "Forecast_mm": series.to_numpy()}).to_csv(tmp, index=False)
    tmp.replace(path)
    return series, what


def run(start: date, end: date, out: Path, cache_dir: Path, pacer: dl.Pacer, fetch=fetch_previous_runs, say=print) -> Optional[pd.DataFrame]:
    segments, _ = dl.load_segments()
    say(f"Rain forecasts made 24 h ahead for {len(segments)} segments, {start} to {end} (Open-Meteo Previous Runs API, free, no key).")
    say(f"About {math.ceil(len(segments) * dl.call_cost(start, end)):,} Open-Meteo calls if nothing is saved yet.\n")
    frames = []
    try:
        for n, seg in enumerate(segments, 1):
            series, what = download_segment(seg, start, end, cache_dir, pacer, fetch)
            frames.append(pd.DataFrame({"Date": series.index.strftime("%Y-%m-%d"), "Location": f"{seg['id']} {seg['name']}",
                                        "Segment_ID": seg["id"], "Forecast_mm": series.to_numpy(), "Source": SOURCE}))
            say(f"[{n:>2}/{len(segments)}] {seg['id']} {seg['name']}: {what} "
                f"({series.notna().mean():.0%} of days complete, largest forecast {np.nanmax(series.to_numpy()) if series.notna().any() else 0:.1f} mm)")
    except dl.DailyLimitReached:
        say(f"\nStopped: Open-Meteo's free daily limit is used up. {len(frames)} of {len(segments)} segments are saved; run again tomorrow.")
        return None
    except KeyboardInterrupt:
        say(f"\nStopped by you. {len(frames)} of {len(segments)} segments are saved; run again to continue.")
        return None
    dataset = pd.concat(frames, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    dataset.to_csv(tmp, index=False)
    tmp.replace(out)
    say(f"\nSaved {out.name}: {len(dataset):,} rows, {dataset['Forecast_mm'].notna().mean():.0%} of segment-days have a complete forecast, "
        f"{int((dataset['Forecast_mm'] >= 64.5).sum())} forecast heavy-rain segment-days.")
    say("Next: python -m training.train_rainfall")
    return dataset


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="last day (default: 2 days ago)")
    parser.add_argument("--hourly-budget", type=float, default=dl.HOURLY_BUDGET)
    args = parser.parse_args(argv)
    end = args.end or (config.today_ist() - timedelta(days=2))
    try:
        result = run(args.start, end, config.RAIN_FORECAST_DATASET, config.OPEN_METEO_FORECAST_CACHE_DIR, dl.Pacer(args.hourly_budget, dl.DAILY_BUDGET))
    except dl.RainfallDownloadError as error:
        print(f"\nDownload stopped: {error}\nSegments downloaded so far are saved; run the command again to continue.")
        return 1
    return 0 if result is not None else 2


if __name__ == "__main__":
    sys.exit(main())
