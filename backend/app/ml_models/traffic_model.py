"""Feature definitions for MODEL 3 - festival-time traffic congestion level.

No uploaded file measures traffic congestion directly (see reports/DATASET_ANALYSIS.md).
The best available signal is the `traffic_density` column (low / medium / high) in
datasets/indian_roads_dataset.csv, combined with festival / event dates from
datasets/significant_dates.csv.

Limitation (also returned by the API): traffic_density is recorded at the time and place
of an accident, so it describes congestion where accidents happened, not all roads.

Inputs are only things known before travelling: city, date, hour, road type, lanes,
signal, weather and festival/event. Columns describing the accident itself, and
risk_score (which is computed from traffic_density), are never used.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from app.utils.events import NO_EVENT, event_on, load_significant_dates, normalise_event

PEAK_HOURS = {8, 9, 10, 17, 18, 19}
CLASSES = ["low", "medium", "high"]

CATEGORICAL_FEATURES = ["city", "road_type", "weather", "festival_event", "day_of_week"]
NUMERIC_FEATURES = ["lanes", "traffic_signal", "hour_sin", "hour_cos", "is_peak_hour", "is_weekend", "is_festival_or_event", "month_sin", "month_cos"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "traffic_density"


def _time_features(dates: pd.Series, hours: pd.Series) -> pd.DataFrame:
    hour_angle = 2 * np.pi * hours.astype(float) / 24.0
    month_angle = 2 * np.pi * (dates.dt.month.astype(float) - 1) / 12.0
    day_name = dates.dt.day_name()
    return pd.DataFrame(
        {
            "day_of_week": day_name,
            "is_weekend": day_name.isin(["Saturday", "Sunday"]).astype(int),
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "is_peak_hour": hours.isin(PEAK_HOURS).astype(int),
            "month_sin": np.sin(month_angle),
            "month_cos": np.cos(month_angle),
        },
        index=dates.index,
    )


def build_training_frame(raw: pd.DataFrame, significant_dates_path: Path) -> pd.DataFrame:
    # Re-derive time features from date / hour exactly as the API does.
    df = raw.drop(columns=["is_peak_hour", "is_weekend", "day_of_week"], errors="ignore").copy()
    df["date"] = pd.to_datetime(df["date"])
    events = load_significant_dates(significant_dates_path)
    from_calendar = df["date"].dt.strftime("%Y-%m-%d").map(events)
    # Festival column of the roads data first, then the significant-dates calendar.
    df["festival_event"] = [
        normalise_event(fest) if isinstance(fest, str) else normalise_event(cal)
        for fest, cal in zip(df["festival"], from_calendar)
    ]
    df["is_festival_or_event"] = (df["festival_event"] != NO_EVENT).astype(int)
    df = pd.concat([df, _time_features(df["date"], df["hour"])], axis=1)
    return df[FEATURES + [TARGET, "date"]]


def resolve_event(day: date, festival: Optional[str], significant_dates_path: Path) -> Tuple[str, Optional[str]]:
    """Returns (festival category used by the model, event text found in the calendar)."""
    calendar_text = event_on(day, significant_dates_path)
    if festival and festival != NO_EVENT:
        return normalise_event(festival), calendar_text  # user's choice wins
    return normalise_event(calendar_text), calendar_text


def build_request_frame(
    city: str,
    day: date,
    hour: int,
    road_type: str,
    lanes: int,
    traffic_signal: bool,
    weather: str,
    festival_event: str,
) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime([day]))
    row = pd.DataFrame(
        [
            {
                "city": city,
                "road_type": road_type,
                "weather": weather,
                "festival_event": festival_event,
                "lanes": int(lanes),
                "traffic_signal": int(bool(traffic_signal)),
                "is_festival_or_event": int(festival_event != NO_EVENT),
            }
        ]
    )
    row = pd.concat([row, _time_features(dates, pd.Series([hour]))], axis=1)
    return row[FEATURES]


def build_hourly_frame(city: str, day: date, road_type: str, lanes: int, traffic_signal: bool, weather: str, festival_event: str) -> pd.DataFrame:
    """Same features as build_request_frame, for all 24 hours of one day at once (fast)."""
    hours = pd.Series(range(24))
    row = {"city": city, "road_type": road_type, "weather": weather, "festival_event": festival_event,
           "lanes": int(lanes), "traffic_signal": int(bool(traffic_signal)), "is_festival_or_event": int(festival_event != NO_EVENT)}
    frame = pd.DataFrame([row] * 24)
    frame = pd.concat([frame, _time_features(pd.Series(pd.to_datetime([day] * 24)), hours)], axis=1)
    return frame[FEATURES]
