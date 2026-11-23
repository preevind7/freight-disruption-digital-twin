"""Feature definitions for MODEL 2 - pre-accident road risk score.

Dataset: datasets/indian_roads_dataset.csv (20,000 accident records, 8 cities, 2022-2025).

Why this target (see reports/DATASET_ANALYSIS.md):
  * Every row is an accident, so "will an accident happen?" cannot be learned.
  * accident_severity does not depend on any road/weather column (not learnable).
  * risk_score = road-condition risk + 0.20 when the accident turned out fatal.
    The fatal part is only known after the crash, so it is removed:
        pre_accident_risk = risk_score - 0.20 * (accident_severity == "fatal")
    This is the risk of the road conditions themselves, known before a trip.

Post-accident columns (cause, accident_severity, vehicles_involved, casualties) and the
raw risk_score are never used as inputs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

PEAK_HOURS = {8, 9, 10, 17, 18, 19}  # verified: matches is_peak_hour in the dataset
WEEKEND_DAYS = {"Saturday", "Sunday"}
NO_FESTIVAL = "None"
FATAL_ADJUSTMENT = 0.20

# Same bands the frontend already uses for risk colours.
RISK_BANDS = [(0.30, "Low"), (0.60, "Moderate"), (float("inf"), "High")]

CATEGORICAL_FEATURES = ["city", "road_type", "weather", "visibility", "traffic_density", "festival", "day_of_week"]
NUMERIC_FEATURES = ["lanes", "traffic_signal", "temperature", "hour_sin", "hour_cos", "is_peak_hour", "is_weekend"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "pre_accident_risk"


def risk_band(score: float) -> str:
    score = round(float(score), 3)  # avoid 0.29999 landing in the wrong band
    for limit, label in RISK_BANDS:
        if score < limit:
            return label
    return "High"


def _hour_features(hours: pd.Series) -> pd.DataFrame:
    angle = 2 * np.pi * hours.astype(float) / 24.0
    return pd.DataFrame(
        {
            "hour_sin": np.sin(angle),
            "hour_cos": np.cos(angle),
            "is_peak_hour": hours.isin(PEAK_HOURS).astype(int),
        },
        index=hours.index,
    )


def build_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    # Re-derive is_peak_hour / is_weekend from hour / day_of_week exactly as the API does.
    df = raw.drop(columns=["is_peak_hour", "is_weekend"], errors="ignore").copy()
    df["festival"] = df["festival"].fillna(NO_FESTIVAL)
    df["is_weekend"] = df["day_of_week"].isin(WEEKEND_DAYS).astype(int)
    df = pd.concat([df, _hour_features(df["hour"])], axis=1)
    fatal = (df["accident_severity"] == "fatal").astype(float)
    df[TARGET] = (df["risk_score"] - FATAL_ADJUSTMENT * fatal).round(2)
    df["date"] = pd.to_datetime(df["date"])
    return df[FEATURES + [TARGET, "date"]]


def build_request_frame(
    city: str,
    road_type: str,
    lanes: int,
    traffic_signal: bool,
    weather: str,
    visibility: str,
    temperature: float,
    traffic_density: str,
    hour: int,
    day_of_week: str,
    festival: Optional[str],
) -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "city": city,
                "road_type": road_type,
                "weather": weather,
                "visibility": visibility,
                "traffic_density": traffic_density,
                "festival": festival or NO_FESTIVAL,
                "day_of_week": day_of_week,
                "lanes": int(lanes),
                "traffic_signal": int(bool(traffic_signal)),
                "temperature": float(temperature),
                "is_weekend": int(day_of_week in WEEKEND_DAYS),
            }
        ]
    )
    row = pd.concat([row, _hour_features(pd.Series([hour]))], axis=1)
    return row[FEATURES]


def build_hourly_frame(city: str, road_type: str, lanes: int, traffic_signal: bool, weather: str, visibility: str,
                       temperature: float, traffic_density_by_hour, day_of_week: str, festival: Optional[str]) -> pd.DataFrame:
    """Same features as build_request_frame, for hours 0-23 at once; traffic density can differ per hour."""
    base = {"city": city, "road_type": road_type, "weather": weather, "visibility": visibility, "festival": festival or NO_FESTIVAL,
            "day_of_week": day_of_week, "lanes": int(lanes), "traffic_signal": int(bool(traffic_signal)),
            "temperature": float(temperature), "is_weekend": int(day_of_week in WEEKEND_DAYS)}
    frame = pd.DataFrame([base] * 24)
    frame["traffic_density"] = list(traffic_density_by_hour)
    frame = pd.concat([frame, _hour_features(pd.Series(range(24)))], axis=1)
    return frame[FEATURES]
