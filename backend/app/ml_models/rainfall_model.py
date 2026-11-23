"""Feature definitions for MODEL 1 - next-day rainfall category / heavy-rain (flood-trigger) risk.

Dataset: datasets/corridor_rainfall_open_meteo.csv (real daily rainfall at the 35 corridor
segments, Open-Meteo / ERA5, made by training/download_rainfall.py) or, if that file is not
there, the uploaded datasets/india_rainfall_2016_2025.csv (20 cities, daily, 2016-2025).

Target: the IMD daily rainfall category of the forecast day, predicted only from the
rainfall of the 7 days before it plus the location and the season.

Same-day columns (Precipitation_mm, Rain_Hours) are NOT used: they are measured on the
day being predicted, so using them would leak the answer.

The same functions are used for training and for serving, so the API always builds
exactly the features the model was trained on.
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LAG_DAYS = 7

# IMD daily rainfall categories (mm/day). Data has 0.1 mm resolution.
CATEGORY_LABELS = [
    "No / very light rain (< 2.5 mm)",
    "Light rain (2.5 - 15.5 mm)",
    "Moderate rain (15.6 - 64.4 mm)",
    "Heavy rain or above (>= 64.5 mm)",
]
CATEGORY_EDGES = [2.5, 15.6, 64.5]
HEAVY_CLASS = 3
RAINY_DAY_MM = 2.5  # IMD definition of a rainy day

CATEGORICAL_FEATURES = ["location"]
NUMERIC_FEATURES = (
    ["doy_sin", "doy_cos"]
    + [f"lag_{i}" for i in range(1, LAG_DAYS + 1)]
    + ["sum_3", "sum_7", "max_7", "rainy_days_7"]
)
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
# Optional input: rain forecast for the day, made the day before (Open-Meteo Previous Runs API,
# training/download_forecasts.py). Used only when it beats past-rain-only on validation.
FORECAST_FEATURE = "forecast_mm"
NUMERIC_FEATURES_FORECAST = NUMERIC_FEATURES + [FORECAST_FEATURE]
FEATURES_FORECAST = CATEGORICAL_FEATURES + NUMERIC_FEATURES_FORECAST


def rainfall_to_category(values) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), CATEGORY_EDGES)


def _season_features(dates: pd.Series) -> pd.DataFrame:
    doy = dates.dt.dayofyear.astype(float)
    angle = 2 * np.pi * doy / 365.25
    return pd.DataFrame({"doy_sin": np.sin(angle), "doy_cos": np.cos(angle)}, index=dates.index)


def _window_features(lags: pd.DataFrame) -> pd.DataFrame:
    """lags has columns lag_1 (yesterday) ... lag_7."""
    out = lags.copy()
    out["sum_3"] = lags[["lag_1", "lag_2", "lag_3"]].sum(axis=1)
    out["sum_7"] = lags.sum(axis=1)
    out["max_7"] = lags.max(axis=1)
    out["rainy_days_7"] = (lags >= RAINY_DAY_MM).sum(axis=1)
    return out


def build_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw daily table into one row per (city, day) with lag features + target."""
    df = raw[["Date", "Location", "Rainfall_mm"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Location", "Date"]).reset_index(drop=True)

    # Lags are only valid if days are consecutive - verify instead of assuming.
    gaps = df.groupby("Location")["Date"].diff().dropna()
    if not (gaps.dt.days == 1).all():
        raise ValueError("Rainfall dataset has missing or duplicate days; lag features would be wrong.")

    grouped = df.groupby("Location")["Rainfall_mm"]
    lags = pd.DataFrame({f"lag_{i}": grouped.shift(i) for i in range(1, LAG_DAYS + 1)})
    features = pd.concat(
        [df[["Location"]].rename(columns={"Location": "location"}), _season_features(df["Date"]), _window_features(lags)],
        axis=1,
    )
    features["date"] = df["Date"]
    features["actual_mm"] = df["Rainfall_mm"]
    features["target"] = rainfall_to_category(df["Rainfall_mm"])
    return features.dropna(subset=[f"lag_{i}" for i in range(1, LAG_DAYS + 1)]).reset_index(drop=True)


def build_request_frame(location: str, forecast_date: date, last_7_days_mm: Sequence[float], forecast_mm: Optional[float] = None) -> pd.DataFrame:
    """last_7_days_mm[0] is yesterday's rainfall, last_7_days_mm[6] is 7 days ago.
    forecast_mm (optional) is the rain forecast for forecast_date made the day before; NaN when unknown.
    Returns all columns of FEATURES_FORECAST; select the model's own feature list before predicting."""
    if len(last_7_days_mm) != LAG_DAYS:
        raise ValueError(f"Exactly {LAG_DAYS} daily rainfall values are required.")
    lags = pd.DataFrame([{f"lag_{i + 1}": float(v) for i, v in enumerate(last_7_days_mm)}])
    dates = pd.Series(pd.to_datetime([forecast_date]))
    frame = pd.concat([pd.DataFrame({"location": [location]}), _season_features(dates), _window_features(lags)], axis=1)
    frame[FORECAST_FEATURE] = np.nan if forecast_mm is None else float(forecast_mm)
    return frame[FEATURES_FORECAST]


HEAVY_MM = CATEGORY_EDGES[-1]
# Heavy-rain alert rules compared on the validation year by training/train_rainfall.py:
#   model                - heavy-rain probability >= threshold
#   persistence          - yesterday had heavy rain (>= 64.5 mm)
#   model_or_persistence - either of the two (threshold re-tuned for the combination)
#   forecast             - the rain forecast made the day before is >= 64.5 mm (only when forecasts exist)
ALERT_RULES = ("model", "persistence", "model_or_persistence", "forecast")
ALERT_RULE_TEXT = {
    "forecast": "a rain forecast of 64.5 mm or more, made the day before",
    "model": "heavy-rain probability at or above the alert threshold",
    "persistence": "heavy rain (>= 64.5 mm) on the previous day",
    "model_or_persistence": "heavy-rain probability at or above the alert threshold, or heavy rain on the previous day",
}


def heavy_alert(rule: str, p_heavy: float, threshold: float, rain_yesterday_mm: float, forecast_mm: Optional[float] = None) -> Tuple[bool, str]:
    """Apply the alert rule chosen on validation. Returns (alert on?, reason shown to the user)."""
    by_model = rule in ("model", "model_or_persistence") and p_heavy >= threshold
    by_persistence = rule in ("persistence", "model_or_persistence") and rain_yesterday_mm >= HEAVY_MM
    by_forecast = rule == "forecast" and forecast_mm is not None and not np.isnan(forecast_mm) and forecast_mm >= HEAVY_MM
    reasons = []
    if by_forecast:
        reasons.append(f"{forecast_mm:.0f} mm forecast for the day (made the day before)")
    if by_model:
        reasons.append(f"{p_heavy:.0%} chance of >= 64.5 mm")
    if by_persistence:
        reasons.append(f"{rain_yesterday_mm:.0f} mm fell the day before (heavy rain often continues)")
    return bool(by_model or by_persistence or by_forecast), "; ".join(reasons)


def known_locations(raw: pd.DataFrame) -> List[str]:
    return sorted(raw["Location"].unique().tolist())
