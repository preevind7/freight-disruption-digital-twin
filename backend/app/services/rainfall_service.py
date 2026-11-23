"""Business logic for MODEL 1 (rainfall / heavy-rain flood-trigger risk)."""
from __future__ import annotations

import calendar
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from app import config
from app.ml_models import rainfall_model as rm
from app.ml_models.registry import ModelNotAvailableError, registry
from app.schemas.rainfall import CategoryProbability, RainfallHistoryResponse, RainfallRequest, RainfallResponse
from app.services.common import model_info
from app.utils.validation import DataNotFoundError, check_choice

NAME = "rainfall"
NOTE_CITY = (
    "Heavy rain (>= 64.5 mm/day, IMD) is used as a flood trigger; the dataset has no flood records. "
    "Forecast is for the whole city and day, not a road segment or hour."
)
NOTE_CORRIDOR = (
    "Heavy rain (>= 64.5 mm/day, IMD) is used as a flood trigger; there are no flood records. "
    "Trained on real daily rainfall at this corridor segment (Open-Meteo / ERA5 reanalysis, about 10-25 km grid). "
    "Forecast is for the whole day, not a single hour."
)


def dataset_path() -> Path:
    """The rainfall file the loaded model was trained on (falls back to the configured file)."""
    try:
        name = registry.get(NAME).get("dataset")
    except ModelNotAvailableError:
        name = None
    return config.DATASETS_DIR / name if name else config.RAINFALL_DATASET


def is_corridor_dataset(path: Path | None = None) -> bool:
    return (path or dataset_path()).name == config.RAINFALL_CORRIDOR_DATASET.name


def forecast_dataset_name() -> str | None:
    """Forecast file the loaded model was trained with (None when it does not use forecasts)."""
    try:
        bundle = registry.get(NAME)
    except ModelNotAvailableError:
        return None
    return bundle.get("forecast_dataset") if bundle.get("uses_forecast") else None


def uses_forecast(bundle: dict) -> bool:
    return bool(bundle.get("uses_forecast"))


def profile(bundle: dict, have_forecast: bool) -> dict:
    """The pipeline + alert rule to use: the main model, or its past-rain-only fallback when the
    main model needs a forecast that does not exist for this date."""
    if uses_forecast(bundle) and not have_forecast and "fallback" in bundle:
        fb = bundle["fallback"]
        return {"pipeline": fb["pipeline"], "features": fb["features"], "model_name": fb["model_name"],
                "threshold": float(fb["heavy_alert_threshold"]), "rule": fb.get("heavy_alert_rule", "model"), "with_forecast": False}
    return {"pipeline": bundle["pipeline"], "features": bundle.get("features", rm.FEATURES), "model_name": bundle["model_name"],
            "threshold": float(bundle["heavy_alert_threshold"]), "rule": alert_rule(bundle), "with_forecast": uses_forecast(bundle)}


def class_probabilities(pipeline, frame: pd.DataFrame, n_classes: int) -> np.ndarray:
    raw = pipeline.predict_proba(frame)
    out = np.zeros((len(frame), n_classes))
    for i, c in enumerate(pipeline.classes_):
        out[:, int(c)] = raw[:, i]
    return out


def alert_rule(bundle: dict) -> str:
    """Alert rule stored by training; bundles trained before rules existed used the model threshold."""
    return bundle.get("heavy_alert_rule", "model")


def _risk_level(p_heavy: float, usual: float, threshold: float, alert: bool) -> str:
    if alert:
        return "High"
    if p_heavy >= max(2 * usual, threshold / 2):
        return "Moderate"
    return "Low"


def predict(request: RainfallRequest) -> RainfallResponse:
    bundle = registry.get(NAME)
    city = check_choice(request.city, bundle["input_options"]["location"], "city")
    forecast_mm = request.forecast_rain_mm
    prof = profile(bundle, forecast_mm is not None)
    features = rm.build_request_frame(city, request.forecast_date, request.rainfall_last_7_days_mm, forecast_mm)
    labels = bundle["classes"]
    proba = class_probabilities(prof["pipeline"], features[prof["features"]], len(labels))[0].tolist()

    heavy = bundle["heavy_class"]
    threshold, rule = prof["threshold"], prof["rule"]
    p_heavy = proba[heavy]
    usual = float(bundle["climatology_heavy"].get(f"{city}|{request.forecast_date.month}", 0.0))
    alert, reason = rm.heavy_alert(rule, p_heavy, threshold, float(request.rainfall_last_7_days_mm[0]), forecast_mm)
    level = _risk_level(p_heavy, usual, threshold, alert)
    month = calendar.month_name[request.forecast_date.month]
    explanation = (
        f"Most likely: {labels[int(np.argmax(proba))]}. "
        f"Chance of heavy rain is {p_heavy:.1%}, compared with {usual:.1%} on a typical {month} day in {city}. "
        f"Heavy-rain alert rule (chosen on validation data): {rm.ALERT_RULE_TEXT[rule]}"
        + (f" ({threshold:.1%})." if rule in ("model", "model_or_persistence") else ".")
        + (f" Alert ON: {reason}." if alert else " Alert off.")
        + (f" Uses the rain forecast made the day before ({forecast_mm:.1f} mm)." if prof["with_forecast"]
           else (" No forecast given, so the past-rain-only model was used." if uses_forecast(bundle) else ""))
    )
    return RainfallResponse(
        city=city,
        forecast_date=request.forecast_date,
        predicted_category=labels[int(np.argmax(proba))],
        category_probabilities=[CategoryProbability(category=l, probability=round(p, 4)) for l, p in zip(labels, proba)],
        heavy_rain_probability=round(p_heavy, 4),
        usual_heavy_rain_probability=round(usual, 4),
        heavy_rain_alert=alert,
        alert_threshold=round(threshold, 4),
        alert_rule=rule,
        alert_rule_text=rm.ALERT_RULE_TEXT[rule],
        alert_reason=reason,
        forecast_rain_mm=forecast_mm,
        used_forecast=prof["with_forecast"],
        model_used=prof["model_name"],
        flood_trigger_risk_level=level,
        explanation=explanation,
        note=NOTE_CORRIDOR if is_corridor_dataset() else NOTE_CITY,
        model=model_info(NAME, bundle),
    )


@lru_cache(maxsize=2)
def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Date", "Location", "Rainfall_mm"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df.set_index(["Location", "Date"]).sort_index()


def _dataset() -> pd.DataFrame:
    return _load(str(dataset_path()))


def history(city: str, forecast_date: date) -> RainfallHistoryResponse:
    """Fill the 7-day input from the observed dataset (handy for demos and for checking a forecast)."""
    path = dataset_path()
    if not path.exists():
        raise DataNotFoundError(f"Dataset {path.name} is not in backend/datasets.")
    df = _dataset()
    city = check_choice(city, sorted(df.index.get_level_values(0).unique()), "location")
    series = df.loc[city]["Rainfall_mm"]
    days = [forecast_date - timedelta(days=i) for i in range(1, rm.LAG_DAYS + 1)]
    missing = [d for d in days if d not in series.index]
    if missing:
        first, last = series.index.min(), series.index.max()
        raise DataNotFoundError(
            f"The dataset has no rainfall for {city} on {missing[0]}. Choose a forecast date between "
            f"{first + timedelta(days=rm.LAG_DAYS)} and {last + timedelta(days=1)}."
        )
    actual = float(series[forecast_date]) if forecast_date in series.index else None
    return RainfallHistoryResponse(
        city=city,
        forecast_date=forecast_date,
        rainfall_last_7_days_mm=[float(series[d]) for d in days],
        actual_rainfall_mm=actual,
        actual_category=rm.CATEGORY_LABELS[int(rm.rainfall_to_category([actual])[0])] if actual is not None else None,
        forecast_rain_mm=_forecast_from_file(city, forecast_date),
        source=path.name,
    )


def _forecast_from_file(city: str, day: date):
    from app.services import weather_source   # imported here: weather_source imports this module
    return weather_source._forecast_dataset_value(city, day)
