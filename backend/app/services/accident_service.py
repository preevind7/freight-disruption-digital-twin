"""Business logic for MODEL 2 (pre-accident road risk score)."""
from __future__ import annotations

from typing import List

import numpy as np

from app.ml_models import accident_model as am
from app.ml_models.registry import registry
from app.schemas.accident import AccidentRequest, AccidentResponse, RiskDriver
from app.services.common import model_info
from app.utils.validation import InvalidInputError, check_choice

NAME = "accident"
NOTE = (
    "Scores how risky the road conditions are, learned from 20,000 accident records. "
    "It is not the probability that an accident will happen."
)


def _linear_drivers(bundle, features) -> List[RiskDriver]:
    """Exact per-input contributions for a linear model: coef x (value - training average)."""
    pipeline = bundle["pipeline"]
    estimator = pipeline.named_steps["model"]
    reference = bundle.get("linear_reference")
    if not hasattr(estimator, "coef_") or reference is None:
        return []
    transformed = pipeline.named_steps["preprocess"].transform(features)[0]
    effects = np.asarray(estimator.coef_) * (transformed - np.asarray(reference))
    drivers, position = [], 0
    for name, width in bundle["feature_widths"]:
        effect = float(effects[position:position + width].sum())
        position += width
        if abs(effect) >= 0.005:
            drivers.append(RiskDriver(factor=name, value=str(features.iloc[0][name]), effect=round(effect, 3)))
    return sorted(drivers, key=lambda d: -abs(d.effect))


def predict(request: AccidentRequest) -> AccidentResponse:
    bundle = registry.get(NAME)
    options = bundle["input_options"]
    city = check_choice(request.city, options["city"], "city")
    road_type = check_choice(request.road_type, options["road_type"], "road_type")
    weather = check_choice(request.weather, options["weather"], "weather")
    visibility = check_choice(request.visibility, options["visibility"], "visibility")
    density = check_choice(request.traffic_density, options["traffic_density"], "traffic_density")
    day = check_choice(request.day_of_week, options["day_of_week"], "day_of_week")
    festival = check_choice(request.festival or am.NO_FESTIVAL, options["festival"], "festival")

    pairs = options["weather_visibility_pairs"]
    if f"{weather}|{visibility}" not in pairs:
        allowed = sorted(p.split("|")[1] for p in pairs if p.startswith(weather + "|"))
        raise InvalidInputError(
            f"Visibility '{visibility}' never occurs with weather '{weather}' in the training data; use {' or '.join(allowed)}.",
            "visibility",
        )

    features = am.build_request_frame(
        city, road_type, request.lanes, request.traffic_signal, weather, visibility,
        request.temperature, density, request.hour, day, festival,
    )
    score = float(np.clip(bundle["pipeline"].predict(features)[0], 0.0, 1.0))
    score = round(score, 3)
    return AccidentResponse(
        risk_score=score,
        risk_level=am.risk_band(score),
        uncertainty=round(float(bundle.get("error_p90", 0.0)), 3),
        is_peak_hour=request.hour in am.PEAK_HOURS,
        drivers=_linear_drivers(bundle, features),
        note=NOTE,
        model=model_info(NAME, bundle),
    )
