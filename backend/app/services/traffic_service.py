"""Business logic for MODEL 3 (festival-time traffic congestion level)."""
from __future__ import annotations

from app.config import SIGNIFICANT_DATES_DATASET
from app.ml_models import traffic_model as tm
from app.ml_models.registry import registry
from app.schemas.traffic import TrafficRequest, TrafficResponse
from app.services.common import model_info
from app.utils.validation import check_choice

NAME = "traffic"
NOTE = (
    "No dataset measures congestion directly: the model learned the traffic density recorded at "
    "20,000 accident sites. Festival dates come from significant_dates.csv."
)


def predict(request: TrafficRequest) -> TrafficResponse:
    bundle = registry.get(NAME)
    options = bundle["input_options"]
    city = check_choice(request.city, options["city"], "city")
    road_type = check_choice(request.road_type, options["road_type"], "road_type")
    weather = check_choice(request.weather, options["weather"], "weather")
    festival_input = check_choice(request.festival, options["festival"], "festival") if request.festival else None

    festival_event, calendar_text = tm.resolve_event(request.date, festival_input, SIGNIFICANT_DATES_DATASET)
    features = tm.build_request_frame(city, request.date, request.hour, road_type, request.lanes, request.traffic_signal, weather, festival_event)

    pipeline = bundle["pipeline"]
    proba = pipeline.predict_proba(features)[0]
    probabilities = {str(c): round(float(p), 4) for c, p in zip(pipeline.classes_, proba)}
    level = max(probabilities, key=probabilities.get)
    return TrafficResponse(
        congestion_level=level,
        probabilities={c: probabilities.get(c, 0.0) for c in tm.CLASSES},
        confidence=probabilities[level],
        festival_event=festival_event,
        calendar_event=calendar_text,
        is_peak_hour=request.hour in tm.PEAK_HOURS,
        note=NOTE,
        model=model_info(NAME, bundle),
    )
