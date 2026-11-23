from fastapi import APIRouter

from app.routes import ERROR_RESPONSES
from app.schemas.traffic import TrafficRequest, TrafficResponse
from app.services import traffic_live, traffic_service

router = APIRouter(tags=["Model 3 - Festival traffic congestion"])


@router.post("/api/predict/traffic", response_model=TrafficResponse, responses=ERROR_RESPONSES)
def predict_traffic(request: TrafficRequest) -> TrafficResponse:
    """Predict the congestion level (low / medium / high), detecting festivals from the event calendar."""
    return traffic_service.predict(request)


@router.get("/api/traffic/live/status")
def live_traffic_status() -> dict:
    """Is TomTom live traffic on? Requests used today, readings saved for future training. Never returns the key."""
    return traffic_live.status()
