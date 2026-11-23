from fastapi import APIRouter

from app.routes import ERROR_RESPONSES
from app.schemas.accident import AccidentRequest, AccidentResponse
from app.services import accident_service

router = APIRouter(tags=["Model 2 - Accident risk"])


@router.post("/api/predict/accident", response_model=AccidentResponse, responses=ERROR_RESPONSES)
def predict_accident(request: AccidentRequest) -> AccidentResponse:
    """Score the accident risk of the given road, weather, traffic and time conditions (0-1)."""
    return accident_service.predict(request)
