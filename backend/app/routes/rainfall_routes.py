from datetime import date

from fastapi import APIRouter, Query

from app.routes import ERROR_RESPONSES
from app.schemas.rainfall import RainfallHistoryResponse, RainfallRequest, RainfallResponse
from app.services import rainfall_service

router = APIRouter(tags=["Model 1 - Rainfall & flood risk"])


@router.post("/api/predict/rainfall", response_model=RainfallResponse, responses=ERROR_RESPONSES)
def predict_rainfall(request: RainfallRequest) -> RainfallResponse:
    """Forecast the next-day rainfall category and heavy-rain (flood-trigger) risk for a city."""
    return rainfall_service.predict(request)


@router.get("/api/rainfall/history", response_model=RainfallHistoryResponse, responses=ERROR_RESPONSES)
def rainfall_history(
    city: str = Query(..., examples=["Mumbai"]),
    forecast_date: date = Query(..., examples=["2025-07-15"]),
) -> RainfallHistoryResponse:
    """Return the 7 observed days before `forecast_date` from the dataset, plus the actual outcome if known."""
    return rainfall_service.history(city, forecast_date)
