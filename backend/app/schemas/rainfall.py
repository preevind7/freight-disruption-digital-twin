from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.common import ModelInfo


class RainfallRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "city": "Mumbai",
                "forecast_date": "2025-07-15",
                "rainfall_last_7_days_mm": [42.0, 18.5, 60.2, 12.0, 5.0, 0.0, 22.4],
            }
        }
    )

    city: str = Field(..., min_length=2, description="A location the rainfall model was trained on: a corridor segment such as 'S03 Khalapur – Lonavala (Bhor Ghat)' (real Open-Meteo data) or, with the uploaded city file, one of its 20 cities. GET /api/models/rainfall lists them.")
    forecast_date: date = Field(..., description="The day to forecast (YYYY-MM-DD)")
    rainfall_last_7_days_mm: List[float] = Field(
        ...,
        min_length=7,
        max_length=7,
        description="Observed daily rainfall in mm for the 7 days before forecast_date. Index 0 = the day before, index 6 = 7 days before.",
    )
    forecast_rain_mm: Optional[float] = Field(
        None, ge=0, le=1000,
        description="Optional: rain forecast for forecast_date made the day before (mm). Used when the model was trained with forecasts; "
                    "otherwise, or when left out, the past-rain-only model is used.",
    )

    @field_validator("rainfall_last_7_days_mm")
    @classmethod
    def non_negative(cls, values: List[float]) -> List[float]:
        for value in values:
            if value < 0 or value > 1000:
                raise ValueError("each daily rainfall value must be between 0 and 1000 mm")
        return values


class CategoryProbability(BaseModel):
    category: str
    probability: float


class RainfallResponse(BaseModel):
    city: str
    forecast_date: date
    predicted_category: str
    category_probabilities: List[CategoryProbability]
    heavy_rain_probability: float = Field(..., description="P(rainfall >= 64.5 mm) - used as the flood-trigger risk")
    usual_heavy_rain_probability: float = Field(..., description="Historical share of heavy-rain days for this city and month")
    heavy_rain_alert: bool
    alert_threshold: float
    alert_rule: str = Field("model", description="Alert rule chosen on the validation year: model, persistence or model_or_persistence")
    alert_rule_text: str = ""
    alert_reason: str = Field("", description="Why the alert is on (empty when it is off)")
    forecast_rain_mm: Optional[float] = None
    used_forecast: bool = False
    model_used: str = ""
    flood_trigger_risk_level: str = Field(..., description="Low / Moderate / High")
    explanation: str
    note: str
    model: ModelInfo


class RainfallHistoryResponse(BaseModel):
    city: str
    forecast_date: date
    rainfall_last_7_days_mm: List[float]
    actual_rainfall_mm: Optional[float] = None
    forecast_rain_mm: Optional[float] = Field(None, description="Rain forecast for forecast_date made the day before, if the forecast file has it")
    actual_category: Optional[str] = None
    source: str
