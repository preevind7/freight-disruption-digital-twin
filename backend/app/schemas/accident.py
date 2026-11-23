from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import ModelInfo


class AccidentRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "city": "Mumbai",
                "road_type": "highway",
                "lanes": 4,
                "traffic_signal": False,
                "weather": "rain",
                "visibility": "low",
                "temperature": 27,
                "traffic_density": "high",
                "hour": 18,
                "day_of_week": "Friday",
                "festival": "None",
            }
        }
    )

    city: str = Field(..., description="City from the accident dataset")
    road_type: str = Field(..., description="highway / rural / urban")
    lanes: int = Field(..., ge=1, le=6)
    traffic_signal: bool
    weather: str = Field(..., description="clear / fog / rain")
    visibility: str = Field(..., description="high / medium / low")
    temperature: float = Field(..., ge=-10, le=55, description="degrees Celsius")
    traffic_density: str = Field(..., description="low / medium / high")
    hour: int = Field(..., ge=0, le=23, description="Hour of travel, 0-23")
    day_of_week: str = Field(..., description="Monday ... Sunday")
    festival: Optional[str] = Field("None", description="None / Diwali / Eid / Holi / New Year")


class RiskDriver(BaseModel):
    factor: str
    value: str
    effect: float = Field(..., description="Change in risk score versus average conditions (+ raises risk)")


class AccidentResponse(BaseModel):
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: str = Field(..., description="Low (< 0.30) / Moderate (0.30-0.59) / High (>= 0.60)")
    uncertainty: float = Field(..., description="90% of validation errors were within +/- this value")
    is_peak_hour: bool
    drivers: List[RiskDriver]
    note: str
    model: ModelInfo
