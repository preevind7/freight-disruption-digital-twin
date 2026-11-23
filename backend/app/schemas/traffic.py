from __future__ import annotations

import datetime as dt
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import ModelInfo


class TrafficRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "city": "Bangalore",
                "date": "2026-11-08",
                "hour": 18,
                "road_type": "urban",
                "lanes": 3,
                "traffic_signal": True,
                "weather": "clear",
                "festival": None,
            }
        }
    )

    city: str
    date: dt.date
    hour: int = Field(..., ge=0, le=23)
    road_type: str = Field(..., description="highway / rural / urban")
    lanes: int = Field(..., ge=1, le=6)
    traffic_signal: bool
    weather: str = Field(..., description="clear / fog / rain")
    festival: Optional[str] = Field(
        None,
        description="Leave empty to detect from significant_dates.csv, or set None / Diwali / Eid / Holi / New Year / Other event",
    )


class TrafficResponse(BaseModel):
    congestion_level: str = Field(..., description="low / medium / high")
    probabilities: Dict[str, float]
    confidence: float = Field(..., description="Probability of the predicted level")
    festival_event: str = Field(..., description="Festival category used by the model")
    calendar_event: Optional[str] = Field(None, description="Event listed in significant_dates.csv for this date, if any")
    is_peak_hour: bool
    note: str
    model: ModelInfo
