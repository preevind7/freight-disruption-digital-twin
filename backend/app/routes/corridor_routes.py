import datetime as dt
from typing import Optional

from fastapi import APIRouter, Query

from app import config
from app.routes import ERROR_RESPONSES
from app.services import corridor_service
from app.utils.validation import InvalidInputError

router = APIRouter(tags=["Corridor forecast (all 3 models)"])
EARLIEST = dt.date(2016, 1, 8)


@router.get("/api/corridor/forecast", responses=ERROR_RESPONSES)
def corridor_forecast(
    corridor_id: str = Query("mumbai-bengaluru", examples=["mumbai-bengaluru", "chennai-bengaluru"]),
    date: Optional[dt.date] = Query(None, description="Day to forecast (YYYY-MM-DD). Default: today (IST)."),
    refresh: bool = Query(False, description="Ignore the saved forecast and recompute (re-fetches live weather)."),
):
    """Hourly disruption risk for every segment of a corridor, built from the rainfall, traffic and accident models."""
    day = date or config.today_ist()
    latest = config.today_ist() + dt.timedelta(days=14)
    if not EARLIEST <= day <= latest:
        raise InvalidInputError(f"date must be between {EARLIEST} and {latest}.", "date")
    return corridor_service.forecast(corridor_id, day, use_cache=not refresh)
