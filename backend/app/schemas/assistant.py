from __future__ import annotations

import datetime as dt
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class TripInfo(BaseModel):
    id: str = Field(..., max_length=40)
    vehicle: Optional[str] = Field(None, max_length=60)
    origin: Optional[str] = Field(None, max_length=60)
    destination: Optional[str] = Field(None, max_length=60)
    current_segment: Optional[str] = Field(None, max_length=10)
    status: Optional[str] = Field(None, max_length=30)


class AssistantRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "question": "Why is S01 risky right now and what should I do?", "corridor_id": "mumbai-bengaluru",
        "date": "2025-06-19", "hour": 15, "segment_id": "S01", "history": [], "trips": []}})

    question: str = Field(..., min_length=1, max_length=1000)
    corridor_id: str = Field("mumbai-bengaluru", max_length=40)
    date: Optional[dt.date] = Field(None, description="Forecast date (default today, IST)")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of day for 'now' (default current IST hour)")
    segment_id: Optional[str] = Field(None, max_length=10, description="Segment the user is asking about, if any")
    history: List[ChatTurn] = Field(default_factory=list, max_length=12)
    trips: List[TripInfo] = Field(default_factory=list, max_length=60)


class Grounding(BaseModel):
    numbers_checked: int
    unsupported: List[str]
    all_supported: bool


class AssistantResponse(BaseModel):
    answer: str
    provider: str
    model: str
    fallback_note: Optional[str] = None
    grounding: Grounding
    focus_segment: str
    date: str
    time: str
    context_summary: dict


class AssistantStatus(BaseModel):
    enabled: bool
    provider: str
    provider_label: Optional[str] = None
    model: Optional[str] = None
    reason: Optional[str] = None
    fallback: Optional[dict] = None
