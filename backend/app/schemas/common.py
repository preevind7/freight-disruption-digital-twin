from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    name: str = Field(..., description="Which of the three models produced this result")
    algorithm: str
    trained_at: Optional[str] = None


class FieldError(BaseModel):
    field: Optional[str] = None
    message: str


class ErrorResponse(BaseModel):
    detail: str
    errors: List[FieldError] = []
