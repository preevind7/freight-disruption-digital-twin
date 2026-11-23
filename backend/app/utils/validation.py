"""Input checks that depend on what the trained model has seen."""
from __future__ import annotations

from typing import Iterable, Optional


class InvalidInputError(ValueError):
    """Raised by services for values the model cannot score. Returned as HTTP 422."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.field = field


def check_choice(value: str, allowed: Iterable[str], field: str) -> str:
    """Case-insensitive match against the values present in the training data."""
    allowed = list(allowed)
    lookup = {str(item).lower(): item for item in allowed}
    match = lookup.get(str(value).strip().lower())
    if match is None:
        raise InvalidInputError(f"'{value}' is not a valid {field}. Allowed values: {', '.join(map(str, allowed))}.", field)
    return match


def check_range(value: float, low: float, high: float, field: str) -> float:
    if value < low or value > high:
        raise InvalidInputError(f"{field} must be between {low} and {high} (range seen in the training data); got {value}.", field)
    return value


class DataNotFoundError(LookupError):
    """Requested record is not in the dataset. Returned as HTTP 404."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class AuthError(Exception):
    """Login / token problems. Returned as HTTP 401 (or 429 when locked out)."""

    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
