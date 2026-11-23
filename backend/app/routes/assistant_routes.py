import datetime as dt

from fastapi import APIRouter, Depends

from app import config
from app.routes import ERROR_RESPONSES
from app.schemas.assistant import AssistantRequest, AssistantResponse, AssistantStatus
from app.schemas.common import ErrorResponse
from app.services import assistant_service
from app.services.auth_service import get_current_user
from app.utils.validation import InvalidInputError

router = APIRouter(tags=["GenAI assistant"])
RESPONSES = {**ERROR_RESPONSES, 429: {"model": ErrorResponse, "description": "Too many questions or provider quota reached"},
             502: {"model": ErrorResponse, "description": "AI provider error"}}


@router.get("/api/assistant/status", response_model=AssistantStatus)
def assistant_status() -> AssistantStatus:
    """Is a real AI (LLM) provider configured on the backend?"""
    return AssistantStatus(**assistant_service.status())


@router.post("/api/assistant/chat", response_model=AssistantResponse, responses=RESPONSES)
def assistant_chat(body: AssistantRequest, user: dict = Depends(get_current_user)) -> AssistantResponse:
    """Ask the AI assistant. It answers from the live ML corridor forecast, and every number is checked."""
    day = body.date or config.today_ist()
    latest = config.today_ist() + dt.timedelta(days=14)
    if not dt.date(2016, 1, 8) <= day <= latest:
        raise InvalidInputError(f"date must be between 2016-01-08 and {latest}.", "date")
    result = assistant_service.ask(user, body.question.strip(), [t.model_dump() for t in body.history], body.corridor_id,
                                   day, body.hour, body.segment_id, [t.model_dump() for t in body.trips])
    return AssistantResponse(**result)
