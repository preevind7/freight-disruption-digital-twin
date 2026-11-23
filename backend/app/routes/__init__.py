from app.schemas.common import ErrorResponse

ERROR_RESPONSES = {
    401: {"model": ErrorResponse, "description": "Login required or session expired"},
    404: {"model": ErrorResponse, "description": "Requested data not found"},
    422: {"model": ErrorResponse, "description": "Invalid input"},
    503: {"model": ErrorResponse, "description": "Model not trained or failed to load"},
}
