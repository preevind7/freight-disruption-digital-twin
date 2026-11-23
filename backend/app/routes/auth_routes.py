from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm

from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserProfile
from app.schemas.common import ErrorResponse
from app.services import auth_service

router = APIRouter(tags=["Authentication"])
AUTH_ERRORS = {401: {"model": ErrorResponse, "description": "Wrong credentials or bad token"}, 429: {"model": ErrorResponse, "description": "Too many failed attempts"}}


@router.post("/api/auth/register", response_model=UserProfile, status_code=201, responses={403: {"model": ErrorResponse, "description": "Sign-up disabled on this server"}, 422: {"model": ErrorResponse, "description": "Invalid details or username taken"}})
def register(body: RegisterRequest) -> UserProfile:
    """Create a new account. Afterwards, sign in with /api/auth/login."""
    return UserProfile(**auth_service.register(body.name, body.username, body.password, body.role))


@router.post("/api/auth/login", response_model=TokenResponse, responses=AUTH_ERRORS)
def login(body: LoginRequest) -> TokenResponse:
    """Sign in with JSON {username, password}. Returns a bearer token for the Authorization header."""
    return TokenResponse(**auth_service.authenticate(body.username, body.password))


@router.post("/api/auth/token", response_model=TokenResponse, responses=AUTH_ERRORS, include_in_schema=False)
def login_form(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """Form-based login used by the Swagger UI 'Authorize' button."""
    return TokenResponse(**auth_service.authenticate(form.username, form.password))


@router.get("/api/auth/me", response_model=UserProfile, responses=AUTH_ERRORS)
def me(user: dict = Depends(auth_service.get_current_user)) -> UserProfile:
    """Who am I? Also a quick way for the frontend to check that a saved token is still valid."""
    return UserProfile(**user)
