"""FastAPI entry point.

Start from the backend folder:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app import config
from app.ml_models.registry import ModelNotAvailableError
from app.routes import accident_routes, assistant_routes, auth_routes, corridor_routes, rainfall_routes, system_routes, traffic_routes
from app.services.auth_service import get_current_user
from app.services.llm_client import LLMError
from app.utils.frontend import frontend_available, frontend_files
from app.utils.validation import AuthError, DataNotFoundError, InvalidInputError

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    system_routes.load_all_models()  # load once at startup, not per request
    yield


app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="Three independent ML models: rainfall/flood-trigger risk, accident risk, festival traffic congestion.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------- consistent JSON errors: {"detail": "...", "errors": [{"field", "message"}]} ----------
@app.exception_handler(RequestValidationError)
async def on_validation_error(_: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        location = [str(part) for part in err.get("loc", []) if part not in ("body", "query")]
        message = err.get("msg", "Invalid value").replace("Value error, ", "")
        errors.append({"field": ".".join(location) or None, "message": message})
    summary = "; ".join(f"{e['field']}: {e['message']}" if e["field"] else e["message"] for e in errors)
    return JSONResponse(status_code=422, content={"detail": f"Invalid input - {summary}", "errors": errors})


@app.exception_handler(InvalidInputError)
async def on_invalid_input(_: Request, exc: InvalidInputError):
    return JSONResponse(status_code=422, content={"detail": exc.message, "errors": [{"field": exc.field, "message": exc.message}]})


@app.exception_handler(AuthError)
async def on_auth_error(_: Request, exc: AuthError):
    headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message, "errors": []}, headers=headers)


@app.exception_handler(LLMError)
async def on_llm_error(_: Request, exc: LLMError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message, "errors": []})


@app.exception_handler(DataNotFoundError)
async def on_not_found(_: Request, exc: DataNotFoundError):
    return JSONResponse(status_code=404, content={"detail": exc.message, "errors": []})


@app.exception_handler(ModelNotAvailableError)
async def on_model_missing(_: Request, exc: ModelNotAvailableError):
    return JSONResponse(status_code=503, content={"detail": str(exc), "errors": []})


@app.exception_handler(Exception)
async def on_unexpected(_: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Check the backend terminal for details.", "errors": []})


# Public: /api/health, /api/auth/login, /api/auth/token. Everything else needs a valid login token.
LOGIN_REQUIRED = [Depends(get_current_user)]
app.include_router(auth_routes.router)
app.include_router(system_routes.public_router)
app.include_router(system_routes.router, dependencies=LOGIN_REQUIRED)
app.include_router(rainfall_routes.router, dependencies=LOGIN_REQUIRED)
app.include_router(accident_routes.router, dependencies=LOGIN_REQUIRED)
app.include_router(traffic_routes.router, dependencies=LOGIN_REQUIRED)
app.include_router(corridor_routes.router, dependencies=LOGIN_REQUIRED)
app.include_router(assistant_routes.router, dependencies=LOGIN_REQUIRED)


if config.SERVE_FRONTEND and frontend_available(config.FRONTEND_DIR):
    # The app itself at "/" (API routes above take precedence); hidden files such as .git are never served.
    app.mount("/", frontend_files(config.FRONTEND_DIR), name="frontend")
    logging.getLogger(__name__).info("Serving the frontend from %s at /", config.FRONTEND_DIR)
else:
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")
