"""Central configuration. Values can be overridden in backend/.env."""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent  # .../backend
load_dotenv(BASE_DIR / ".env")


def _path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    return Path(value).resolve() if value else default


DATASETS_DIR = _path("DATASETS_DIR", BASE_DIR / "datasets")
MODELS_DIR = _path("MODELS_DIR", BASE_DIR / "trained_models")
REPORTS_DIR = _path("REPORTS_DIR", BASE_DIR / "reports")

# Rainfall training data. The real per-segment file from Open-Meteo (made by
# `python -m training.download_rainfall`) is used when it exists; otherwise the uploaded city file.
# At serving time the app always reads the file the loaded model was trained on (see rainfall_service).
RAINFALL_CITY_DATASET = DATASETS_DIR / "india_rainfall_2016_2025.csv"
RAINFALL_CORRIDOR_DATASET = DATASETS_DIR / "corridor_rainfall_open_meteo.csv"
RAINFALL_DATASET = _path(
    "RAINFALL_DATASET",
    RAINFALL_CORRIDOR_DATASET if RAINFALL_CORRIDOR_DATASET.exists() else RAINFALL_CITY_DATASET,
)
OPEN_METEO_CACHE_DIR = DATASETS_DIR / "open_meteo_cache"
# Rain forecasts made 24 h ahead per segment (training/download_forecasts.py). Optional model input.
RAIN_FORECAST_DATASET = DATASETS_DIR / "corridor_rain_forecast_open_meteo.csv"
OPEN_METEO_FORECAST_CACHE_DIR = DATASETS_DIR / "open_meteo_forecast_cache"
ACCIDENT_DATASET = DATASETS_DIR / "indian_roads_dataset.csv"
SIGNIFICANT_DATES_DATASET = DATASETS_DIR / "significant_dates.csv"

RAINFALL_MODEL_PATH = MODELS_DIR / "rainfall_model.joblib"
ACCIDENT_MODEL_PATH = MODELS_DIR / "accident_model.joblib"
TRAFFIC_MODEL_PATH = MODELS_DIR / "traffic_model.joblib"

APP_NAME = "Freight Disruption Digital Twin - ML API"
APP_VERSION = "1.0.0"

# Comma-separated list of frontend origins allowed to call the API.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,"
        "http://127.0.0.1:8080,http://localhost:8080,"
        "https://preevind7.github.io",
    ).split(",")
    if origin.strip()
]

RANDOM_STATE = 42

# ---------------- authentication ----------------
USERS_FILE = _path("USERS_FILE", BASE_DIR / "users.json")
# Secret used to sign login tokens. Set JWT_SECRET in .env (python -m app.manage_users secret prints a new one).
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", "480"))  # 8 hours
MAX_FAILED_LOGINS = int(os.getenv("MAX_FAILED_LOGINS", "5"))
LOCKOUT_MINUTES = int(os.getenv("LOCKOUT_MINUTES", "5"))
# Let people create their own accounts from the login page (set to false to disable).
ALLOW_REGISTRATION = os.getenv("ALLOW_REGISTRATION", "true").strip().lower() in ("1", "true", "yes")

# ---------------- serving the frontend (one link for app + API, e.g. through a Cloudflare tunnel) ----------------
# Default: "DEMO FOLDER" next to backend (this PC) or "frontend" next to backend (the GitHub repository layout).
_FRONTEND_CANDIDATES = [BASE_DIR.parent / "DEMO FOLDER", BASE_DIR.parent / "frontend"]
FRONTEND_DIR = _path("FRONTEND_DIR", next((p for p in _FRONTEND_CANDIDATES if (p / "index.html").exists()), _FRONTEND_CANDIDATES[0]))
SERVE_FRONTEND = os.getenv("SERVE_FRONTEND", "true").strip().lower() in ("1", "true", "yes")

# ---------------- corridor forecast ----------------
CORRIDORS_FILE = BASE_DIR / "app" / "data" / "corridors.json"
# Fetch the last 7 days of rain from Open-Meteo (free, no key) for current dates. false = offline replay only.
LIVE_WEATHER = os.getenv("LIVE_WEATHER", "true").strip().lower() in ("1", "true", "yes")
LIVE_WEATHER_TIMEOUT = float(os.getenv("LIVE_WEATHER_TIMEOUT", "4"))
FORECAST_CACHE_MINUTES = int(os.getenv("FORECAST_CACHE_MINUTES", "30"))
IST = timezone(timedelta(hours=5, minutes=30))

# ---------------- live traffic (TomTom Traffic Flow API, free key, no card) ----------------
# The key stays in backend/.env and is only used by the backend; it is never sent to the browser.
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "").strip()
LIVE_TRAFFIC = os.getenv("LIVE_TRAFFIC", "true").strip().lower() in ("1", "true", "yes")
LIVE_TRAFFIC_TIMEOUT = float(os.getenv("LIVE_TRAFFIC_TIMEOUT", "5"))
TRAFFIC_CACHE_MINUTES = int(os.getenv("TRAFFIC_CACHE_MINUTES", "30"))     # 35 segments x 2 per hour x 24 h = 1,680 calls/day
TRAFFIC_DAILY_BUDGET = int(os.getenv("TRAFFIC_DAILY_BUDGET", "2300"))     # TomTom free plan: 2,500 non-tile requests/day
TRAFFIC_LOG_FILE = _path("TRAFFIC_LOG_FILE", DATASETS_DIR / "traffic_observations.csv")


def today_ist() -> date:
    return datetime.now(IST).date()

# ---------------- AI assistant (LLM) ----------------
# Provider: none | gemini | groq | openai | anthropic | ollama   (see README "AI assistant")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()          # empty = provider default
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()    # empty = provider default
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1500"))
# Optional backup provider, used automatically when the main one is busy, too slow or over its quota.
# Example: LLM_FALLBACK_PROVIDER=groq + LLM_FALLBACK_API_KEY=gsk_... (free key from console.groq.com/keys)
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "none").strip().lower()
LLM_FALLBACK_API_KEY = os.getenv("LLM_FALLBACK_API_KEY", "").strip()
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "").strip()
LLM_FALLBACK_BASE_URL = os.getenv("LLM_FALLBACK_BASE_URL", "").strip()
LLM_PRIMARY_TIMEOUT_WITH_FALLBACK = float(os.getenv("LLM_PRIMARY_TIMEOUT_WITH_FALLBACK", "40"))
LLM_FALLBACK_TIMEOUT = float(os.getenv("LLM_FALLBACK_TIMEOUT", "30"))   # main 40 s + backup 30 s stays under the page's 90 s limit
ASSISTANT_REQUESTS_PER_MINUTE = int(os.getenv("ASSISTANT_REQUESTS_PER_MINUTE", "10"))
