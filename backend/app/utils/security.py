"""Password hashing and login tokens.

Passwords: PBKDF2-HMAC-SHA256 with a random salt (Python standard library, no plain text stored).
Tokens:    JWT signed with JWT_SECRET from .env, valid for TOKEN_EXPIRE_MINUTES.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt

from app import config

logger = logging.getLogger(__name__)

PBKDF2_ITERATIONS = 310_000

_temporary_secret = None


def _secret() -> str:
    """JWT_SECRET from .env, or a per-process temporary one (with a warning) if it is missing."""
    global _temporary_secret
    if config.JWT_SECRET:
        return config.JWT_SECRET
    if _temporary_secret is None:
        _temporary_secret = secrets.token_urlsafe(48)
        logger.warning(
            "JWT_SECRET is not set in .env - using a temporary secret. Users will be logged out whenever the "
            "server restarts. Run: python -m app.manage_users secret"
        )
    return _temporary_secret


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS, base64.b64encode(salt).decode(), base64.b64encode(digest).decode()
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt_b64, digest_b64 = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), base64.b64decode(salt_b64), int(iterations))
        return hmac.compare_digest(digest, base64.b64decode(digest_b64))
    except (ValueError, TypeError):
        return False


def create_access_token(username: str, role: str) -> Dict[str, Any]:
    expires = datetime.now(timezone.utc) + timedelta(minutes=config.TOKEN_EXPIRE_MINUTES)
    token = jwt.encode({"sub": username, "role": role, "exp": expires}, _secret(), algorithm=config.JWT_ALGORITHM)
    return {"access_token": token, "token_type": "bearer", "expires_at": expires.replace(microsecond=0).isoformat()}


def decode_access_token(token: str) -> Dict[str, Any]:
    """Raises jwt.ExpiredSignatureError / jwt.InvalidTokenError on bad tokens."""
    return jwt.decode(token, _secret(), algorithms=[config.JWT_ALGORITHM], options={"require": ["sub", "exp"]})
