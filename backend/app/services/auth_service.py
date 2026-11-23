"""Login checks and the `get_current_user` dependency that protects the API."""
from __future__ import annotations

import threading
import time
from typing import Dict, Tuple

import jwt
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer

from app import config
from app.services import user_store
from app.utils.security import create_access_token, decode_access_token, hash_password, verify_password
from app.utils.validation import AuthError, InvalidInputError

# tokenUrl powers the "Authorize" button in Swagger UI (/docs).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)

_DUMMY_HASH = hash_password("timing-equaliser")  # so unknown usernames take as long as wrong passwords
_failures: Dict[str, Tuple[int, float]] = {}  # username -> (failed attempts, time of first failure)
_failures_lock = threading.Lock()


def _check_lockout(username: str) -> None:
    with _failures_lock:
        count, first = _failures.get(username, (0, 0.0))
        window = config.LOCKOUT_MINUTES * 60
        if count and time.time() - first > window:
            _failures.pop(username, None)
            return
        if count >= config.MAX_FAILED_LOGINS:
            wait = int(window - (time.time() - first)) // 60 + 1
            raise AuthError(f"Too many failed attempts. Try again in about {wait} minute(s).", status_code=429)


def _record_failure(username: str) -> None:
    with _failures_lock:
        count, first = _failures.get(username, (0, time.time()))
        _failures[username] = (count + 1, first)


def authenticate(username: str, password: str) -> dict:
    key = username.strip().lower()
    _check_lockout(key)
    user = user_store.get_user(key)
    valid = verify_password(password, user["password_hash"] if user else _DUMMY_HASH)
    if not user or not valid or user.get("disabled"):
        _record_failure(key)
        raise AuthError("Incorrect username or password.")
    with _failures_lock:
        _failures.pop(key, None)
    profile = user_store.public_profile(user)
    return {**create_access_token(user["username"], user["role"]), "user": profile}


def register(name: str, username: str, password: str, role: str) -> dict:
    """Create a new account (password stored only as a hash). Returns the public profile."""
    if not config.ALLOW_REGISTRATION:
        raise AuthError("Creating accounts is disabled on this server. Ask the administrator for an account.", status_code=403)
    if password.strip().lower() == username.lower() or username.lower() in password.lower():
        raise InvalidInputError("Password must not contain your username.", "password")
    user = {"username": username, "name": name, "role": role, "password_hash": hash_password(password)}
    if not user_store.create_user(user):
        raise InvalidInputError(f"The username '{username}' is already taken. Please choose another one.", "username")
    return user_store.public_profile(user)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    if not token:
        raise AuthError("Login required. Sign in to get an access token.")
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise AuthError("Your session has expired. Please sign in again.")
    except jwt.InvalidTokenError:
        raise AuthError("Invalid access token. Please sign in again.")
    user = user_store.get_user(str(payload.get("sub", "")))
    if not user or user.get("disabled"):
        raise AuthError("This account no longer exists or is disabled.")
    return user_store.public_profile(user)
