"""Very small user store: backend/users.json (password hashes only, never plain text).

Manage accounts with:  python -m app.manage_users --help
"""
from __future__ import annotations

import json
import threading
from typing import Dict, List, Optional

from app import config

# The four target users from the project proposal; ids match the frontend roles.
ROLES = {
    "dispatcher": "Freight Dispatcher",
    "operator": "Fleet Operator",
    "planner": "Highway Planner",
    "researcher": "Researcher / Student",
}

_lock = threading.Lock()


def _read() -> Dict[str, dict]:
    if not config.USERS_FILE.exists():
        return {}
    with open(config.USERS_FILE, encoding="utf-8") as handle:
        data = json.load(handle)
    return {u["username"].lower(): u for u in data.get("users", [])}


def _write(users: Dict[str, dict]) -> None:
    config.USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.USERS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"users": sorted(users.values(), key=lambda u: u["username"])}, handle, indent=2)
    tmp.replace(config.USERS_FILE)


def get_user(username: str) -> Optional[dict]:
    return _read().get(username.strip().lower())


def list_users() -> List[dict]:
    return sorted(_read().values(), key=lambda u: u["username"])


def save_user(user: dict) -> None:
    if user["role"] not in ROLES:
        raise ValueError(f"role must be one of: {', '.join(ROLES)}")
    with _lock:
        users = _read()
        users[user["username"].lower()] = user
        _write(users)


def create_user(user: dict) -> bool:
    """Add a new user. Returns False (and changes nothing) if the username is taken."""
    if user["role"] not in ROLES:
        raise ValueError(f"role must be one of: {', '.join(ROLES)}")
    with _lock:
        users = _read()
        key = user["username"].lower()
        if key in users:
            return False
        users[key] = user
        _write(users)
        return True


def delete_user(username: str) -> bool:
    with _lock:
        users = _read()
        if users.pop(username.strip().lower(), None) is None:
            return False
        _write(users)
        return True


def public_profile(user: dict) -> dict:
    return {
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
        "role_title": ROLES[user["role"]],
    }
