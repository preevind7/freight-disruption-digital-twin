"""Serve the frontend (index.html, css, js) from the backend, so one address works for the app and the API.

Hidden files and folders (names starting with ".", e.g. .git, .env) are never served.
"""
from __future__ import annotations

from pathlib import Path

from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles


class SafeStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        parts = [p for p in path.replace("\\", "/").split("/") if p and p != "."]   # "." = the site root
        if any(p.startswith(".") for p in parts):
            raise HTTPException(status_code=404)
        return await super().get_response(path, scope)


def frontend_available(directory: Path) -> bool:
    return directory.is_dir() and (directory / "index.html").is_file()


def frontend_files(directory: Path) -> SafeStaticFiles:
    return SafeStaticFiles(directory=str(directory), html=True)
