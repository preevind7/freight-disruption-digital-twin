"""The backend can serve the frontend (one public link); hidden files such as .git are never served."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.utils.frontend import frontend_available, frontend_files


def make_site(tmp_path):
    site = tmp_path / "DEMO FOLDER"
    (site / "js").mkdir(parents=True)
    (site / ".git").mkdir()
    (site / "index.html").write_text("<!doctype html><title>Freight Twin</title>", encoding="utf-8")
    (site / "js" / "api.js").write_text("console.log('api')", encoding="utf-8")
    (site / ".git" / "config").write_text("[remote] url = https://token@github.com/x/y", encoding="utf-8")
    (site / ".env").write_text("SECRET=1", encoding="utf-8")
    return site


def test_frontend_served_and_hidden_files_blocked(tmp_path):
    site = make_site(tmp_path)
    assert frontend_available(site)
    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"status": "ok"}
    app.mount("/", frontend_files(site), name="frontend")
    c = TestClient(app)
    assert "Freight Twin" in c.get("/").text
    assert c.get("/js/api.js").status_code == 200
    assert c.get("/api/health").json() == {"status": "ok"}          # API routes still win
    for hidden in ("/.git/config", "/.env", "/js/../.git/config", "/.git/", "/%2Egit/config"):
        r = c.get(hidden)
        assert r.status_code == 404 and "token" not in r.text and "SECRET" not in r.text, hidden


def test_folder_without_index_is_not_served(tmp_path):
    assert not frontend_available(tmp_path)
    assert not frontend_available(tmp_path / "missing")
