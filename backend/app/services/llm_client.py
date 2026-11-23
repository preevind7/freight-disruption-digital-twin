"""Minimal client for large language model (LLM) APIs, using only the Python standard library.

Supported providers (set LLM_PROVIDER in backend/.env):
  gemini     Google Gemini via its OpenAI-compatible endpoint (free tier available, key from Google AI Studio)
  groq       Groq (free tier available, key from console.groq.com)
  openai     OpenAI (paid; set LLM_MODEL)
  anthropic  Anthropic Claude Messages API (paid)
  ollama     a model running on your own computer with Ollama (free, offline, no key)

Optional backup: LLM_FALLBACK_PROVIDER / LLM_FALLBACK_API_KEY / LLM_FALLBACK_MODEL. When the main
provider is busy (HTTP 5xx), too slow or over its quota, the same question goes to the backup.

The API keys are read on the backend only and are never sent to the browser.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from app import config

PRESETS: Dict[str, dict] = {
    "gemini": {"kind": "openai", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-3.8-flash",
               "needs_key": True, "extra": {"reasoning_effort": "low"}, "label": "Google Gemini"},
    # Groq deprecated llama-3.3-70b-versatile and recommends openai/gpt-oss-120b instead.
    "groq": {"kind": "openai", "base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-120b", "needs_key": True,
             "extra": {"reasoning_effort": "low"}, "label": "Groq"},
    "openai": {"kind": "openai", "base_url": "https://api.openai.com/v1", "model": "", "needs_key": True, "label": "OpenAI"},
    "anthropic": {"kind": "anthropic", "base_url": "https://api.anthropic.com/v1", "model": "claude-haiku-5-5", "needs_key": True, "label": "Anthropic Claude"},
    "ollama": {"kind": "openai", "base_url": "http://localhost:11434/v1", "model": "llama3.2", "needs_key": False, "label": "Ollama (local)"},
}


class LLMError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502, transient: bool = False):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.transient = transient  # worth retrying (provider busy / temporary failure)


def _resolve(provider: str, api_key: str, model: str, base_url: str, names: dict) -> dict:
    if provider in ("", "none", "off", "false"):
        return {"enabled": False, "provider": "none", "reason": f"No AI provider is set. Add {names['provider']} and {names['key']} to backend/.env (see README, section 'AI assistant')."}
    preset = PRESETS.get(provider)
    if preset is None:
        return {"enabled": False, "provider": provider, "reason": f"Unknown {names['provider']} '{provider}'. Use one of: {', '.join(PRESETS)}."}
    model = model or preset["model"]
    if not model:
        return {"enabled": False, "provider": provider, "reason": f"Set {names['model']} in backend/.env for provider '{provider}'."}
    if preset["needs_key"] and not api_key:
        return {"enabled": False, "provider": provider, "reason": f"{names['key']} is empty in backend/.env (needed for {preset['label']})."}
    extra = dict(preset.get("extra", {}))
    if provider == "groq" and not model.startswith("openai/gpt-oss"):
        extra.pop("reasoning_effort", None)        # only Groq's gpt-oss models accept it
    return {"enabled": True, "provider": provider, "label": preset["label"], "kind": preset["kind"], "model": model,
            "base_url": (base_url or preset["base_url"]).rstrip("/"), "extra": extra, "api_key": api_key,
            "key_var": names["key"], "model_var": names["model"], "timeout": config.LLM_TIMEOUT, "reason": None}


def settings() -> dict:
    """Resolved main provider settings plus whether the assistant can run. Contains the key: never return it to a client."""
    return _resolve(config.LLM_PROVIDER, config.LLM_API_KEY, config.LLM_MODEL, config.LLM_BASE_URL,
                    {"provider": "LLM_PROVIDER", "key": "LLM_API_KEY", "model": "LLM_MODEL"})


def fallback_settings() -> dict:
    """Backup provider settings (enabled=False when none is configured)."""
    return _resolve(config.LLM_FALLBACK_PROVIDER, config.LLM_FALLBACK_API_KEY, config.LLM_FALLBACK_MODEL, config.LLM_FALLBACK_BASE_URL,
                    {"provider": "LLM_FALLBACK_PROVIDER", "key": "LLM_FALLBACK_API_KEY", "model": "LLM_FALLBACK_MODEL"})


_call = threading.local()   # per-request timeout, so _post keeps a simple signature


# Some providers (e.g. Groq, behind Cloudflare) refuse the default "Python-urllib" User-Agent with HTTP 403.
USER_AGENT = "FreightTwin/1.0 (freight disruption digital twin backend)"


def _post(url: str, headers: dict, body: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **headers})
    key_var = getattr(_call, "key_var", None) or "LLM_API_KEY"
    model_var = getattr(_call, "model_var", None) or "LLM_MODEL"
    try:
        with urllib.request.urlopen(request, timeout=getattr(_call, "timeout", None) or config.LLM_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            raw = exc.read().decode("utf-8", "replace")
            try:
                payload = json.loads(raw)
                err = payload.get("error", payload)
                detail = (err.get("message") if isinstance(err, dict) else str(err)) or ""
            except ValueError:
                detail = re.sub(r"<[^>]+>", " ", raw)          # HTML error pages (e.g. Cloudflare) -> plain text
                detail = re.sub(r"\s+", " ", detail).strip()
        except Exception:
            pass
        detail = detail[:300]
        if exc.code == 401:
            raise LLMError(f"The AI provider rejected the API key (HTTP 401). Check {key_var} in backend/.env." + (f" ({detail})" if detail else ""), 502)
        if exc.code == 403:
            raise LLMError(f"The AI provider refused access (HTTP 403): the key ({key_var}), access to this model, or a network block."
                           + (f" ({detail})" if detail else ""), 502)
        if exc.code == 429:
            raise LLMError("The AI provider's rate limit or free quota was reached. Wait a minute and try again." + (f" ({detail})" if detail else ""), 429)
        if exc.code == 404:
            raise LLMError(f"The AI provider does not know model or URL (404). Check {model_var} in backend/.env. {detail}".strip(), 502)
        if exc.code in (500, 502, 503, 504):
            raise LLMError(f"The AI provider is busy or temporarily unavailable (HTTP {exc.code}). Try again in a minute, or set {model_var} "
                           f"in backend/.env to another model." + (f" ({detail})" if detail else ""), 503, transient=True)
        raise LLMError(f"The AI provider returned HTTP {exc.code}. {detail}".strip(), 502)
    except TimeoutError:
        raise LLMError("The AI provider did not answer in time. Try again.", 504)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise LLMError("The AI provider did not answer in time. Try again.", 504)
        raise LLMError(f"Cannot reach the AI provider ({exc.reason}). Check the internet connection of the backend computer.", 502)


def _clean(text: str) -> str:
    # Some open models wrap their reasoning in <think>...</think>; only the final answer is shown.
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


RETRY_WAITS = (2, 5)  # seconds to wait before retry 1 and retry 2 when the provider is busy
FALLBACK_WORTHY = (429, 503, 504)   # busy, over quota, too slow -> try the backup provider


def chat(system: str, messages: List[Dict[str, str]], cfg: Optional[dict] = None, waits: Optional[tuple] = None) -> str:
    """Ask one provider; retries automatically when it is temporarily busy (HTTP 5xx)."""
    cfg = cfg or settings()
    if not cfg["enabled"]:
        raise LLMError(cfg["reason"], 503)
    waits = RETRY_WAITS if waits is None else waits
    for attempt in range(len(waits) + 1):
        try:
            return _chat_once(system, messages, cfg)
        except LLMError as exc:
            if not exc.transient or attempt == len(waits):
                raise
            time.sleep(waits[attempt])
    raise LLMError("The AI provider is unavailable.", 503)


def chat_with_fallback(system: str, messages: List[Dict[str, str]]) -> tuple:
    """(answer, settings of the provider that answered, note). Uses the backup provider when the main one
    is busy, too slow or over quota. With a backup, the main provider gets one quick retry and a shorter timeout."""
    main, backup = settings(), fallback_settings()
    if not backup["enabled"]:
        return chat(system, messages, main), main, None
    if not main["enabled"]:
        return chat(system, messages, backup), backup, f"Main AI provider not configured: {main['reason']}"
    main = dict(main, timeout=min(config.LLM_TIMEOUT, config.LLM_PRIMARY_TIMEOUT_WITH_FALLBACK))
    backup = dict(backup, timeout=min(config.LLM_TIMEOUT, config.LLM_FALLBACK_TIMEOUT))
    try:
        return chat(system, messages, main, waits=RETRY_WAITS[:1]), main, None
    except LLMError as exc:
        if exc.status_code not in FALLBACK_WORTHY and not exc.transient:
            raise
        try:
            answer = chat(system, messages, backup, waits=RETRY_WAITS[:1])
        except LLMError as backup_exc:
            raise LLMError(f"Both AI providers failed. {main['label']}: {exc.message} {backup['label']}: {backup_exc.message}",
                           backup_exc.status_code, transient=backup_exc.transient) from None
        return answer, backup, f"{main['label']} ({main['model']}) was unavailable ({exc.message.split('.')[0]}), so the backup answered."


def _chat_once(system: str, messages: List[Dict[str, str]], cfg: dict) -> str:
    _call.timeout, _call.key_var, _call.model_var = cfg.get("timeout"), cfg.get("key_var"), cfg.get("model_var")
    key = cfg.get("api_key", config.LLM_API_KEY)
    if cfg["kind"] == "anthropic":
        data = _post(
            cfg["base_url"] + "/messages",
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
            {"model": cfg["model"], "max_tokens": config.LLM_MAX_TOKENS, "temperature": 0.2, "system": system, "messages": messages},
        )
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    else:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        body = {"model": cfg["model"], "messages": [{"role": "system", "content": system}] + messages,
                "max_tokens": config.LLM_MAX_TOKENS, "temperature": 0.2, **cfg.get("extra", {})}
        data = _post(cfg["base_url"] + "/chat/completions", headers, body)
        choices = data.get("choices") or []
        text = (choices[0].get("message", {}).get("content") or "") if choices else ""
    text = _clean(text)
    if not text:
        raise LLMError("The AI provider returned an empty answer. Try again, or raise LLM_MAX_TOKENS in backend/.env.", 502)
    return text
