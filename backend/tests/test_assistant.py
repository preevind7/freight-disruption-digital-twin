"""AI assistant tests. The real LLM provider is replaced by a fake, so no internet or API key is needed."""
import json

import pytest

from app import config
from app.services import assistant_service, corridor_service, llm_client

URL = "/api/assistant/chat"
BODY = {"question": "Why is S01 risky and what should I do?", "corridor_id": "mumbai-bengaluru", "date": "2025-06-19", "hour": 15,
        "segment_id": "S01", "trips": [{"id": "TRK-2041", "vehicle": "Truck 16 t", "origin": "Mumbai", "destination": "Bengaluru", "current_segment": "S10", "status": "Watch"}]}


@pytest.fixture
def fake_provider(monkeypatch):
    """Configure a provider and capture what is sent to it."""
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_MODEL", "")
    monkeypatch.setattr(config, "ASSISTANT_REQUESTS_PER_MINUTE", 100)
    assistant_service._rate.clear()
    sent = {}

    def fake_post(url, headers, body):
        sent.update(url=url, headers=headers, body=body)
        system = body["messages"][0]["content"] if "messages" in body and body["messages"][0]["role"] == "system" else body.get("system", "")
        context = json.loads(system.split("CONTEXT (JSON):", 1)[1])
        focus = context["focus_segment"]
        answer = sent.get("answer_template", "{id} risk is {risk} ({band}) at {time}; heavy-rain chance {pct}%.").format(
            id=focus["id"], risk=focus["risk_now"], band=focus["band_now"], time=context["time_now"][:5], pct=round(focus["p_heavy_rain"] * 100))
        if url.endswith("/messages"):
            return {"content": [{"type": "text", "text": answer}]}
        return {"choices": [{"message": {"content": answer}}]}

    monkeypatch.setattr(llm_client, "_post", fake_post)
    return sent


def test_requires_login(anon_client):
    assert anon_client.post(URL, json=BODY).status_code == 401
    assert anon_client.get("/api/assistant/status").status_code == 401


def test_not_configured_returns_503(client, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "none")
    assert client.get("/api/assistant/status").json()["enabled"] is False
    r = client.post(URL, json=BODY)
    assert r.status_code == 503 and "LLM_PROVIDER" in r.json()["detail"]


def test_missing_key_is_reported(client, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    status = client.get("/api/assistant/status").json()
    assert status["enabled"] is False and "LLM_API_KEY" in status["reason"]


def test_answer_grounded_in_forecast(client, fake_provider):
    status = client.get("/api/assistant/status").json()
    assert status["enabled"] and status["model"] == "openai/gpt-oss-120b" and status["fallback"] is None
    r = client.post(URL, json=BODY)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"].startswith("S01 risk is 0.")
    assert body["grounding"]["all_supported"] is True and body["grounding"]["numbers_checked"] >= 2
    # what the provider received
    assert fake_provider["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert fake_provider["headers"]["Authorization"] == "Bearer test-key"
    system = fake_provider["body"]["messages"][0]["content"]
    assert "Use ONLY the facts" in system and '"focus_segment"' in system and "TRK-2041" in system
    assert fake_provider["body"]["messages"][-1] == {"role": "user", "content": BODY["question"]}


def test_invented_numbers_are_flagged(client, fake_provider):
    # invented values that cannot occur in any real context (a risk score is 0-1; no input has 3 decimals near 987)
    fake_provider["answer_template"] = "{id} risk is {risk}. Expect a 137 minute delay and 987.654 mm of rain at {time}."
    body = client.post(URL, json=BODY).json()
    assert body["grounding"]["all_supported"] is False
    assert "137" in body["grounding"]["unsupported"] and "987.654" in body["grounding"]["unsupported"]


def test_history_is_forwarded(client, fake_provider):
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    client.post(URL, json={**BODY, "history": history})
    assert fake_provider["body"]["messages"][1:3] == history


def test_anthropic_format(client, fake_provider, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    r = client.post(URL, json=BODY)
    assert r.status_code == 200, r.text
    assert fake_provider["url"] == "https://api.anthropic.com/v1/messages"
    assert fake_provider["headers"]["x-api-key"] == "test-key" and fake_provider["body"]["model"] == "claude-haiku-5-5"


def test_provider_errors_are_passed_on(client, fake_provider, monkeypatch):
    def quota(*args):
        raise llm_client.LLMError("The AI provider's rate limit or free quota was reached.", 429)
    monkeypatch.setattr(llm_client, "_post", quota)
    r = client.post(URL, json=BODY)
    assert r.status_code == 429 and "quota" in r.json()["detail"]


def test_rate_limit_per_user(client, fake_provider, monkeypatch):
    monkeypatch.setattr(config, "ASSISTANT_REQUESTS_PER_MINUTE", 2)
    codes = [client.post(URL, json=BODY).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_input_validation(client, fake_provider):
    assert client.post(URL, json={**BODY, "question": ""}).status_code == 422
    assert client.post(URL, json={**BODY, "hour": 25}).status_code == 422
    assert client.post(URL, json={**BODY, "corridor_id": "delhi-agra"}).status_code == 422


def test_check_numbers_rules():
    context = {"a": 0.6338, "b": 69.2, "c": "Heavy-rain alert on S01: 63% chance", "d": "2025-06-19", "e": [0.82, 125]}
    ok = assistant_service.check_numbers("S01 on 2025-06-19 at 15:00: risk 0.82, 63% (63.4%) heavy rain, about 69 mm, delay 125 min, NH48A.", context)
    assert ok["all_supported"], ok
    bad = assistant_service.check_numbers("Risk 0.55 on 2024-01-01 with 300 mm.", context)
    assert set(bad["unsupported"]) == {"0.55", "2024-01-01", "300"}


def test_number_must_belong_to_the_named_segment(client):
    from datetime import date
    user = {"name": "T", "role": "dispatcher", "role_title": "Freight Dispatcher", "username": "t"}
    ctx = assistant_service.build_context(user, "mumbai-bengaluru", date(2025, 6, 19), 15, "S01", [])
    s01 = ctx["focus_segment"]
    others = [s for s in ctx["segments_now"] if s["id"] != "S01"]
    for other in others:
        right = assistant_service.check_numbers(f"S01 is at {s01['risk_now']}. {other['id']} is at {other['risk_now']}.", ctx)
        assert right["all_supported"], other["id"]
    # Segment awareness, on a fixed context (live values depend on the rainfall data in use and may
    # coincide with a general fact such as a model metric, which is then rightly accepted).
    fixed = {"segments_now": [{"id": "S01", "risk_now": 0.82}, {"id": "S04", "risk_now": 0.45}],
             "focus_segment": {"id": "S01", "risk_now": 0.82}, "heavy_alert_threshold": 0.0531}
    assert assistant_service.check_numbers("S01 is at 0.82. S04 is at 0.45.", fixed)["all_supported"]
    assert not assistant_service.check_numbers("S04 is at 0.82.", fixed)["all_supported"]


def test_busy_provider_is_retried(client, fake_provider, monkeypatch):
    import urllib.error
    calls = {"n": 0}
    real = llm_client._post

    def flaky(url, headers, body):
        calls["n"] += 1
        if calls["n"] <= 2:  # Google-style "busy" twice, then success
            raise llm_client.LLMError("busy (HTTP 503)", 503, transient=True)
        return real(url, headers, body)
    monkeypatch.setattr(llm_client, "_post", flaky)
    monkeypatch.setattr(llm_client, "RETRY_WAITS", (0, 0))
    r = client.post(URL, json=BODY)
    assert r.status_code == 200 and calls["n"] == 3


def test_busy_provider_gives_up_after_retries(client, fake_provider, monkeypatch):
    def always_busy(*args):
        raise llm_client.LLMError("The AI provider is busy or temporarily unavailable (HTTP 503).", 503, transient=True)
    monkeypatch.setattr(llm_client, "_post", always_busy)
    monkeypatch.setattr(llm_client, "RETRY_WAITS", (0, 0))
    r = client.post(URL, json=BODY)
    assert r.status_code == 503 and "busy" in r.json()["detail"]


def test_http_503_from_provider_is_marked_transient(monkeypatch):
    import io, urllib.error
    def raise_503(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", {}, io.BytesIO(b'{"error":{"message":"The model is overloaded."}}'))
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", raise_503)
    with pytest.raises(llm_client.LLMError) as err:
        llm_client._post("https://example.invalid/v1/chat/completions", {}, {})
    assert err.value.transient and err.value.status_code == 503 and "overloaded" in err.value.message



# ---------------- backup provider ----------------
@pytest.fixture
def with_backup(fake_provider, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "LLM_API_KEY", "main-key")
    monkeypatch.setattr(config, "LLM_FALLBACK_PROVIDER", "groq")
    monkeypatch.setattr(config, "LLM_FALLBACK_API_KEY", "backup-key")
    monkeypatch.setattr(config, "LLM_FALLBACK_MODEL", "")
    monkeypatch.setattr(llm_client, "RETRY_WAITS", (0, 0))
    return fake_provider


def test_backup_answers_when_main_is_busy(client, with_backup, monkeypatch):
    real = llm_client._post
    seen = []

    def main_busy(url, headers, body):
        seen.append((url, headers.get("Authorization"), getattr(llm_client._call, "timeout", None)))
        if "googleapis" in url:
            raise llm_client.LLMError("The AI provider is busy or temporarily unavailable (HTTP 503).", 503, transient=True)
        return real(url, headers, body)
    monkeypatch.setattr(llm_client, "_post", main_busy)
    r = client.post(URL, json=BODY)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "Groq" and body["model"] == "openai/gpt-oss-120b" and "Google Gemini" in body["fallback_note"]
    google = [s for s in seen if "googleapis" in s[0]]
    assert len(google) == 2                                   # one quick retry only, then the backup
    assert all(s[1] == "Bearer main-key" and s[2] == config.LLM_PRIMARY_TIMEOUT_WITH_FALLBACK for s in google)
    assert seen[-1][1] == "Bearer backup-key" and "groq.com" in seen[-1][0]
    assert with_backup["body"]["reasoning_effort"] == "low"
    assert "backup-key" not in r.text and "main-key" not in r.text


def test_backup_also_used_on_timeout_and_quota(client, with_backup, monkeypatch):
    real = llm_client._post
    for error in (llm_client.LLMError("did not answer in time", 504), llm_client.LLMError("quota reached", 429)):
        def main_fails(url, headers, body, error=error):
            if "googleapis" in url:
                raise error
            return real(url, headers, body)
        monkeypatch.setattr(llm_client, "_post", main_fails)
        assistant_service._rate.clear()
        assert client.post(URL, json=BODY).json()["provider"] == "Groq"


def test_no_backup_for_a_rejected_key(client, with_backup, monkeypatch):
    def bad_key(url, headers, body):
        raise llm_client.LLMError("The AI provider rejected the API key.", 502)
    monkeypatch.setattr(llm_client, "_post", bad_key)
    r = client.post(URL, json=BODY)
    assert r.status_code == 502 and "rejected the API key" in r.json()["detail"]


def test_both_failing_reports_both(client, with_backup, monkeypatch):
    def all_busy(url, headers, body):
        raise llm_client.LLMError("busy (HTTP 503)", 503, transient=True)
    monkeypatch.setattr(llm_client, "_post", all_busy)
    r = client.post(URL, json=BODY)
    assert r.status_code == 503 and "Both AI providers failed" in r.json()["detail"]


def test_main_answer_has_no_fallback_note_and_status_shows_backup(client, with_backup):
    r = client.post(URL, json=BODY).json()
    assert r["provider"] == "Google Gemini" and r["fallback_note"] is None
    st = client.get("/api/assistant/status").json()
    assert st["fallback"] == {"provider_label": "Groq", "model": "openai/gpt-oss-120b"} and "key" not in json.dumps(st).lower().replace("api_key", "")


def test_timeout_inside_urlerror_is_a_504(monkeypatch):
    import urllib.error
    def slow(request, timeout):
        raise urllib.error.URLError(TimeoutError("timed out"))
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", slow)
    with pytest.raises(llm_client.LLMError) as err:
        llm_client._post("https://example.invalid/v1/chat/completions", {}, {})
    assert err.value.status_code == 504


def test_requests_send_a_non_python_user_agent(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"choices":[{"message":{"content":"ok"}}]}'

    def capture(request, timeout):
        seen["ua"] = request.get_header("User-agent")
        return Resp()
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", capture)
    llm_client._post("https://api.groq.com/openai/v1/chat/completions", {"Authorization": "Bearer x"}, {})
    assert seen["ua"] == llm_client.USER_AGENT and "python" not in seen["ua"].lower()


def test_403_html_page_and_401_name_the_right_setting(monkeypatch):
    import io, urllib.error
    html = b"<html><body><h1>Access denied</h1><p>error code: 1010</p></body></html>"

    def forbidden(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, io.BytesIO(html))
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", forbidden)
    llm_client._call.key_var = "LLM_FALLBACK_API_KEY"
    with pytest.raises(llm_client.LLMError) as err:
        llm_client._post("https://api.groq.com/openai/v1/chat/completions", {}, {})
    assert "HTTP 403" in err.value.message and "LLM_FALLBACK_API_KEY" in err.value.message and "error code: 1010" in err.value.message
    assert "<" not in err.value.message

    def unauthorized(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"error":{"message":"Invalid API Key"}}'))
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", unauthorized)
    with pytest.raises(llm_client.LLMError) as err:
        llm_client._post("https://api.groq.com/openai/v1/chat/completions", {}, {})
    assert "HTTP 401" in err.value.message and "LLM_FALLBACK_API_KEY" in err.value.message and "Invalid API Key" in err.value.message
    llm_client._call.key_var = None


def test_backup_gets_its_own_timeout(client, with_backup, monkeypatch):
    real = llm_client._post
    timeouts = {}

    def record(url, headers, body):
        timeouts["google" if "googleapis" in url else "groq"] = llm_client._call.timeout
        if "googleapis" in url:
            raise llm_client.LLMError("did not answer in time", 504)
        return real(url, headers, body)
    monkeypatch.setattr(llm_client, "_post", record)
    assert client.post(URL, json=BODY).status_code == 200
    assert timeouts == {"google": config.LLM_PRIMARY_TIMEOUT_WITH_FALLBACK, "groq": config.LLM_FALLBACK_TIMEOUT}
