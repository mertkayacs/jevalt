"""Teacher client: quota pauses, retry limits and the response cache (no network)."""

import asyncio
import json

import httpx
import pytest

from jevalt.data import providers
from jevalt.data.providers import Client, QuotaError

OK = {"model": "glm-5.3", "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}], "usage": {"completion_tokens": 3}}


def client_with(tmp_path, monkeypatch, responses):
    monkeypatch.setattr(providers, "_load_keys", lambda: {"kimi": "k-test", "zai": "z-test", "ollama": "o-test"})
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        status, body = responses[min(len(calls), len(responses)) - 1]
        return httpx.Response(status, json=body)

    client = Client(run="test", root=tmp_path)
    client.PAUSE_S, client.BACKOFF_S = 0.01, 0.001
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client, calls


def test_quota_errors_pause_and_then_succeed(tmp_path, monkeypatch):
    quota = (429, {"error": {"message": "you (someone) have reached your session usage limit"}})
    client, calls = client_with(tmp_path, monkeypatch, [quota] * 14 + [(200, OK)])
    reply = asyncio.run(client.chat("glm", [{"role": "user", "content": "hi"}]))
    assert reply.text == '{"ok": true}' and len(calls) == 15
    log = (tmp_path / "logs" / "quota.log").read_text()
    assert "pause_30m" in log and "someone" not in log  # account names are redacted


def test_server_errors_give_up_after_ten_attempts(tmp_path, monkeypatch):
    client, calls = client_with(tmp_path, monkeypatch, [(503, {"error": "busy"})])
    with pytest.raises(QuotaError):
        asyncio.run(client.chat("glm", [{"role": "user", "content": "hi"}]))
    assert len(calls) == 10


def test_bad_request_is_not_retried(tmp_path, monkeypatch):
    client, calls = client_with(tmp_path, monkeypatch, [(400, {"error": "bad temperature"})])
    with pytest.raises(RuntimeError):
        asyncio.run(client.chat("glm", [{"role": "user", "content": "hi"}]))
    assert len(calls) == 1


def test_cache_serves_reruns_and_never_stores_keys(tmp_path, monkeypatch):
    client, calls = client_with(tmp_path, monkeypatch, [(200, OK)])
    msgs = [{"role": "user", "content": "hi"}]
    first = asyncio.run(client.chat("glm", msgs))
    second = asyncio.run(client.chat("glm", msgs))
    assert len(calls) == 1 and not first.cached and second.cached
    assert calls[0].headers["authorization"] == "Bearer z-test"
    cached = next((tmp_path / "cache" / "zai").glob("*.json")).read_text()
    assert "z-test" not in cached and json.loads(cached)["request"]["thinking"] == {"type": "disabled"}
    other = asyncio.run(client.chat("glm", msgs, attempt=1))  # a retry is a fresh call
    assert len(calls) == 2 and not other.cached


def test_no_wait_mode_fails_fast_and_stops_calling(tmp_path, monkeypatch):
    from jevalt.data.providers import Unavailable

    quota = (429, {"error": {"message": "session usage limit"}})
    client, calls = client_with(tmp_path, monkeypatch, [quota])
    client.wait_on_quota = False
    with pytest.raises(Unavailable):
        asyncio.run(client.chat("ds", [{"role": "user", "content": "a"}]))
    with pytest.raises(Unavailable):
        asyncio.run(client.chat("ds", [{"role": "user", "content": "b"}]))
    assert len(calls) == 1  # the second call never left the process
    assert asyncio.run(client.available("ollama")) is False


def test_kimi_budget_cap_stops_calls(tmp_path, monkeypatch):
    from jevalt.data.providers import Unavailable

    client, calls = client_with(tmp_path, monkeypatch, [(200, OK)])

    async def meter():
        return {"week_used": "46"}

    client.kimi_usage = meter
    with pytest.raises(Unavailable):
        asyncio.run(client.chat("k3n", [{"role": "user", "content": "hi"}]))
    assert not calls  # nothing was sent once the plan hit the cap


def test_request_rate_limits_back_off_even_in_no_wait_mode(tmp_path, monkeypatch):
    rate = (429, {"error": {"code": "1302", "message": "Rate limit reached for requests"}})
    client, calls = client_with(tmp_path, monkeypatch, [rate, rate, (200, OK)])
    client.wait_on_quota = False
    monkeypatch.setattr(providers.random, "random", lambda: 0.0)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(providers.asyncio, "sleep", lambda s: real_sleep(0))
    reply = asyncio.run(client.chat("glm", [{"role": "user", "content": "hi"}]))
    assert reply.text == '{"ok": true}' and len(calls) == 3
