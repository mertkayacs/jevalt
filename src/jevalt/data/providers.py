"""Async client for the teacher models (all OpenAI-compatible ``/chat/completions``).

Per provider: at most ``concurrency`` requests in flight, exponential backoff with
jitter on 429/5xx/timeouts, a 30 minute pause after 3 consecutive quota errors.
Every successful raw response is cached under ``data/cache/<provider>/<sha>.json``
keyed by the request body (plus an attempt number), so reruns cost nothing.

Keys come from environment variables or the local credential files and are never
written to logs, caches or exceptions.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from .common import data_dir, extract_json, sha

PROVIDERS = {
    # Kimi Code rejects SDK user agents, a curl-like one works.
    "kimi": {"base": "https://api.kimi.com/coding/v1", "headers": {"User-Agent": "curl/8.5.0"}},
    "zai": {"base": "https://api.z.ai/api/coding/paas/v4", "headers": {}},
    "ollama": {"base": "https://ollama.com/v1", "headers": {}},
    # Hugging Face Inference Providers through the HF router (open-weight models, billed to the HF account).
    "hfr": {"base": "https://router.huggingface.co/v1", "headers": {}},
}
# GLM is the only open-weight lane with quota left (Ollama weekly limit, Kimi over its shared cap). Labels then pair
# GLM with the writer's own recorded distribution (a different lab) and both must agree; see label.pair_for.
GLM_ONLY = os.environ.get("JEVALT_GLM_ONLY") == "1"
# HF router lane (JEVALT_HFR=1): Gemma, Qwen and DeepSeek through router.huggingface.co, provider pinned per model.
HFR = os.environ.get("JEVALT_HFR") == "1"


@dataclass(frozen=True)
class Teacher:
    name: str
    provider: str
    model: str
    lab: str
    params: dict = field(default_factory=dict)
    temperature: float | None = None  # fixed by the provider when set
    thinks: bool = False  # reasoning tokens count against max_tokens

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.model}"


_NO_THINK = {"reasoning_effort": "none"}  # Ollama's switch for thinking models
TEACHERS = {
    t.name: t
    for t in [
        # Kimi Code only accepts temperature 1 with thinking and 0.6 without.
        Teacher("k3", "kimi", "k3", "moonshot", temperature=1.0, thinks=True),
        Teacher("k3n", "kimi", "k3", "moonshot", {"thinking": {"type": "disabled"}}, temperature=0.6),
        Teacher("k3o", "ollama", "kimi-k3", "moonshot", _NO_THINK),
        Teacher("k3ot", "ollama", "kimi-k3", "moonshot", thinks=True),  # thinking on (Ollama default)
        Teacher("glm", "zai", "glm-5.2", "zhipu", {"thinking": {"type": "disabled"}}),
        Teacher("glmt", "zai", "glm-5.2", "zhipu", {"thinking": {"type": "enabled"}}, thinks=True),
        Teacher("ds", "ollama", "deepseek-v4-pro:0813", "deepseek", _NO_THINK),
        Teacher("dst", "ollama", "deepseek-v4-pro:0813", "deepseek", thinks=True),
        Teacher("mistral", "ollama", "mistral-large-3:675b", "mistral"),
        Teacher("minimax", "ollama", "minimax-m3", "minimax", _NO_THINK),
        Teacher("gemma", "ollama", "gemma4:31b", "google"),
        # Router models, picked 2026-10-01 from a probe on 12 Turkish pilot rows (38 questions; every model matched the
        # three-teacher pilot gold on 32 or 33) and from speed under load: Gemma 4 26B-A4B on Novita (127 tok/s; the
        # 31B endpoints returned 503/504), Qwen3.5-122B-A10B (118 tok/s), Qwen3.5-35B-A3B (half the price, 23/27 vs
        # 22/27 on the German probe), DeepSeek V4.1 Flash (102 tok/s), DeepSeek V4 Pro for traces and Turkish edits.
        Teacher("ds-r", "hfr", "deepseek-ai/DeepSeek-V4-Pro-0813:deepinfra", "deepseek"),
        Teacher("dsf-r", "hfr", "deepseek-ai/DeepSeek-V4.1-Flash:deepinfra", "deepseek"),
        Teacher("gemma26-n", "hfr", "google/gemma-4-26B-A4B-it:novita", "google"),
        Teacher("qwen35-r", "hfr", "Qwen/Qwen3.5-122B-A10B:deepinfra", "alibaba", {"chat_template_kwargs": {"enable_thinking": False}}),
        Teacher("qwen35s-r", "hfr", "Qwen/Qwen3.5-35B-A3B:deepinfra", "alibaba", {"chat_template_kwargs": {"enable_thinking": False}}),
    ]
}

QUOTA_WORDS = re.compile(r"quota|rate.?limit|too many|exceed|insufficient|usage limit|capacity", re.I)
# A used-up plan window (session, weekly, credits), as opposed to a short request-rate limit.
EXHAUSTED = re.compile(r"usage limit|quota|insufficient|balance|exceeded your|limit exceeded|upgrade|billing", re.I)
SECRETISH = re.compile(r"(Bearer\s+|sk-|key[=:]\s*)\S+", re.I)
ACCOUNT = re.compile(r"\byou \([^)]*\)|\buser(name)?[=:]\s*\S+", re.I)  # providers echo the account name


def _load_keys(exits: bool = True) -> dict[str, str]:
    """API keys from environment variables only. If *exits* a missing or empty key
    for a required provider raises SystemExit with the variable name, so the
    pipeline never retries against an empty key."""
    keys = {
        "kimi": os.environ.get("KIMI_API_KEY", ""),
        "zai": os.environ.get("ZAI_API_KEY", ""),
        "ollama": os.environ.get("OLLAMA_API_KEY", ""),
        "hfr": os.environ.get("HF_TOKEN", ""),
    }
    required = [("zai", "ZAI_API_KEY"), ("ollama", "OLLAMA_API_KEY")] + ([("hfr", "HF_TOKEN")] if HFR else [])
    if exits:
        for provider, env_var in required:
            if not keys[provider]:
                raise SystemExit(f"${env_var} is empty or not set. Source the keys script before running the pipeline.")
    return keys


class QuotaError(RuntimeError):
    pass


class GiveUp(RuntimeError):
    """A teacher could not produce a usable reply within the allowed attempts."""


class Unavailable(GiveUp):
    """The provider is quota-limited and the client was told not to wait for it."""


@dataclass
class Reply:
    text: str
    model: str
    finish: str
    usage: dict
    cached: bool


class Client:
    PAUSE_S = 1800  # provider pause after 3 consecutive quota errors
    BACKOFF_S = 2.0  # first retry wait; doubles per attempt, capped at 120 s
    PROBE_S = 600  # how long an availability answer is trusted
    KIMI_CAP = 45  # percentage of the weekly Kimi quota this pipeline may consume
    KIMI_CHECK_EVERY = 5  # Kimi calls between meter reads
    PROBE_MODEL = {"ollama": "gemma4:31b", "zai": "glm-5.2", "kimi": "k3", "hfr": "google/gemma-4-26B-A4B-it:novita"}

    def __init__(self, *, concurrency: int = 4, run: str = "dev", root: Path | None = None, wait_on_quota: bool = True):
        self.root = root or data_dir()
        self.run = run
        # False: a quota-limited provider fails fast with Unavailable so callers can
        # switch teachers instead of idling through a plan's session window.
        self.wait_on_quota = wait_on_quota
        self._limited_until: dict[str, float] = defaultdict(float)
        self._probed: dict[str, tuple[float, bool]] = {}
        self._kimi_calls = 0
        self.kimi_week_used: float | None = None
        self._keys: dict[str, str] | None = None
        self._sems = {p: asyncio.Semaphore(min(concurrency, 4) if p == "zai" else 32 if p == "hfr" else concurrency) for p in PROVIDERS}
        self._paused_until: dict[str, float] = defaultdict(float)
        self._quota_streak: Counter = Counter()
        self.stats: dict[str, Counter] = defaultdict(Counter)
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(connect=30, read=900, write=60, pool=None))
        (self.root / "logs").mkdir(parents=True, exist_ok=True)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # ------------------------------------------------------------ logging

    def _log(self, name: str, record: dict) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run": self.run, **record}
        with open(self.root / "logs" / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _quota_log(self, provider: str, model: str, event: str, **info: Any) -> None:
        msg = ACCOUNT.sub("[account]", SECRETISH.sub(r"\1***", str(info.pop("msg", ""))))[:240]
        self._log("quota.log", {"provider": provider, "model": model, "event": event, **info, "msg": msg})

    # ------------------------------------------------------------ requests

    async def chat(
        self,
        teacher: str,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        attempt: int = 0,
        purpose: str = "",
        cache_only: bool = False,
    ) -> Reply:
        t = TEACHERS[teacher]
        body: dict[str, Any] = {"model": t.model, "messages": messages, "max_tokens": max_tokens, **t.params}
        body["temperature"] = t.temperature if t.temperature is not None else temperature
        key = sha({"provider": t.provider, "body": body, "attempt": attempt})
        path = self.root / "cache" / t.provider / f"{key}.json"
        if path.exists():
            self.stats[t.provider]["cache_hits"] += 1
            return self._reply(json.loads(path.read_text())["response"], cached=True)
        if cache_only:
            raise Unavailable(f"{t.id}: not in cache and cache_only was set")
        started = time.monotonic()
        response = await self._post(t, body)
        latency = round(time.monotonic() - started, 2)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"request": body, "attempt": attempt, "response": response, "latency_s": latency}, ensure_ascii=False))
        tmp.replace(path)
        usage = response.get("usage") or {}
        self.stats[t.provider]["calls"] += 1
        self.stats[t.provider]["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
        self.stats[t.provider]["completion_tokens"] += usage.get("completion_tokens", 0) or 0
        self._log(
            "calls.jsonl",
            {
                "provider": t.provider,
                "model": response.get("model", t.model),
                "teacher": teacher,
                "purpose": purpose,
                "latency_s": latency,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                **({"usd": usage["estimated_cost"]} if usage.get("estimated_cost") is not None else {}),
                "cache": key[:16],
            },
        )
        return self._reply(response, cached=False)

    @staticmethod
    def _reply(response: dict, *, cached: bool) -> Reply:
        choice = response["choices"][0]
        return Reply(
            text=choice["message"].get("content") or "",
            model=response.get("model", ""),
            finish=choice.get("finish_reason") or "",
            usage=response.get("usage") or {},
            cached=cached,
        )

    async def _post(self, t: Teacher, body: dict) -> dict:
        if self._keys is None:
            self._keys = _load_keys()
        spec = PROVIDERS[t.provider]
        headers = {"Authorization": f"Bearer {self._keys[t.provider]}", "Content-Type": "application/json", **spec["headers"]}
        url = spec["base"] + "/chat/completions"
        # Quota errors only pause the provider (a plan's session window can last hours);
        # other failures give up after 10 attempts.
        failures, quota_hits, rate_hits, deadline = 0, 0, 0, time.time() + 12 * 3600
        while True:
            attempt = failures + quota_hits
            if not self.wait_on_quota and self._limited_until[t.provider] > time.time():
                raise Unavailable(f"{t.id}: quota-limited")
            if t.provider == "kimi" and attempt == 0:
                await self._kimi_budget(t)
            await self._wait_if_paused(t)
            async with self._sems[t.provider]:
                try:
                    r = await self._http.post(url, json=body, headers=headers)
                    status, text = r.status_code, r.text
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    status, text = 0, type(exc).__name__
            if status == 200:
                try:
                    data = json.loads(text)
                    data["choices"][0]["message"]
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    status = -1  # malformed 200, retry like a server error
                else:
                    self._quota_streak[t.provider] = 0
                    return data
            limited = status == 429 or (status in (402, 403) and QUOTA_WORDS.search(text) is not None)
            quota = limited and EXHAUSTED.search(text) is not None
            if limited and not quota:  # request-rate limit: back off and retry, the plan is fine
                rate_hits += 1
                if rate_hits > 40:
                    raise QuotaError(f"{t.id}: still rate-limited after 40 retries")
                wait = min(120.0, max(5.0, self.BACKOFF_S * 2 ** min(rate_hits, 5))) * (0.5 + random.random())
                self._quota_log(t.provider, t.model, "rate_limited", status=status, attempt=rate_hits, wait_s=round(wait, 1), msg=text)
                await asyncio.sleep(wait)
                continue
            if status in (400, 401, 404, 422) or (status in (402, 403) and not limited):
                self.stats[t.provider]["errors"] += 1
                self._quota_log(t.provider, t.model, "error", status=status, msg=text)
                raise RuntimeError(f"{t.id} rejected the request with HTTP {status}: {SECRETISH.sub('***', text)[:300]}")
            wait = min(120.0, self.BACKOFF_S * 2 ** min(attempt, 6)) * (0.5 + random.random())
            if quota and not self.wait_on_quota:
                self._limited_until[t.provider] = time.time() + self.PROBE_S
                self._quota_log(t.provider, t.model, "unavailable", status=status, msg=text)
                raise Unavailable(f"{t.id}: quota-limited")
            if quota:
                if time.time() > deadline:
                    raise QuotaError(f"{t.id}: quota still exhausted after 12 hours")
                quota_hits += 1
                self._quota_streak[t.provider] += 1
                if self._quota_streak[t.provider] >= 3:
                    self._paused_until[t.provider] = time.time() + self.PAUSE_S
                    self._quota_streak[t.provider] = 0
                    self._quota_log(t.provider, t.model, "pause_30m", status=status, msg=text)
                    continue
            else:
                failures += 1
                if failures >= 10:
                    raise QuotaError(f"{t.id}: gave up after repeated failures")
            self.stats[t.provider]["retries"] += 1
            self._quota_log(t.provider, t.model, "retry", status=status, attempt=attempt, wait_s=round(wait, 1), msg=text)
            await asyncio.sleep(wait)

    async def _wait_if_paused(self, t: Teacher) -> None:
        while (left := self._paused_until[t.provider] - time.time()) > 0:
            await asyncio.sleep(min(left, 60, self.PAUSE_S))
        if self._paused_until[t.provider]:
            self._paused_until[t.provider] = 0
            self._quota_log(t.provider, t.model, "resume")

    async def _kimi_budget(self, t: Teacher) -> None:
        """Return the provider's quota usage if measurable, else None.
        The Kimi provider has a weekly budget cap to keep other consumers working."""
        if self._kimi_calls % self.KIMI_CHECK_EVERY == 0:
            try:
                self.kimi_week_used = float((await self.kimi_usage()).get("week_used") or 0)
            except (httpx.HTTPError, ValueError):
                pass
        self._kimi_calls += 1
        if self.kimi_week_used is not None and self.kimi_week_used >= self.KIMI_CAP:
            self._limited_until["kimi"] = time.time() + 7 * 24 * 3600
            self._quota_log(t.provider, t.model, "budget_cap", status=f"{self.kimi_week_used:.0f}%")
            raise Unavailable(f"{t.id}: shared weekly budget cap reached ({self.kimi_week_used:.0f}%)")

    def usable(self, provider: str) -> bool:
        """What this client already knows, without a network probe."""
        if self._limited_until[provider] > time.time():
            return False
        return not (provider == "kimi" and self.kimi_week_used is not None and self.kimi_week_used >= self.KIMI_CAP)

    async def available(self, provider: str) -> bool:
        """Whether a provider answers right now (a 5-token probe, trusted for PROBE_S)."""
        now = time.time()
        if self._limited_until[provider] > now:
            return False
        seen = self._probed.get(provider)
        if seen and now - seen[0] < self.PROBE_S:
            return seen[1]
        if self._keys is None:
            self._keys = _load_keys()
        spec = PROVIDERS[provider]
        body = {"model": self.PROBE_MODEL[provider], "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5}
        if provider == "kimi":
            body["temperature"] = 1.0
        try:
            r = await self._http.post(spec["base"] + "/chat/completions", json=body,
                                      headers={"Authorization": f"Bearer {self._keys[provider]}", **spec["headers"]})
            ok = r.status_code == 200
        except httpx.HTTPError:
            ok = False
        self._probed[provider] = (now, ok)
        if not ok:
            self._limited_until[provider] = now + self.PROBE_S
        self._quota_log(provider, self.PROBE_MODEL[provider], "probe", status=200 if ok else "limited")
        return ok

    async def ask_json(
        self,
        teacher: str,
        messages: list[dict],
        *,
        check: Callable[[Any], Any] | None = None,
        parse: Callable[[str], Any] = extract_json,
        tries: int = 3,
        allow_truncated: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, Reply]:
        """Ask for JSON; on a bad or unusable reply ask again, telling the model what was wrong.

        ``check`` raises ValueError to reject a reply, or returns a cleaned value.
        """
        error = None
        for attempt in range(tries):
            msgs = messages
            if error:
                msgs = messages + [{"role": "user", "content": f"Your previous reply could not be used: {error}\nReply again with the complete, corrected JSON only."}]
            reply = await self.chat(teacher, msgs, attempt=attempt, **kwargs)
            try:
                if reply.finish == "length" and not allow_truncated:
                    raise ValueError("the reply was cut off before the end")
                data = parse(reply.text)
                if check is not None:
                    checked = check(data)
                    data = data if checked is None else checked
                return data, reply
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                error = str(exc)[:300] or type(exc).__name__
        raise GiveUp(f"{teacher}: {error}")

    # ------------------------------------------------------------ quota

    async def kimi_usage(self) -> dict:
        """Kimi Code usage (percent of the 5 hour and weekly windows). Only numbers are kept."""
        if self._keys is None:
            self._keys = _load_keys()
        spec = PROVIDERS["kimi"]
        r = await self._http.get(spec["base"] + "/usages", headers={"Authorization": f"Bearer {self._keys['kimi']}", **spec["headers"]})
        r.raise_for_status()
        data = r.json()
        usage = data.get("usages", {})
        week = data.get("usage", {})
        return {
            "limit_5h_used_ratio": usage.get("limit_5h", {}).get("used_ratio"),
            "limit_5h_reset": usage.get("limit_5h", {}).get("reset_time"),
            "limit_7d_used_ratio": usage.get("limit_7d", {}).get("used_ratio"),
            "limit_7d_reset": usage.get("limit_7d", {}).get("reset_time"),
            "week_used": week.get("used"),
            "week_limit": week.get("limit"),
        }
