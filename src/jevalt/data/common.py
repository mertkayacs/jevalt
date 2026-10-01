"""Shared helpers: paths, JSONL streaming, hashing, seeded RNG and distributions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
from collections.abc import AsyncIterator, Awaitable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from jevalt.format import compile_request

LANGS = ("en", "tr", "de")
LANG_NAMES = {"en": "English", "tr": "Turkish", "de": "German"}
SMOOTH = 0.02
# License for rows written by teacher models. Open project decision, kept in one place.
GEN_LICENSE = "apache-2.0"


def data_dir() -> Path:
    return Path(os.environ.get("JEVALT_DATA", "./data")).expanduser().resolve()


def sha(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def rng(*parts: Any) -> random.Random:
    """Deterministic RNG per purpose, so reruns rebuild the same rows."""
    return random.Random(int(sha(parts)[:16], 16))


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


async def bounded(coros: Iterable[Awaitable], limit: int = 32) -> AsyncIterator[Any]:
    """Run awaitables with at most ``limit`` in flight, yielding results as they finish."""
    pending: set[asyncio.Task] = set()
    for coro in coros:
        pending.add(asyncio.ensure_future(coro))
        if len(pending) >= limit:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                yield task.result()
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            yield task.result()


# ---------------------------------------------------------------- rows


def has_tag(row: Mapping, prefix: str) -> str | None:
    """Value of the first tag ``prefix:value`` (or the bare tag itself), else None."""
    for tag in row.get("tags", ()):
        if tag == prefix:
            return tag
        if tag.startswith(prefix + ":"):
            return tag.split(":", 1)[1]
    return None


def compiled(row: Mapping):
    """Compile a row exactly as training will: rows tagged ``abstain`` get the unknown option."""
    request = {"state": row["state"], "questions": row["questions"]}
    return compile_request(request, abstain=has_tag(row, "abstain") is not None, language=row.get("lang", "en"))


def field_values(row: Mapping) -> dict[str, tuple[str, ...]]:
    """Option values per field in prompt order, the keys a target ``dist`` must use."""
    return {f.name: f.values for f in compiled(row).fields}


def question_values(question: Mapping, *, abstain: bool = False, lang: str = "en") -> tuple[str, ...]:
    fields = compile_request({"state": "", "questions": {"q": question}}, abstain=abstain, language=lang).fields
    return fields[0].values


# ---------------------------------------------------------------- distributions


def normalize(dist: Mapping[str, float]) -> dict[str, float]:
    total = sum(dist.values())
    if total <= 0:
        raise ValueError("distribution has no mass")
    return {k: v / total for k, v in dist.items()}


def smooth(dist: Mapping[str, float], eps: float = SMOOTH) -> dict[str, float]:
    n = len(dist)
    return {k: (1 - eps) * v + eps / n for k, v in dist.items()}


def one_hot(label: str, values: Iterable[str], eps: float = SMOOTH) -> dict[str, float]:
    return smooth({v: float(v == label) for v in values}, eps)


def argmax(dist: Mapping[str, float]) -> str:
    return max(dist, key=lambda k: dist[k])


def argmax_set(dist: Mapping[str, float], tol: float = 1e-9) -> set[str]:
    top = max(dist.values())
    return {k for k, v in dist.items() if v >= top - tol}


def target(dist: Mapping[str, float]) -> dict:
    return {"label": argmax(dist), "dist": {k: round(v, 6) for k, v in dist.items()}}


# ---------------------------------------------------------------- teacher replies


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> Any:
    """Parse the JSON value in a model reply, tolerating fences and chatter around it."""
    text = (text or "").strip()
    try:
        return json.loads(_FENCE.sub("", text))
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON value in reply")
    decoder = json.JSONDecoder()
    start = min(starts)
    while start >= 0:
        try:
            value, _ = decoder.raw_decode(text[start:])
            return value
        except json.JSONDecodeError:
            nxt = [i for i in (text.find("{", start + 1), text.find("[", start + 1)) if i >= 0]
            start = min(nxt) if nxt else -1
    raise ValueError("reply is not valid JSON")


def text_of(value: Any) -> str:
    """Readable text of a state or instruction value, for prompts and reports."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def words(value: Any) -> int:
    return len(text_of(value).split())
