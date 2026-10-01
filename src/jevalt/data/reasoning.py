"""R: short reasoning traces verified against gold (DATA.md section 6).

A teacher that did not label the row writes a trace in the row language
(at most 90 words) plus its own answers without seeing the gold. The trace is
kept only when every answer matches the gold argmax; otherwise one fresh
attempt, then the row keeps no trace.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from . import prompts
from .common import field_values, has_tag, words
from .providers import GLM_ONLY, HFR, TEACHERS, Client, GiveUp
from .validate import detect, folded

MAX_WORDS = 90
SELF_TALK = re.compile(
    r"\bas an ai\b|\bi am an ai\b|\blanguage model\b|\bprobabilit(y|ies)\b|\bteacher model\b|\bannotat"
    r"|yapay zek[aâ] olarak|\bdil modeli|\bolasılı[kğ]|\bmodel olarak"
    r"|\bals ki\b|\bsprachmodell|\bwahrscheinlichkeit(en)?\b",
    re.IGNORECASE,
)
SYMBOL = re.compile(r"`|\b(option|answer|choice|seçenek|şık|cevap|antwort|auswahl)\s+[A-Za-z0-9]\b(?!\w)", re.IGNORECASE)


def trace_problems(trace: str, lang: str) -> list[str]:
    problems = []
    if not trace or not trace.strip():
        return ["empty trace"]
    if words(trace) > MAX_WORDS:
        problems.append(f"{words(trace)} words")
    if SELF_TALK.search(trace):
        problems.append("talks about itself or probabilities")
    if SYMBOL.search(trace):
        problems.append("mentions an option symbol")
    if re.search(r"^\s*([-*•]|\d+[.)])\s", trace, re.MULTILINE) or "#" in trace:
        problems.append("list or markdown")
    top, prob = detect(trace)
    ok_langs = {"de": {"de", "als", "bar"}}.get(lang, {lang})
    if top not in ok_langs or folded(trace, lang):
        problems.append(f"language {top} ({prob:.2f})")
    return problems


def candidates(row: Mapping) -> list[str]:
    """Trace writers allowed for a row, best first. K3 writes Turkish and German traces
    only; the writer's lab never labeled the row (or wrote its template, for programmatic
    rows)."""
    if row.get("source", "").startswith("prog:"):
        labs = {lab_of(has_tag(row, "writer") or "")}
    else:
        labs = {TEACHERS[t].lab for t in row.get("teachers", {}) if t in TEACHERS}
    if HFR:
        # DeepSeek V4 Pro: 4.50 (TR) and 4.65 (DE) in the native-writer bake-offs; GLM thinking as the fallback.
        return [t for t in ("ds-r", "glmt") if TEACHERS[t].lab not in labs]
    if GLM_ONLY:
        # Only GLM has quota. The usual rule (a lab that did not label the row) cannot hold; every trace is still
        # accepted only when its conclusion matches the gold answer.
        return ["glmt"]
    # K3 last: on the Kimi plan a trace costs about as much quota as a generation call.
    if row["lang"] == "tr":
        order = ["ds", "mistral", "glmt", "k3", "minimax"]
    else:
        order = ["glmt", "dst", *(["k3"] if row["lang"] in {"tr", "de"} else []), "mistral", "minimax"]
    return [t for t in order if TEACHERS[t].lab not in labs]


def reasoner_for(row: Mapping, up: set[str]) -> str | None:
    """First allowed trace writer whose provider is in ``up``."""
    return next((t for t in candidates(row) if TEACHERS[t].provider in up), None)


def lab_of(model_id: str) -> str:
    """Lab behind a recorded model id such as ``zai/glm-5.3`` or ``ollama/kimi-k3``."""
    model_id = model_id.lower()
    for key, lab in (("deepseek", "deepseek"), ("glm", "zhipu"), ("kimi", "moonshot"), ("k3", "moonshot"), ("mistral", "mistral"), ("minimax", "minimax"), ("gemma", "google"), ("qwen", "alibaba")):
        if key in model_id:
            return lab
    return ""


async def trace_row(client: Client, row: Mapping, teacher: str) -> tuple[dict | None, dict]:
    """(trace record, attempt log)."""
    values = field_values(row)
    gold = {f: t["label"] for f, t in row["targets"].items()}
    log: dict[str, Any] = {"teacher": teacher, "attempts": []}
    for attempt in range(2):
        messages = prompts.reason_messages(row["lang"], row, MAX_WORDS)
        if attempt:  # a fresh sample rather than the cached first reply
            messages[-1]["content"] += "\n\nThink it through again carefully before answering."

        def check(data: Any) -> dict:
            if not isinstance(data, Mapping) or not isinstance(data.get("trace"), str) or not isinstance(data.get("answers"), Mapping):
                raise ValueError('expected {"trace": "...", "answers": {...}}')
            return data

        try:
            data, reply = await client.ask_json(teacher, messages, check=check, temperature=0.3, max_tokens=16000, purpose="reason")
        except GiveUp as exc:
            log["attempts"].append({"error": str(exc)[:200]})
            continue
        answers = {f: _as_option(data["answers"].get(f, ""), values[f]) for f in gold}
        wrong = {f: a for f, a in answers.items() if a != gold[f]}
        problems = trace_problems(data["trace"], row["lang"])
        log["attempts"].append({"wrong": wrong, "problems": problems, "trace": data["trace"]})
        if not wrong and problems and all(p.endswith(" words") for p in problems):
            # Right answers, too long: the same model shortens its own trace once.
            follow = messages + [{"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
                                 {"role": "user", "content": f"Shorten the trace to at most {MAX_WORDS * 3 // 4} words. Keep the facts, the rule and every conclusion. Same JSON format."}]
            try:
                short, reply = await client.ask_json(teacher, follow, check=check, temperature=0.3, max_tokens=16000, purpose="reason")
                problems = trace_problems(short["trace"], row["lang"])
                log["attempts"].append({"shortened": True, "problems": problems, "trace": short["trace"]})
                data = short
            except GiveUp as exc:
                log["attempts"].append({"error": str(exc)[:200]})
        if not wrong and not problems:
            return {"trace": data["trace"].strip(), "model": f"{TEACHERS[teacher].provider}/{reply.model}"}, log
    return None, log


def _as_option(answer: Any, values: Sequence[str]) -> str:
    """The option a free-form answer names (case and backticks ignored), else the raw text."""
    text = str(answer).strip().strip("`").strip()
    return next((v for v in values if v.casefold() == text.casefold()), text)


def pick_rows(rows_by_set: Mapping[str, Sequence[Mapping]], plan: Mapping[str, int], up: set[str]) -> list[tuple[Mapping, str]]:
    """(row, trace writer) pairs: G rows where teachers disagreed at least once, then the
    programmatic sets, ``plan[set]`` each. Rows nobody may trace right now are skipped and
    the total is filled from the programmatic sets."""
    def eligible(name: str) -> list[tuple[Mapping, str]]:
        pool = list(rows_by_set.get(name, []))
        if name == "g":
            pool = [r for r in pool if any(len({max(r["teachers"][t]["dist"][f], key=r["teachers"][t]["dist"][f].get) for t in r["teachers"]}) > 1 for f in r["targets"])]
        return [(r, w) for r in pool if (w := reasoner_for(r, up))]

    picked, taken = [], set()
    for name, n in plan.items():
        for r, w in eligible(name)[:n]:
            picked.append((r, w))
            taken.add(r["id"])
    total = sum(plan.values())
    for name in plan:
        if name == "g":
            continue
        for r, w in eligible(name):
            if len(picked) >= total:
                break
            if r["id"] not in taken:
                picked.append((r, w))
                taken.add(r["id"])
    return picked


async def add_traces(client: Client, picked: Sequence[tuple[Mapping, str]]) -> tuple[dict[str, dict], list[dict]]:
    """{row_id: trace record} for accepted traces, plus a log line per row."""
    from .common import bounded

    async def one(row: Mapping, writer: str):
        return row, *(await trace_row(client, row, writer))

    accepted, logs = {}, []
    async for row, rec, log in bounded([one(row, w) for row, w in picked], limit=12):
        logs.append({"id": row["id"], **log, "accepted": rec is not None})
        if rec:
            accepted[row["id"]] = rec
    return accepted, logs
