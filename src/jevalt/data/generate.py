"""G: scenario rows written natively by teacher models from diversity cards,
then validated and (TR, DE) passed through a native edit by a different model."""

from __future__ import annotations

import os

import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from . import prompts
from .common import GEN_LICENSE, argmax, bounded, rng, sha, text_of, words
from .providers import HFR, TEACHERS, Client, GiveUp, Unavailable
from .validate import check_question, check_row, language_check, leaks, scrub

# K3 runs on Ollama to avoid consuming the Kimi Code weekly meter.
WRITERS = {"en": ["glm"], "tr": ["mistral", "mistral", "mistral", "ds", "ds"], "de": ["ds", "ds", "ds", "gemma", "gemma"]}
# Editors are from a different lab then the writer where possible.
EDITORS = {"tr": ["glm", "k3"], "de": ["mistral"]}
# When the configured writer's provider is quota-limited, fall back to GLM on Z.ai.
FALLBACK_WRITER = {"en": "glmt", "tr": "glmt", "de": "glmt"}
if HFR:
    # German writer bake-off: DeepSeek V4 Pro 4.65 and Gemma 4 31B 4.54 (CIs overlap). Gemma 4 26B-A4B (the 31B
    # endpoints were overloaded) and DeepSeek V4.1 Flash write at a fraction of V4 Pro's price per row.
    WRITERS["de"] = ["gemma26-n", "dsf-r"]
    EDITORS["de"] = ["ds-r", "gemma26-n"]
    FALLBACK_WRITER["de"] = "dsf-r"
ROWS_PER_CALL = 8
# Router replies over a few minutes hit the gateway timeout, so languages written on the router get smaller batches.
# Turkish keeps 8: its rows come from the cache of the original 8-row batches.
ROWS_PER_CALL_BY_LANG = {"de": 4} if HFR else {}
WRITER_CALLS = 32 if HFR else 8  # writer calls in flight

FORMATS = ["plain text message", "email with headers", "chat log with speakers and times", "JSON record", "log lines", "table-like text"]
REGISTERS = ["formal", "casual", "angry", "terse", "verbose"]
REGIONS = {
    "en": [("US English", 0.6), ("UK English", 0.3), ("international English written by a fluent non-native", 0.1)],
    "tr": [("Istanbul, colloquial", 0.3), ("Anatolian, with regional expressions", 0.2), ("formal corporate Turkish (kurumsal)", 0.5)],
    "de": [("Hochdeutsch", 0.8), ("Swiss spelling and vocabulary", 0.1), ("Austrian spelling and vocabulary", 0.1)],
}
COMPLICATIONS = [
    "the deciding fact appears once, in the middle, next to a similar but irrelevant fact",
    "an earlier statement is corrected later in the state",
    "two people in the state disagree",
    "the main case is mixed with an unrelated side topic",
    "a key date or amount is given indirectly (relative date, sum of parts)",
    "the author is not the person affected",
    "the tone suggests one answer but the facts support another",
    "the state quotes an older message",
    "some fields are empty or marked as not provided",
    "numbers must be compared or added to decide",
]
NPC_QUESTIONS = [
    ("next_action", "choice"),
    ("should_share_food", "noul"),
    ("trust_in_player", "score"),
    ("urgency", "score"),
    ("mood", "score"),
    ("accept_quest", "noul"),
    ("who_to_ask_for_help", "choice"),
]


def _pick(r, weighted: Sequence[tuple[Any, float]]) -> Any:
    return r.choices([v for v, _ in weighted], [w for _, w in weighted])[0]


def _length(r) -> int:
    lo, hi = _pick(r, [((40, 150), 0.5), ((150, 400), 0.3), ((400, 800), 0.15), ((800, 1500), 0.05)])
    return int(round(r.randint(lo, hi), -1))


def _question(r, kind: str, field: str | None = None) -> dict:
    q: dict[str, Any] = {"type": kind, "phrasing": _pick(r, [("direct", 0.5), ("criteria-heavy", 0.3), ("structured", 0.2)])}
    if field:
        q["field"] = field
    ambiguous = r.random() < 0.2
    if kind == "choice":
        span = _pick(r, [("small", 0.82), ((9, 30), 0.15), ((31, 60), 0.03)])
        if span == "small":
            n = _pick(r, [(2, 0.1), (3, 0.25), (4, 0.25), (5, 0.15), (6, 0.1), (7, 0.08), (8, 0.07)])
        else:
            n = r.randint(*span)
        q["options"] = n
        q["target"] = "ambiguous" if ambiguous else f"position {r.randint(1, n)}"
    elif kind == "score":
        q["levels"] = n = _pick(r, [(2, 0.05), (3, 0.3), (4, 0.25), (5, 0.25), (r.randint(6, 10), 0.15)])
        q["target"] = "ambiguous" if ambiguous else f"level {r.randint(0, n - 1)}"
    else:
        q["target"] = "ambiguous" if ambiguous else f"answer: {r.choice(['yes', 'no'])}"
    return q


def make_card(lang: str, domain: str, index: int, seed: str) -> dict:
    r = rng("card", lang, domain, index, seed)
    region = _pick(r, REGIONS[lang])
    register = r.choice(REGISTERS)
    if lang == "de":
        region += "; address the reader with " + ("Sie" if register in {"formal", "verbose"} or r.random() < 0.4 else "du")
    if r.random() < 0.1:
        register += ", with a few realistic typos"
    card: dict[str, Any] = {
        "domain": domain,
        "topic": r.choice(prompts.DOMAINS[domain][1]),
        "length": _length(r),
        "register": register,
        "region": region,
        "complication": r.choice(COMPLICATIONS) if r.random() < 0.6 else "",
    }
    n = _pick(r, [(2, 0.2), (3, 0.3), (4, 0.25), (5, 0.15), (6, 0.1)])
    if domain == "game_npc":
        card["format"] = "JSON record" if r.random() < 0.85 else "table-like text"
        picked = r.sample(NPC_QUESTIONS[1:], n - 1) if r.random() < 0.85 else r.sample(NPC_QUESTIONS[1:], n)
        if len(picked) < n:
            picked = [NPC_QUESTIONS[0]] + picked
        card["questions"] = [_question(r, kind, field) for field, kind in picked]
        for q in card["questions"]:
            if q["field"] == "next_action":
                q["options"] = r.randint(3, 10)
            elif q["field"] == "who_to_ask_for_help":
                q["options"] = r.randint(3, 6)
            elif q["type"] == "score":
                q["levels"] = r.randint(3, 5)
            if q["type"] != "noul" and not q["target"].startswith("ambiguous"):
                q["target"] = f"position {r.randint(1, q['options'])}" if q["type"] == "choice" else f"level {r.randint(0, q['levels'] - 1)}"
            if q["field"] == "accept_quest":
                q["target"] += " (the state must contain the player's quest offer)"
    else:
        card["format"] = r.choice(FORMATS)
        kinds = [_pick(r, [("choice", 0.45), ("noul", 0.30), ("score", 0.25)]) for _ in range(n)]
        card["questions"] = [_question(r, k) for k in kinds]
    return card


def plan_cards(lang: str, n_rows: int, *, npc_share: float = 0.25, seed: str = "pilot", names: Mapping | None = None) -> list[dict]:
    """Cards for ``n_rows`` rows: a share of game_npc, other domains round-robin.
    ``names`` is a teacher-written pool; each card offers a few names so rows do not
    all reuse the same handful."""
    r = rng("plan", lang, seed)
    n_npc = math.ceil(n_rows * npc_share)
    others = [d for d in prompts.DOMAINS if d != "game_npc"]
    r.shuffle(others)
    domains = ["game_npc"] * n_npc + [others[i % len(others)] for i in range(n_rows - n_npc)]
    r.shuffle(domains)
    cards = [make_card(lang, d, i, seed) for i, d in enumerate(domains)]
    for card in cards:
        if names:
            if card["domain"] == "game_npc":
                card["names"] = r.sample(names["villagers"], min(6, len(names["villagers"])))
            else:
                card["names"] = r.sample(names["people"], min(3, len(names["people"]))) + r.sample(names["orgs"], min(2, len(names["orgs"])))
    return cards


CARD_MARK = re.compile(r"^\s*#{2,}\s*card\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)


def split_blocks(text: str) -> dict[int, Any]:
    """``### card N`` blocks -> parsed JSON per card (or the parse error).

    One JSON object per row means a single bracket slip in a long reply costs
    one row instead of the whole batch.
    """
    parts = CARD_MARK.split(text or "")
    out: dict[int, Any] = {}
    for i in range(1, len(parts) - 1, 2):
        try:
            out[int(parts[i])] = unnest_writer_dist(first_object(parts[i + 1]))
        except ValueError as exc:
            out[int(parts[i])] = exc
    return out


def first_object(chunk: str) -> Any:
    """The JSON object that starts at the first brace. Never falls back to an inner
    object (a broken row must not turn into its own questions dict)."""
    start = chunk.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    text = chunk[start:]
    decoder = json.JSONDecoder()
    try:
        return decoder.raw_decode(text)[0]
    except json.JSONDecodeError:
        pass
    try:  # the usual slip: a closing brace missing, so the object never ends
        return json.loads(close_brackets(text.rstrip().removesuffix("```").rstrip()))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from None


def close_brackets(text: str) -> str:
    """Append the closers for brackets still open at the end (strings respected)."""
    stack: list[str] = []
    in_str = esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    return text + "".join(reversed(stack))


def unnest_writer_dist(value: Any) -> Any:
    """With a brace missing, writer_dist ends up as a question; move it back out."""
    if isinstance(value, Mapping) and "writer_dist" not in value and isinstance(value.get("questions"), Mapping):
        stray = value["questions"].get("writer_dist")
        if isinstance(stray, Mapping) and "type" not in stray:
            value = dict(value)
            value["questions"] = {k: v for k, v in value["questions"].items() if k != "writer_dist"}
            value["writer_dist"] = stray
    return value


def _check_blocks(n_cards: int):
    def check(blocks: dict[int, Any]) -> dict[int, Any]:
        good = {n: b for n, b in blocks.items() if isinstance(b, Mapping) and "state" in b and isinstance(b.get("questions"), Mapping)}
        if len(good) < max(1, math.ceil(n_cards / 2)):
            raise ValueError(f'expected {n_cards} blocks, each "### card <n>" followed by a JSON object with state and questions; got {len(good)} usable')
        return blocks

    return check


def _tags(card: Mapping, writer_id: str, questions: Mapping) -> list[str]:
    kinds = {q.get("type") if isinstance(q, Mapping) else None for q in questions.values()}  # malformed rows fail validation later
    length = "short" if card["length"] < 150 else "medium" if card["length"] < 400 else "long"
    fmt = re.sub(r"[^a-z]+", "_", card["format"].split(" with")[0].lower()).strip("_")
    return [
        f"domain:{card['domain']}",
        f"type:{kinds.pop() if len(kinds) == 1 else 'mixed'}",
        f"fmt:{fmt}",
        f"len:{length}",
        f"writer:{writer_id}",
    ]


async def write_batch(client: Client, lang: str, writer: str, cards: list[dict], batch: int, *, cache_only: bool = False) -> tuple[list[dict], list[dict]]:
    numbered = [{**c, "card": i + 1} for i, c in enumerate(cards)]
    messages = prompts.gen_messages(lang, numbered)
    try:
        blocks, reply = await client.ask_json(
            writer, messages, parse=split_blocks, check=_check_blocks(len(cards)), allow_truncated=True,
            temperature=0.9, max_tokens=32000, purpose=f"gen:{lang}", cache_only=cache_only
        )
    except GiveUp as exc:
        return [], [{"stage": "gen", "reason": "writer_failed", "detail": str(exc), "batch": batch}]
    writer_id = f"{TEACHERS[writer].provider}/{reply.model or TEACHERS[writer].model}"
    out, drops = [], []
    for idx in range(1, len(cards) + 1):
        raw = blocks.get(idx)
        if not (isinstance(raw, Mapping) and "state" in raw and isinstance(raw.get("questions"), Mapping)):
            reason = "row_json_invalid" if raw is not None else "writer_skipped_card"
            drops.append({"stage": "gen", "reason": reason, "detail": str(raw)[:200] if raw is not None else "", "batch": batch, "card": idx})
            continue
        card = cards[idx - 1]
        row = {
            "lang": lang,
            "source": f"gen:{writer}",
            "license": GEN_LICENSE,
            "split": "train",
            "state": raw["state"],
            "questions": raw["questions"],
        }
        row["id"] = f"{lang}-g-{sha([lang, row['state'], row['questions']])[:10]}"
        row["tags"] = _tags(card, writer_id, raw["questions"]) + [f"group:{row['id']}"]
        row["writer"] = {"model": writer_id, "dist": raw.get("writer_dist") if isinstance(raw.get("writer_dist"), Mapping) else {}}
        row["card"] = card
        out.append({"id": row["id"], **{k: v for k, v in row.items() if k != "id"}})
    return out, drops


# Same model without thinking: used when a thinking writer spends its whole token
# budget reasoning and returns no usable rows.
NO_THINK = {"glmt": "glm", "k3ot": "k3o", "dst": "ds", "k3": "k3n"}
RETRY_CHUNK = 4
TOPUP_WRITER = {"en": "glmt", "tr": "glmt", "de": "dsf-r" if HFR else "glmt"}
# For TR/DE, when an Ollama writer returns incomplete batches, try the
# other Ollama writer of that language before falling back to GLM.
OLLAMA_SIBLING = {"mistral": "ds", "ds": "mistral", "gemma": "ds", "k3o": "ds"}
OLLAMA_WRITERS_FROM_CACHE = False
# Stop at what is already written: every write is served from the cache, unwritten cards are dropped.
CACHE_ONLY_WRITES = os.environ.get("JEVALT_GEN_CACHE_ONLY") == "1"


async def write_cards(client: Client, lang: str, writer: str, cards: list[dict], batch: int) -> tuple[list[dict], list[dict]]:
    """Write a batch; re-request missing cards once in chunks of at most four, and
    fall back to the no-thinking sibling for chunks that still return nothing.

    Ollama writers are used from cache only: re-generation from Ollama is not
    guaranteed to be available after the initial write. Missing cards go to the
    language's fallback writer, so reruns hit the cache."""
    on_ollama = TEACHERS[writer].provider == "ollama"
    rows, drops = await write_batch(client, lang, writer, cards, batch, cache_only=CACHE_ONLY_WRITES or (on_ollama and OLLAMA_WRITERS_FROM_CACHE))
    missing = [cards[d["card"] - 1] for d in drops if d.get("card")]
    if any(d["reason"] == "writer_failed" for d in drops):
        missing = list(cards)
    if not missing:
        return rows, drops
    drops = [dict(d, recovered_later=True) for d in drops]
    # For Ollama writers: try the sibling first, then GLM.
    sibling = OLLAMA_SIBLING.get(writer) if on_ollama else None
    if on_ollama:
        retries = [w for w in (sibling, FALLBACK_WRITER[lang]) if w]
    else:  # a router writer's missing cards go to the language's fallback writer (Gemma on Novita slowed to minutes per call)
        retries = [FALLBACK_WRITER[lang]] if HFR and FALLBACK_WRITER[lang] != writer else [writer]
    for retry in retries:
        if not missing:
            break
        for i in range(0, len(missing), RETRY_CHUNK):
            chunk = missing[i : i + RETRY_CHUNK]
            got, lost = await write_batch(client, lang, retry, chunk, batch, cache_only=CACHE_ONLY_WRITES)
            if not got and retry in NO_THINK:
                got, lost = await write_batch(client, lang, NO_THINK[retry], chunk, batch, cache_only=CACHE_ONLY_WRITES)
            rows += got
            drops += lost
    return rows, drops


async def generate(client: Client, lang: str, cards: list[dict], *, rows_per_call: int | None = None) -> tuple[list[dict], list[dict]]:
    rows_per_call = rows_per_call or ROWS_PER_CALL_BY_LANG.get(lang, ROWS_PER_CALL)
    batches = [cards[i : i + rows_per_call] for i in range(0, len(cards), rows_per_call)]
    writers = WRITERS[lang]
    rows, drops = [], []
    coros = [write_cards(client, lang, writers[b % len(writers)], batch, b) for b, batch in enumerate(batches)]
    async for got, dropped in bounded(coros, limit=WRITER_CALLS):
        rows += got
        drops += dropped
    rows.sort(key=lambda r: r["id"])
    return rows, drops


# ---------------------------------------------------------------- validation


def clean_writer_dist(row: Mapping) -> dict:
    """Writer distributions renormalised over the real option values; broken ones dropped."""
    from .common import field_values

    values = field_values(row)
    out = {}
    for name, dist in (row.get("writer", {}).get("dist") or {}).items():
        if name not in values or not isinstance(dist, Mapping):
            continue
        clean = {v: float(dist.get(v, 0) or 0) for v in values[name] if isinstance(dist.get(v, 0), (int, float))}
        total = sum(clean.values())
        if total > 0 and set(clean) == set(values[name]):
            out[name] = {k: v / total for k, v in clean.items()}
    return out


FOLD = {
    "de": {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"},
    "tr": {"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u", "Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U"},
}


def ascii_ident(name: str, lang: str = "en") -> str:
    """Fold a teacher's field name to an ASCII identifier the way a developer writing in
    that language would (German ü -> ue, Turkish ü -> u)."""
    table = FOLD.get(lang, {})
    folded = "".join(table.get(ch, ch) for ch in str(name))
    folded = unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode()
    folded = re.sub(r"\W+", "_", folded).strip("_") or "field"
    return folded if re.match(r"^[A-Za-z_]", folded) else f"q_{folded}"


def salvage(row: dict, min_q: int) -> list[str]:
    """Fold field names to ASCII and drop malformed questions in place. Returns notes;
    the caller drops the row when fewer than ``min_q`` or under half the questions remain."""
    notes = []
    questions = row.get("questions")
    if not isinstance(questions, Mapping):
        return notes
    renamed = {}
    for name in list(questions):
        new = ascii_ident(name, row.get("lang", "en"))
        while new != name and (new in questions or new in renamed.values()):
            new += "_x"
        renamed[name] = new
    dist = row.get("writer", {}).get("dist") or {}
    row["questions"] = {renamed[k]: v for k, v in questions.items()}
    row["writer"]["dist"] = {renamed.get(k, k): v for k, v in dist.items()}
    notes += [f"renamed {k!r}" for k, v in renamed.items() if k != v]
    bad = [k for k, q in row["questions"].items() if check_question(k, q)]
    for k in bad:
        notes.append(f"dropped malformed question {k}: {'; '.join(check_question(k, row['questions'][k]))[:120]}")
        del row["questions"][k]
    row["_salvage"] = {"asked": len(questions), "kept": len(row["questions"])}
    return notes


def validate_rows(rows: list[dict], *, min_q: int = 2) -> tuple[list[dict], list[dict]]:
    """Scrub PII, salvage and check schema and leaks; mark (not drop) rows failing the
    language check (the native edit pass may still fix them)."""
    kept, drops = [], []
    for row in rows:
        lang = row["lang"]
        row["state"], n_state = scrub(row["state"], lang)
        row["questions"], n_q = scrub(row["questions"], lang)
        notes = salvage(row, min_q)
        info = row.pop("_salvage", {"asked": 0, "kept": 0})
        if info["kept"] < max(min_q, (info["asked"] + 1) // 2):
            drops.append({"stage": "validate", "id": row["id"], "reason": "schema", "detail": "; ".join(notes)[:300] or "too few valid questions"})
            continue
        if notes:
            row["salvage"] = notes
        problems = check_row(row, min_q=min_q)
        if problems:
            drops.append({"stage": "validate", "id": row["id"], "reason": "schema", "detail": "; ".join(problems)[:300]})
            continue
        row["writer"]["dist"] = clean_writer_dist(row)
        leaked = leaks(row, row["writer"]["dist"])
        if leaked:
            drops.append({"stage": "validate", "id": row["id"], "reason": "leak", "detail": "; ".join(leaked)[:300]})
            continue
        row["checks"] = {"pii_replaced": n_state + n_q, "lang": language_check(row)}
        kept.append(row)
    return kept, drops


# ---------------------------------------------------------------- native edit pass (TR, DE)


def _shape(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shape(v) for v in value]
    return type(value).__name__ if not isinstance(value, str) else "str"


def _numbers(value: Any) -> list[str]:
    return sorted(re.findall(r"\d+", text_of(value)))


def edit_ok(before: Mapping, after: Mapping) -> str | None:
    """Why an edited row is rejected, or None. Structure, numbers and labels must not move."""
    if not isinstance(after.get("questions"), Mapping) or "state" not in after:
        return "missing state or questions"
    if _shape(before["state"]) != _shape(after["state"]):
        return "state structure changed"
    if _numbers(before["state"]) != _numbers(after["state"]):
        return "numbers in the state changed"
    if list(before["questions"]) != list(after["questions"]):
        return "field names changed"
    for name, q in before["questions"].items():
        new = after["questions"][name]
        if not isinstance(new, Mapping) or new.get("type") != q.get("type"):
            return f"{name}: type changed"
        crit, new_crit = q.get("criteria"), new.get("criteria")
        if isinstance(crit, Mapping) and (not isinstance(new_crit, Mapping) or list(crit) != list(new_crit)):
            return f"{name}: option labels changed"
        if isinstance(crit, list) and (not isinstance(new_crit, list) or len(crit) != len(new_crit)):
            return f"{name}: score levels changed"
    ratio = (words(after["state"]) + 1) / (words(before["state"]) + 1)
    if not 0.7 <= ratio <= 1.4:
        return "state length changed too much"
    return None


ROUTER_EDITORS = ("ds-r", "gemma26-n")


async def edit_row(client: Client, row: dict) -> dict:
    lang = row["lang"]
    writer = row["source"].split(":", 1)[1]
    editor = next((e for e in EDITORS[lang] if TEACHERS[e].lab != TEACHERS[writer].lab), None)
    if editor is None:
        row["edit"] = {"editor": None, "status": "no_editor"}
        return row
    messages = prompts.edit_messages(lang, row)
    try:
        if HFR and lang == "tr":
            # Edits GLM already made come from the cache. The rest go to DeepSeek V4 Pro (Turkish writer bake-off
            # 4.50) when another lab wrote the row; a DeepSeek row is edited (by Gemma) only if it fails the
            # language check, otherwise it keeps the writer's text.
            try:
                data, reply = await client.ask_json(editor, messages, temperature=0.2, max_tokens=16000, purpose=f"edit:{lang}", cache_only=True)
            except Unavailable:
                flagged = not row["checks"]["lang"]["ok"]
                editor = next((e for e in ROUTER_EDITORS if TEACHERS[e].lab != TEACHERS[writer].lab and (flagged or e == "ds-r")), None)
                if editor is None:
                    row["edit"] = {"editor": None, "status": "deferred"}
                    return row
                data, reply = await client.ask_json(editor, messages, temperature=0.2, max_tokens=16000, purpose=f"edit:{lang}")
        else:
            data, reply = await client.ask_json(editor, messages, temperature=0.2, max_tokens=16000, purpose=f"edit:{lang}")
    except Unavailable:  # the editor's plan is quota-limited: edit later, keep the row
        row["edit"] = {"editor": editor, "status": "deferred"}
        return row
    except GiveUp as exc:
        row["edit"] = {"editor": editor, "status": "failed", "detail": str(exc)[:200]}
        return row
    reason = edit_ok(row, data)
    if reason:
        row["edit"] = {"editor": editor, "status": "rejected", "detail": reason}
        return row
    changed = data["state"] != row["state"] or data["questions"] != row["questions"]
    row["edit"] = {"editor": f"{TEACHERS[editor].provider}/{reply.model}", "status": "changed" if changed else "unchanged", "changes": data.get("changes", [])[:12]}
    if changed:
        row["edit"]["before"] = {"state": row["state"], "questions": row["questions"]}
        row["state"], row["questions"] = data["state"], data["questions"]
        row["tags"].append(f"edited:{editor}")
    return row


async def native_edit(client: Client, rows: list[dict], *, sample: float = 0.1, seed: str = "pilot") -> tuple[list[dict], list[dict]]:
    """Edit a seeded sample plus every language-flagged row, then drop rows that still fail."""
    lang = rows[0]["lang"] if rows else "en"
    if lang in EDITORS:
        r = rng("edit", lang, seed)
        chosen = [row for row in rows if not row["checks"]["lang"]["ok"] or r.random() < sample]
        if rows and not chosen:
            chosen = [rows[0]]
        async for _ in bounded([edit_row(client, row) for row in chosen], limit=24 if HFR else 8):
            pass
    kept, drops = [], []
    for row in rows:
        if "edit" in row and row["edit"]["status"] == "changed":
            problems = check_row(row, min_q=2)
            if problems:  # an edit that broke the row falls back to the original text
                row["state"], row["questions"] = row["edit"]["before"]["state"], row["edit"]["before"]["questions"]
                row["edit"]["status"] = "reverted"
                row["tags"] = [t for t in row["tags"] if not t.startswith("edited:")]
            row["checks"]["lang"] = language_check(row)
        if not row["checks"]["lang"]["ok"]:
            drops.append({"stage": "language", "id": row["id"], "reason": "language", "detail": str(row["checks"]["lang"])})
            continue
        kept.append(row)
    return kept, drops


def card_adherence(rows: list[dict]) -> dict:
    """How closely writers followed the cards: option counts, lengths and intended answers."""
    stats = {"questions": 0, "count_match": 0, "len_ratio": [], "target_hit": 0, "targets": 0}
    for row in rows:
        card = row["card"]
        stats["len_ratio"].append(words(row["state"]) / max(1, card["length"]))
        dist = row["writer"].get("dist", {})
        for spec, (name, q) in zip(card["questions"], row["questions"].items()):
            stats["questions"] += 1
            if spec["type"] == "choice":
                stats["count_match"] += len(q.get("criteria") or {}) == spec.get("options")
            elif spec["type"] == "score":
                stats["count_match"] += len(q.get("criteria") or []) == spec.get("levels")
            else:
                stats["count_match"] += 1
            target = spec["target"]
            if name in dist and not target.startswith("ambiguous"):
                stats["targets"] += 1
                top = argmax(dist[name])
                if target.startswith("answer: "):
                    stats["target_hit"] += top == target.split()[1]
                elif target.startswith("position ") and spec["type"] == "choice":
                    labels = list((q.get("criteria") or {}))
                    k = int(target.split()[1]) - 1
                    stats["target_hit"] += k < len(labels) and labels[k] == top
                elif target.startswith("level "):
                    stats["target_hit"] += top == target.split()[1]
    return stats
