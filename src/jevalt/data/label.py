"""Three-teacher soft labels (DATA.md section 4, steps 3 and 4).

Each teacher sees every question's options in its own seeded shuffled order and
returns a verbalised probability per option. Gold = mean of the three
distributions, smoothed 0.98 * mean + 0.02 * uniform. A question survives only
when at least two teachers share the argmax (and the mean agrees with that
majority); a row that loses more than half of its questions is dropped.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from . import prompts
from .common import argmax, argmax_set, bounded, field_values, rng, smooth, target
from .providers import GLM_ONLY, HFR, TEACHERS, Client, GiveUp

# Three labs for the full label set. FALLBACK (GLM and K3, teachers:2, both must
# agree) is a smaller pair for when the full set cannot answer.
FULL = ("glm", "ds", "mistral")
FALLBACK = ("glm", "k3n")
LABELERS = FULL


async def pick_labelers(client: Client, *, allow_fallback: bool = False, wait_hours: float = 12) -> tuple[str, ...]:
    """The full three-lab set, waiting for the Ollama plan to recover if needed."""
    import asyncio
    import time

    deadline = time.time() + wait_hours * 3600
    while not await client.available("ollama"):
        if allow_fallback:
            return FALLBACK
        if time.time() > deadline:
            raise RuntimeError("Ollama stayed quota-limited; the three-lab label set is unavailable")
        await asyncio.sleep(client.PROBE_S)
    return FULL
ROWS_PER_CALL = 12


WRITER = "writer"  # pseudo-teacher: the writer's own recorded distribution (GLM-only mode)
ROUTER_LABELERS = ("dsf-r", "qwen35-r", "gemma26-n")  # preference order on the HF router
# German and the fix-set checks use Qwen3.5-35B-A3B (half the price; 23/27 vs 22/27 on the German pilot probe).
ROUTER_LABELERS_BY_LANG = {"de": ("dsf-r", "qwen35s-r", "gemma26-n")}


def pair_for(writer_lab: str, editor_lab: str | None = None, lang: str = "") -> tuple[str, str] | None:
    """Two teachers for a teacher-written row, both from labs other than the writer's (and, on the HF router, the
    editor's). Subscription lanes: GLM (Z.ai) plus an Ollama lab; GLM-only mode pairs GLM with the writer's own
    distribution."""
    if HFR:
        # HF router: the first two of DeepSeek V4.1 Flash, Qwen3.5 and Gemma 4 whose labs neither wrote nor edited
        # the row. G rows get no traces (both teachers must agree), so DeepSeek labeling one never collides with
        # DeepSeek writing traces.
        touched = {writer_lab, editor_lab} - {None}
        pair = tuple(t for t in ROUTER_LABELERS_BY_LANG.get(lang, ROUTER_LABELERS) if TEACHERS[t].lab not in touched)[:2]
        return pair if len(pair) == 2 else None
    if GLM_ONLY:
        # Only GLM has quota: pair it with the writer's distribution, a different lab. GLM-written rows get no label.
        return None if writer_lab in {"zhipu", ""} else ("glm", WRITER)
    if writer_lab == "deepseek":
        return ("glm", "mistral")
    if writer_lab in {"mistral", "google", "alibaba"}:
        return ("glm", "ds")
    return ("ds", "mistral")  # GLM, K3 or unknown writer: GLM must not label its own rows


def fix_teachers(lang: str) -> tuple[str, ...]:
    """Teachers for fix-set probes, independent of the language's main writers."""
    if HFR:
        return ("gemma26-n", "qwen35s-r")
    if GLM_ONLY:
        return ("glm", "glmt")  # same lab, two modes: a check on constructed targets, not an independent label
    # Fix-set targets are constructed or rule-checked; teachers verify them. Two labs (both must agree)
    # halve the calls on the quota-limited Ollama plan compared with the three-lab set.
    return ("glm", "ds")
SUM_TOL = 0.02


def display_order(row_id: str, field: str, values: Sequence[str], teacher: str) -> list[str]:
    order = list(values)
    rng("order", row_id, field, teacher).shuffle(order)
    return order


def parse_dist(raw: Any, values: Sequence[str]) -> tuple[dict[str, float] | None, str | None]:
    """Teacher answer for one question -> distribution over ``values`` (or a problem)."""
    if not isinstance(raw, Mapping) or not raw:
        return None, "no probabilities"
    lookup = {str(v).strip("`").casefold(): v for v in values}
    dist = dict.fromkeys(values, 0.0)
    for key, p in raw.items():
        label = lookup.get(str(key).strip().strip("`").casefold())
        if label is None:
            return None, f"unknown option {key!r}"
        try:
            p = float(p)
        except (TypeError, ValueError):
            return None, f"non-numeric probability for {key!r}"
        if not 0 <= p <= 1 or math.isnan(p):
            return None, f"probability out of range for {key!r}"
        dist[label] += p
    total = sum(dist.values())
    given = sum(1 for key in raw if lookup.get(str(key).strip().strip("`").casefold()) is not None)
    if len(values) == 2 and given == 1 and total <= 1:
        # A single probability on a two-option question implies its complement
        # (teachers often write {"yes": 0.95} and leave the 0.05 out).
        missing = next(v for v in values if dist[v] == 0.0)
        dist[missing] = 1 - total
        total = 1.0
    if abs(total - 1) > SUM_TOL:
        return None, f"probabilities sum to {total:.2f}"
    return {k: v / total for k, v in dist.items()}, None


async def _ask(client: Client, teacher: str, rows: Sequence[Mapping], values: Mapping[str, Mapping], unknown: bool, second: bool = False) -> dict:
    keys = {f"r{i + 1}": row for i, row in enumerate(rows)}
    blocks = [
        prompts.row_block(key, row, orders={f: display_order(row["id"], f, values[row["id"]][f], teacher) for f in row["questions"]}, unknown=unknown)
        for key, row in keys.items()
    ]

    def check(data: Any) -> dict:
        got = data.get("rows") if isinstance(data, Mapping) else None
        if not isinstance(got, Mapping) or not got:
            raise ValueError('expected {"rows": {"r1": {...}}}')
        return got

    messages = prompts.label_messages(blocks)
    if second:  # a changed prompt, so the retry is a fresh call and not the cached reply
        messages[-1]["content"] += "\n\nSecond pass: an earlier answer for these rows was incomplete. Answer every question of every row."
    try:
        budget = 24000 if TEACHERS[teacher].thinks else 8000  # reasoning tokens count against max_tokens
        got, _ = await client.ask_json(teacher, messages, check=check, temperature=0.2, max_tokens=budget, purpose="label")
    except GiveUp:
        got = {}
    out = {}
    for key, row in keys.items():
        answers = got.get(key) if isinstance(got.get(key), Mapping) else {}
        out[row["id"]] = {f: parse_dist(answers.get(f), values[row["id"]][f]) for f in row["questions"]}
    return out


async def teacher_dists(
    client: Client,
    rows: Sequence[Mapping],
    *,
    teachers: Sequence[str] = LABELERS,
    unknown: bool = False,
    rows_per_call: int = ROWS_PER_CALL,
) -> dict[str, dict[str, dict[str, dict | None]]]:
    """{row_id: {teacher: {field: dist or None}}}; failed questions get one targeted retry."""
    values = {row["id"]: _values(row, unknown) for row in rows}
    result: dict[str, dict[str, dict]] = {row["id"]: {} for row in rows}
    problems: dict[str, dict[str, dict[str, str]]] = {row["id"]: {} for row in rows}

    async def run(teacher: str, batch: Sequence[Mapping], second: bool = False) -> tuple[str, dict]:
        return teacher, await _ask(client, teacher, batch, values, unknown, second)

    batches = [rows[i : i + rows_per_call] for i in range(0, len(rows), rows_per_call)]
    async for teacher, got in bounded([run(t, b) for t in teachers for b in batches], limit=24):
        for rid, fields in got.items():
            result[rid][teacher] = {f: d for f, (d, _) in fields.items()}
            problems[rid][teacher] = {f: p for f, (_, p) in fields.items() if p}
    retry = [(t, [row for row in rows if problems[row["id"]].get(t)]) for t in teachers]
    coros = [run(t, sub[i : i + rows_per_call], True) for t, sub in retry for i in range(0, len(sub), rows_per_call)]
    async for teacher, got in bounded(coros, limit=24):
        for rid, fields in got.items():
            for f, (d, _) in fields.items():
                if result[rid][teacher].get(f) is None and d is not None:
                    result[rid][teacher][f] = d
    # Last pass for answers still missing, in batches of three rows: long batches are where teachers skip rows.
    missing = [(t, [row for row in rows if any(result[row["id"]].get(t, {}).get(f) is None for f in row["questions"])]) for t in teachers]
    coros = [run(t, sub[i : i + 3]) for t, sub in missing for i in range(0, len(sub), 3)]
    async for teacher, got in bounded(coros, limit=24):
        for rid, fields in got.items():
            for f, (d, _) in fields.items():
                if result[rid].setdefault(teacher, {}).get(f) is None and d is not None:
                    result[rid][teacher][f] = d
    return result


ROWS_PER_CALL_BY_TEACHER = {"gemma26-n": 6}  # Gemma 4 26B skipped rows in 12-row batches (2026-10-01 EN run)


async def dists_by_teacher(client: Client, rows: Sequence[Mapping], pairs: Mapping[str, Sequence[str]]) -> dict[str, dict[str, dict]]:
    """Each teacher labels its own rows (in the given order), so a row moving to another pair leaves the other
    teachers' batches, and their cached replies, unchanged. ``pairs`` maps row id -> teachers.
    Returns {row_id: {teacher: {field: dist or None}}}."""
    import asyncio

    by_teacher: dict[str, list[Mapping]] = {}
    for row in rows:
        for t in pairs[row["id"]]:
            by_teacher.setdefault(t, []).append(row)
    names = list(by_teacher)
    results = await asyncio.gather(*[
        teacher_dists(client, by_teacher[t], teachers=(t,), rows_per_call=ROWS_PER_CALL_BY_TEACHER.get(t, ROWS_PER_CALL)) for t in names
    ])
    out: dict[str, dict[str, dict]] = {row["id"]: {} for row in rows}
    for t, res in zip(names, results):
        for rid, per in res.items():
            out[rid][t] = per.get(t, {})
    return out


def _values(row: Mapping, unknown: bool) -> dict[str, tuple[str, ...]]:
    if not unknown:
        return field_values(row)
    return field_values({**row, "tags": list(row.get("tags", [])) + ["abstain"]})


def aggregate(values: Sequence[str], dists: Mapping[str, Mapping[str, float] | None]) -> tuple[dict | None, str | None]:
    """Gold distribution for one question, or the reason it is dropped. Every configured
    teacher must answer and at least two must share the argmax."""
    present = [d for d in dists.values() if d is not None]
    if len(present) < max(2, len(dists)):
        return None, "teacher_missing"
    votes: dict[str, int] = {}
    for d in present:
        for label in argmax_set(d):
            votes[label] = votes.get(label, 0) + 1
    majority = {label for label, n in votes.items() if n >= 2}
    if not majority:
        return None, "no_majority"
    mean = {v: sum(d.get(v, 0.0) for d in present) / len(present) for v in values}
    gold = smooth(mean)
    if argmax(gold) not in majority:
        return None, "mean_disagrees_with_majority"
    return gold, None


def writer_argmax_disagrees(row: dict, targets: dict, teachers: Sequence[str]) -> list[str]:
    """Fields where the writer's argmax differs from the teachers' mean argmax.
    The writer's own distribution is a filter: the row drops if they disagree
    on any question (two-teacher G row rule from DATA.md full-run brief)."""
    from .generate import clean_writer_dist

    writer = clean_writer_dist(row)  # over the real option values; malformed fields are left out of the filter
    return [field for field in targets if field in writer and argmax(writer[field]) != targets[field]["label"]]


def apply_labels(row: dict, dists: Mapping[str, Mapping[str, dict | None]], *, teachers: Sequence[str] = LABELERS) -> tuple[dict | None, list[dict]]:
    """Attach targets and teacher distributions; drop failed questions or the whole row."""
    values = field_values(row)
    kept, drops = {}, []
    for field in row["questions"]:
        per = {t: dists.get(t, {}).get(field) for t in teachers}
        gold, reason = aggregate(values[field], per)
        if gold is None:
            drops.append({"stage": "label", "id": row["id"], "field": field, "reason": reason})
        else:
            kept[field] = gold
    n = len(row["questions"])
    if n - len(kept) > n / 2 or not kept:
        drops.append({"stage": "label", "id": row["id"], "reason": "row_lost_most_questions", "detail": f"kept {len(kept)} of {n}"})
        return None, drops
    row = dict(row)
    row["tags"] = [t for t in row.get("tags", []) if not t.startswith("teachers:")] + [f"teachers:{len(teachers)}"]
    row["questions"] = {f: q for f, q in row["questions"].items() if f in kept}
    row["targets"] = {f: target(d) for f, d in kept.items()}
    row["teachers"] = {
        t: {
            "model": TEACHERS[t].id if t in TEACHERS else (row.get("writer") or {}).get("model", t),
            "dist": {f: {k: round(v, 4) for k, v in dists[t][f].items()} for f in kept},
        }
        for t in teachers
    }
    return row, drops


def keep_fields(row: dict, fields: Sequence[str]) -> dict:
    """The row with only ``fields`` in its questions, targets and teacher distributions."""
    row = dict(row)
    row["questions"] = {f: q for f, q in row["questions"].items() if f in fields}
    row["targets"] = {f: t for f, t in row["targets"].items() if f in fields}
    row["teachers"] = {t: {**v, "dist": {f: d for f, d in v["dist"].items() if f in fields}} for t, v in row.get("teachers", {}).items()}
    return row


def agreement_stats(rows: Sequence[Mapping]) -> dict:
    """Pairwise argmax agreement, unanimity, confidence and triviality over kept
    questions; each row uses its own teacher set."""
    stats: dict[str, Any] = {"questions": 0, "unanimous": 0, "trivial": 0, "pair": {}, "pair_n": {}, "tv": [], "sets": {}}
    for row in rows:
        teachers = [t for t in row.get("teachers", {})]
        key = "+".join(teachers)
        stats["sets"][key] = stats["sets"].get(key, 0) + 1
        for field in row.get("targets", {}):
            ds = [row["teachers"][t]["dist"][field] for t in teachers if field in row["teachers"][t]["dist"]]
            if len(ds) < 2:
                continue
            tops = [argmax(d) for d in ds]
            stats["questions"] += 1
            stats["unanimous"] += len(set(tops)) == 1
            stats["trivial"] += len(set(tops)) == 1 and all(d[tops[0]] >= 0.95 for d in ds)
            for i, a in enumerate(teachers):
                for b in teachers[i + 1 :]:
                    name = f"{a}-{b}"
                    stats["pair_n"][name] = stats["pair_n"].get(name, 0) + 1
                    stats["pair"][name] = stats["pair"].get(name, 0) + (argmax(row["teachers"][a]["dist"][field]) == argmax(row["teachers"][b]["dist"][field]))
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    stats["tv"].append(0.5 * sum(abs(ds[i].get(k, 0) - ds[j].get(k, 0)) for k in set(ds[i]) | set(ds[j])))
    return stats
