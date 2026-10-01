"""F: fix sets, one generator per weakness (DATA.md section 5).

Derived from labeled G rows: F1 unknown, F2 negation twins, F3 Noul/Choice twins,
F4 option permutations, F7 distractor padding, F8 prompt injection, F11 ordinal
targets. Programmatic with teacher-written templates (see ``rules``): F5 dates,
F6 numbers, F9 multi-hop, F10 inverted criteria, F12 long policies.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from jevalt.format import UNKNOWN

from . import prompts
from .common import (
    GEN_LICENSE,
    LANG_NAMES,
    argmax,
    bounded,
    data_dir,
    field_values,
    has_tag,
    one_hot,
    question_values,
    rng,
    sha,
    smooth,
    target,
    text_of,
)
from .label import FALLBACK, LABELERS, teacher_dists  # LABELERS: default when a caller passes none
from .providers import HFR, TEACHERS, Client, GiveUp
from .rules import RULES, SETS, check_variant, check_vocab, instance, template_messages
from .validate import FIELD_RE, LABEL_RE, check_row, language_check

FIX_NAME = {
    "f1": "unknown", "f2": "negation", "f3": "noul_choice", "f4": "permutation", "f5": "dates", "f6": "numbers",
    "f7": "padding", "f8": "injection", "f9": "multihop", "f10": "inverted", "f11": "ordinal", "f12": "policy",
}
# Teachers that write surface templates, rotated per rule; reasoning is on for precision.
TEMPLATE_WRITERS = {"en": ["glmt", "glmt"], "tr": ["glmt", "glmt"], "de": ["glmt", "glmt"]}
# GLM (Z.ai) rewrites G rows for F1-F3 and writes the injection pools: the plan with
# the most headroom does the bulk work.
HELPER = "glm"


def helper_for(lang: str) -> str:
    return HELPER


_ONCE: dict[tuple, asyncio.Task] = {}


def _once(key: tuple, factory):
    """Share one in-flight request between concurrent callers (sets reuse templates)."""
    task = _ONCE.get(key)
    if task is None or (task.done() and task.exception() is not None):
        task = _ONCE[key] = asyncio.ensure_future(factory())
    return task


def _templates_dir(lang: str):
    path = data_dir() / "templates" / lang
    path.mkdir(parents=True, exist_ok=True)
    return path


def _base(row: Mapping, fset: str, **extra: Any) -> dict:
    """Common fields of a fix-set row derived from ``row``."""
    tags = [t for t in row.get("tags", []) if t.split(":")[0] in {"domain", "writer", "group", "fmt"}]
    return {"lang": row["lang"], "source": f"fix:{FIX_NAME[fset]}+{row['source']}", "license": row.get("license", GEN_LICENSE),
            "split": "train", "tags": tags + [f"fix:{FIX_NAME[fset]}"], **extra}


def _finish(row: dict, fset: str) -> dict:
    row["id"] = f"{row['lang']}-{fset}-{sha([row['state'], row['questions'], row['targets'], row['tags']])[:10]}"
    kinds = {q["type"] for q in row["questions"].values()}
    row["tags"] = [t for t in row["tags"] if not t.startswith("type:")] + [f"type:{kinds.pop() if len(kinds) == 1 else 'mixed'}"]
    return {"id": row.pop("id"), **row}


async def _writer(client: Client, preferred: str) -> str:
    """The preferred teacher, or GLM with thinking when its provider is quota-limited."""
    if TEACHERS[preferred].provider == "ollama" and not await client.available("ollama"):
        return "glmt"
    return preferred


def _clear_questions(rows: Sequence[Mapping], kinds: set[str], min_top: float) -> list[tuple[Mapping, str]]:
    """(row, field) pairs whose teachers were unanimous and confident enough."""
    out = []
    for row in rows:
        for field, t in row.get("targets", {}).items():
            q = row["questions"][field]
            if q["type"] not in kinds:
                continue
            tops = {argmax(row["teachers"][x]["dist"][field]) for x in row["teachers"]}
            if len(tops) == 1 and t["dist"][t["label"]] >= min_top:
                out.append((row, field))
    return out


def _half(pairs: list[tuple[Mapping, str]], parity: int, need: int) -> list[tuple[Mapping, str]]:
    """A stable half of the source questions, so F2 and F3 twins never share a source
    (their original Noul rows would be identical); the whole pool if the half is too small."""
    half = [(row, f) for row, f in pairs if int(sha([row["id"], f])[:2], 16) % 2 == parity]
    return half if len(half) >= need else pairs


def _spread(pairs: list[tuple[Mapping, str]], n: int, r) -> list[tuple[Mapping, str]]:
    """Pick n pairs, at most one per row first, cycling over domains."""
    r.shuffle(pairs)
    by_domain: dict[str, list] = {}
    seen = set()
    for row, field in pairs:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        by_domain.setdefault(has_tag(row, "domain") or "", []).append((row, field))
    out = []
    while len(out) < n and any(by_domain.values()):
        for bucket in by_domain.values():
            if bucket and len(out) < n:
                out.append(bucket.pop())
    return out


def _single(row: Mapping, field: str, **overrides: Any) -> dict:
    """A copy of ``row`` with only one question (``question=`` for a new field name)."""
    q = overrides.pop("question") if "question" in overrides else row["questions"][field]
    return {**{k: v for k, v in row.items() if k not in {"targets", "teachers", "reasoning"}}, "questions": {field: q}, **overrides}


def _mean(dists: Mapping[str, Mapping | None], values: Sequence[str]) -> dict[str, float] | None:
    """Mean over the teachers, or None unless every configured teacher answered."""
    present = [d for d in dists.values() if d is not None]
    if len(present) < max(2, len(dists)):
        return None
    return {v: sum(d.get(v, 0.0) for d in present) / len(present) for v in values}


def _teachers_field(dists: Mapping[str, Mapping], field: str) -> dict:
    return {t: {"model": TEACHERS[t].id, "dist": {field: {k: round(v, 4) for k, v in d[field].items()}}} for t, d in dists.items() if d.get(field)}


async def _label_fallback(client: Client, probes: Sequence[Mapping],
                          dists: dict, teachers: Sequence[str],
                          ) -> tuple[dict, dict[str, tuple[str, ...]]]:
    """Retry probes whose teacher labels are all None with the fallback pair.

    Returns updated *dists* (mutated in place) and a map from probe id to the
    teacher set actually used for it.
    """
    active: dict[str, tuple[str, ...]] = {}
    if len(teachers) <= 2:
        return dists, active
    silent = [p for p in probes if all(dists[p["id"]].get(t) is None for t in teachers)]
    if not silent:
        return dists, active
    again = await teacher_dists(client, silent, teachers=FALLBACK)
    for p in silent:
        dists[p["id"]] = again[p["id"]]
        active[p["id"]] = FALLBACK
    return dists, active


# ---------------------------------------------------------------- F1 unknown


async def build_f1(client: Client, rows: Sequence[Mapping], n: int, seed: str, teachers: Sequence[str] = LABELERS) -> tuple[list[dict], list[dict]]:
    """Per source question: removed fact + unknown (gold unknown), removed fact without
    unknown (gold = teacher prior), original + unknown (control, unknown ~ 0)."""
    r = rng("f1", seed, rows[0]["lang"] if rows else "")
    # Twice the sources needed: the rewrite can be infeasible and the teachers must then
    # confirm the fact is really gone. Choice questions first: holistic scores rarely hinge on one fact.
    want = 2 * math.ceil(n / 3) + 2
    sources = _spread(_clear_questions(rows, {"choice"}, 0.7), want, r)
    used = {row["id"] for row, _ in sources}
    sources += _spread([(row, f) for row, f in _clear_questions(rows, {"score"}, 0.7) if row["id"] not in used], want - len(sources), r)
    lang = rows[0]["lang"] if rows else "en"
    drops: list[dict] = []

    async def rewrite(row: Mapping, field: str):
        q = row["questions"][field]
        # Finding every deciding fact benefits from reasoning; on the router DeepSeek V4.1 Flash (Gemma and Qwen label).
        helper = "dsf-r" if HFR else "glmt"
        msg = prompts.REMOVE_FACT.format(language=LANG_NAMES[lang], state=text_of(row["state"]), field=field,
                                         instructions=text_of(q.get("instructions", "")), options="\n".join(prompts.option_lines(q, lang=lang)))

        def check(data: Any) -> Any:
            if not isinstance(data, Mapping):
                raise ValueError("expected a JSON object")
            if data.get("feasible") is False:
                return data
            new = data.get("state")
            if isinstance(new, str) and not isinstance(row["state"], str):
                try:  # a JSON state returned as text
                    new = json.loads(new)
                except json.JSONDecodeError:
                    pass
            if new is None or type(new) is not type(row["state"]) or new == row["state"]:
                raise ValueError("state must be rewritten and keep its type (a JSON object stays an object)")
            return {**data, "state": new}

        try:
            data, _ = await client.ask_json(helper, [{"role": "user", "content": msg}], check=check, temperature=0.3, max_tokens=24000, purpose="f1:rewrite")
        except GiveUp as exc:
            return row, field, None, str(exc)
        if data.get("feasible") is False:
            return row, field, None, "infeasible: holistic question"
        return row, field, {"state": data["state"], "removed": data.get("removed", ""), "helper": helper}, None

    rewrites = []
    async for row, field, got, err in bounded([rewrite(row, f) for row, f in sources], limit=8):
        if got is None:
            drops.append({"stage": "f1", "id": row["id"], "reason": "infeasible" if err.startswith("infeasible") else "rewrite_failed", "detail": err})
        else:
            rewrites.append((row, field, got))
    removed = [_single(row, field, state=got["state"], id=f"{row['id']}~{field}") for row, field, got in rewrites]
    with_unknown = await teacher_dists(client, removed, teachers=teachers, unknown=True)
    without = await teacher_dists(client, removed, teachers=teachers, unknown=False)
    with_unknown, _fallback_u = await _label_fallback(client, removed, with_unknown, teachers)
    without, _fallback_w = await _label_fallback(client, removed, without, teachers)
    out = []
    for (row, field, got), probe in zip(rewrites, removed):
        active_u = _fallback_u.get(probe["id"], teachers)
        active_w = _fallback_w.get(probe["id"], teachers)
        vals_u = question_values(row["questions"][field], abstain=True, lang=lang)
        mean_u = _mean({t: with_unknown[probe["id"]].get(t, {}).get(field) for t in active_u}, vals_u)
        votes = sum(argmax(d[field]) == UNKNOWN for d in with_unknown[probe["id"]].values() if d.get(field))
        if mean_u is None or votes < 2:
            drops.append({"stage": "f1", "id": row["id"], "field": field, "reason": "fact_not_removed", "detail": f"unknown votes {votes}"})
            continue
        if language_check(probe)["ok"] is False:
            drops.append({"stage": "f1", "id": row["id"], "reason": "language"})
            continue
        vals = question_values(row["questions"][field], lang=lang)
        prior = _mean({t: without[probe["id"]].get(t, {}).get(field) for t in active_w}, vals)
        twin = f"twin:f1-{row['id']}-{field}"
        base = _base(row, "f1", questions={field: row["questions"][field]})
        a = {**base, "state": got["state"], "tags": base["tags"] + ["abstain", twin], "targets": {field: target(one_hot(UNKNOWN, vals_u))},
             "teachers": _teachers_field(with_unknown[probe["id"]], field), "note": {"removed": got["removed"], "helper": got["helper"]}}
        out.append(_finish(a, "f1"))
        if prior is not None:
            b = {**base, "state": got["state"], "tags": base["tags"] + [twin], "targets": {field: target(smooth(prior))},
                 "teachers": _teachers_field(without[probe["id"]], field), "note": {"removed": got["removed"], "helper": got["helper"]}}
            out.append(_finish(b, "f1"))
        orig = list(row["teachers"])
        orig_mean = {v: sum(row["teachers"][t]["dist"][field].get(v, 0) for t in orig) / len(orig) for v in vals}
        c = {**base, "state": row["state"], "tags": base["tags"] + ["abstain", twin, "control"],
             "targets": {field: target(smooth({**orig_mean, UNKNOWN: 0.0}))}}
        out.append(_finish(c, "f1"))
    return out[:n], drops


# ---------------------------------------------------------------- F2 negation, F3 Noul vs Choice


def _noul_yes(row: Mapping, field: str) -> float:
    return row["targets"][field]["dist"]["yes"]


def complement(dist: Mapping[str, float]) -> dict[str, float]:
    """Gold for the negated Noul: P(not q = yes) = P(q = no)."""
    return {"no": dist["yes"], "yes": dist["no"]}


def twin_dist(noul: Mapping[str, float], options: Sequence[str], yes_option: str) -> dict[str, float]:
    """Gold for the two-option Choice twin: the yes-option gets exactly P(yes)."""
    return {k: (noul["yes"] if k == yes_option else noul["no"]) for k in options}


async def build_f2(client: Client, rows: Sequence[Mapping], n: int, seed: str, teachers: Sequence[str] = LABELERS) -> tuple[list[dict], list[dict]]:
    lang = rows[0]["lang"] if rows else "en"
    r = rng("f2", seed, lang)
    want = n // 2 + 4  # a few spares: the ensemble must confirm every negation
    sources = _spread(_half(_clear_questions(rows, {"noul"}, 0.8), 0, want), want, r)
    items = [{"id": f"q{i + 1}", "question": row["questions"][field]} for i, (row, field) in enumerate(sources)]
    if not items:
        return [], []
    helper = helper_for(lang)

    def check(data: Any) -> dict:
        got = {x.get("id"): x for x in (data.get("items") or []) if isinstance(x, Mapping)} if isinstance(data, Mapping) else {}
        if len(got) < len(items) / 2:
            raise ValueError("answer every item")
        return got

    try:
        got, _ = await client.ask_json(helper, [{"role": "user", "content": prompts.NEGATE.format(language=LANG_NAMES[lang], items=prompts.item_list(items, lang))}],
                                       check=check, temperature=0.2, max_tokens=8000, purpose="f2:negate")
    except GiveUp as exc:
        return [], [{"stage": "f2", "reason": "negation_failed", "detail": str(exc)}]
    drops, probes, pairs = [], [], []
    for item, (row, field) in zip(items, sources):
        neg = got.get(item["id"])
        q = row["questions"][field]
        if not neg or not FIELD_RE.match(str(neg.get("field", ""))) or neg.get("field") == field or not neg.get("instructions"):
            drops.append({"stage": "f2", "id": row["id"], "reason": "bad_negation"})
            continue
        crit = prompts.noul_descriptions(q)
        neg_q = {"type": "noul", "instructions": neg["instructions"]}
        if crit["yes"] and crit["no"]:
            neg_q["criteria"] = {"yes": crit["no"], "no": crit["yes"]}  # the negation's yes is the original's no
        probe = _single(row, neg["field"], question=neg_q, id=f"{row['id']}~neg~{field}")
        probes.append(probe)
        pairs.append((row, field, neg["field"], neg_q, probe))
    dists = await teacher_dists(client, probes, teachers=teachers)
    dists, _fallback = await _label_fallback(client, probes, dists, teachers)
    out = []
    for row, field, neg_field, neg_q, probe in pairs:
        p_yes = _noul_yes(row, field)
        active = _fallback.get(probe["id"], teachers)
        mean = _mean({t: dists[probe["id"]].get(t, {}).get(neg_field) for t in active}, ["no", "yes"])
        if mean is None or abs(mean["yes"] - (1 - p_yes)) > 0.25 or (mean["yes"] > 0.5) == (p_yes > 0.5):
            drops.append({"stage": "f2", "id": row["id"], "field": field, "reason": "negation_not_complementary", "detail": f"p={p_yes:.2f} neg={mean}"})
            continue
        twin = f"twin:f2-{row['id']}-{field}"
        pos = {**_base(row, "f2"), "state": row["state"], "questions": {field: row["questions"][field]}, "targets": {field: row["targets"][field]}}
        pos["tags"] += [twin, "polarity:original"]
        neg_dist = complement(row["targets"][field]["dist"])
        negr = {**_base(row, "f2"), "state": row["state"], "questions": {neg_field: neg_q}, "targets": {neg_field: target(neg_dist)},
                "teachers": _teachers_field(dists[probe["id"]], neg_field)}
        negr["tags"] += [twin, "polarity:negated"]
        out += [_finish(pos, "f2"), _finish(negr, "f2")]
    return out[:n], drops


async def build_f3(client: Client, rows: Sequence[Mapping], n: int, seed: str, teachers: Sequence[str] = LABELERS) -> tuple[list[dict], list[dict]]:
    lang = rows[0]["lang"] if rows else "en"
    r = rng("f3", seed, lang)
    sources = _spread(_half(_clear_questions(rows, {"noul"}, 0.6), 1, n // 2 + 2), n // 2 + 2, r)
    items = [{"id": f"q{i + 1}", "question": row["questions"][field]} for i, (row, field) in enumerate(sources)]
    if not items:
        return [], []
    helper = helper_for(lang)

    def check(data: Any) -> dict:
        got = {x.get("id"): x for x in (data.get("items") or []) if isinstance(x, Mapping)} if isinstance(data, Mapping) else {}
        if len(got) < len(items) / 2:
            raise ValueError("answer every item")
        return got

    try:
        got, _ = await client.ask_json(helper, [{"role": "user", "content": prompts.CHOICE_TWIN.format(language=LANG_NAMES[lang], items=prompts.item_list(items, lang))}],
                                       check=check, temperature=0.2, max_tokens=8000, purpose="f3:twin")
    except GiveUp as exc:
        return [], [{"stage": "f3", "reason": "twin_failed", "detail": str(exc)}]
    drops, probes, pairs = [], [], []
    for item, (row, field) in zip(items, sources):
        tw = got.get(item["id"])
        opts = tw.get("options") if tw else None
        if not tw or not isinstance(opts, Mapping) or len(opts) != 2 or tw.get("yes_option") not in opts or not FIELD_RE.match(str(tw.get("field", ""))) \
                or any(not LABEL_RE.match(str(k)) for k in opts) or tw.get("field") == field:
            drops.append({"stage": "f3", "id": row["id"], "reason": "bad_choice_twin"})
            continue
        labels = list(opts)
        if r.random() < 0.5:
            labels.reverse()
        choice_q = {"type": "choice", "instructions": tw["instructions"], "criteria": {k: opts[k] for k in labels}}
        probe = _single(row, tw["field"], question=choice_q, id=f"{row['id']}~choice~{field}")
        probes.append(probe)
        pairs.append((row, field, tw["field"], choice_q, tw["yes_option"], probe))
    dists = await teacher_dists(client, probes, teachers=teachers)
    dists, _fallback = await _label_fallback(client, probes, dists, teachers)
    out = []
    for row, field, cfield, choice_q, yes_opt, probe in pairs:
        no_opt = next(k for k in choice_q["criteria"] if k != yes_opt)
        p_yes = _noul_yes(row, field)
        active = _fallback.get(probe["id"], teachers)
        mean = _mean({t: dists[probe["id"]].get(t, {}).get(cfield) for t in active}, [yes_opt, no_opt])
        if mean is None or abs(mean[yes_opt] - p_yes) > 0.25 or (mean[yes_opt] > 0.5) != (p_yes > 0.5):
            drops.append({"stage": "f3", "id": row["id"], "field": field, "reason": "choice_twin_disagrees", "detail": f"p={p_yes:.2f} choice={mean}"})
            continue
        twin = f"twin:f3-{row['id']}-{field}"
        noul = {**_base(row, "f3"), "state": row["state"], "questions": {field: row["questions"][field]}, "targets": {field: row["targets"][field]}}
        noul["tags"] += [twin, "form:noul"]
        dist = twin_dist(row["targets"][field]["dist"], list(choice_q["criteria"]), yes_opt)
        choice = {**_base(row, "f3"), "state": row["state"], "questions": {cfield: choice_q}, "targets": {cfield: target(dist)},
                  "teachers": _teachers_field(dists[probe["id"]], cfield)}
        choice["tags"] += [twin, "form:choice", f"yes_option:{yes_opt}"]
        out += [_finish(noul, "f3"), _finish(choice, "f3")]
    return out[:n], drops


# ---------------------------------------------------------------- F4 permutations, F7 padding, F11 ordinal (no teacher calls)


def build_f4(rows: Sequence[Mapping], n: int, seed: str, per_row: int = 3) -> list[dict]:
    r = rng("f4", seed, rows[0]["lang"] if rows else "")
    out = []
    for row in rows:
        choice_fields = [f for f, q in row["questions"].items() if q["type"] == "choice" and len(q["criteria"]) >= 3]
        if not choice_fields:
            continue
        for k in range(per_row):
            questions = {}
            for f, q in row["questions"].items():
                if f in choice_fields:
                    labels = list(q["criteria"])
                    r.shuffle(labels)
                    q = {**q, "criteria": {label: q["criteria"][label] for label in labels}}
                questions[f] = q
            new = {**_base(row, "f4"), "state": row["state"], "questions": questions, "targets": row["targets"], "teachers": row.get("teachers")}
            new["tags"] += [f"twin:f4-{row['id']}", f"perm:{k}"]
            out.append(_finish(new, "f4"))
            if len(out) >= n:
                return out
    return out


def build_f7(rows: Sequence[Mapping], keys: Mapping[str, str], n: int, seed: str, budgets: Sequence[int] = (1000, 2000, 3000)) -> list[dict]:
    """Wrap the state with unrelated states from other rows up to a token budget
    (estimated as chars / 3.5); the gold does not change."""
    r = rng("f7", seed, rows[0]["lang"] if rows else "")
    out = []
    pool = list(rows)
    for i, row in enumerate(rows[:n]):
        budget = budgets[i % len(budgets)]
        others, size = [], len(text_of(row["state"])) / 3.5
        domain = has_tag(row, "domain")
        for other in r.sample(pool, len(pool)):
            if other["id"] == row["id"] or has_tag(other, "domain") == domain:
                continue
            others.append(other["state"])
            size += len(text_of(other["state"])) / 3.5
            if size >= budget:
                break
        state = {keys["current"]: row["state"], keys["others"]: others}
        new = {**_base(row, "f7"), "state": state, "questions": row["questions"], "targets": row["targets"], "teachers": row.get("teachers")}
        new["tags"] += [f"twin:f7-{row['id']}", f"pad:{budget}"]
        out.append(_finish(new, "f7"))
    return out


def ordinal_smooth(dist: Mapping[str, float], order: Sequence[str], spill: float = 0.1) -> dict[str, float]:
    """Move ``spill`` of each level's mass to its neighbours (half each side, all of it
    inward at the ends), then the standard 0.02 smoothing."""
    out = dict.fromkeys(order, 0.0)
    for i, level in enumerate(order):
        p = dist.get(level, 0.0)
        neighbours = [order[j] for j in (i - 1, i + 1) if 0 <= j < len(order)]
        out[level] += p * (1 - spill)
        for nb in neighbours:
            out[nb] += p * spill / len(neighbours)
    return smooth(out)


def build_f11(rows: Sequence[Mapping], n: int) -> list[dict]:
    """Score questions where teachers split between adjacent levels."""
    out = []
    for row in rows:
        for field, q in row["questions"].items():
            if q["type"] != "score" or field not in row.get("targets", {}):
                continue
            order = list(field_values(row)[field])
            tops = sorted({order.index(argmax(row["teachers"][t]["dist"][field])) for t in row["teachers"]})
            if len(tops) != 2 or tops[1] - tops[0] != 1:
                continue
            mean = {v: sum(row["teachers"][t]["dist"][field].get(v, 0) for t in row["teachers"]) / len(row["teachers"]) for v in order}
            new = {**_base(row, "f11"), "state": row["state"], "questions": {field: q}, "targets": {field: target(ordinal_smooth(mean, order))},
                   "teachers": {t: {"model": d["model"], "dist": {field: d["dist"][field]}} for t, d in row["teachers"].items()}}
            new["tags"] += [f"twin:f11-{row['id']}-{field}"]
            out.append(_finish(new, "f11"))
            if len(out) >= n:
                return out
    return out


# ---------------------------------------------------------------- F8 injection


async def injection_pool(client: Client, lang: str) -> list[dict]:
    return await _once(("injections", lang), lambda: _injection_pool(client, lang))


async def _injection_pool(client: Client, lang: str) -> list[dict]:
    path = _templates_dir(lang) / "injections.json"
    if path.exists():
        return json.loads(path.read_text())
    writer = HELPER

    def check(data: Any) -> list:
        snippets = [s for s in (data.get("snippets") or []) if isinstance(s, Mapping)] if isinstance(data, Mapping) else []
        good = [s for s in snippets if "{field}" in str(s.get("text", "")) and "{answer}" in str(s.get("text", "")) and FIELD_RE.match(str(s.get("json_key", "")))]
        if len(good) < 12:
            raise ValueError("need at least 12 snippets that contain {field} and {answer} and have an ASCII snake_case json_key")
        return good

    snippets, reply = await client.ask_json(writer, [{"role": "user", "content": prompts.INJECTIONS.format(n=24, language=LANG_NAMES[lang])}],
                                            check=check, temperature=0.7, max_tokens=8000, purpose="f8:pool")
    for s in snippets:
        s["writer"] = f"{TEACHERS[writer].provider}/{reply.model}"
    path.write_text(json.dumps(snippets, ensure_ascii=False, indent=1))
    return snippets


def inject(state: Any, text: str, key: str, r) -> tuple[Any, str]:
    """Insert hostile text at the start, middle or end of a state."""
    where = r.choice(["start", "middle", "end"])
    if isinstance(state, str):
        lines = state.split("\n")
        pos = {"start": 0, "end": len(lines), "middle": len(lines) // 2}[where]
        return "\n".join(lines[:pos] + [text] + lines[pos:]), where
    if isinstance(state, list):
        pos = {"start": 0, "end": len(state), "middle": len(state) // 2}[where]
        return state[:pos] + [text] + state[pos:], where
    items = list(state.items())
    pos = {"start": 0, "end": len(items), "middle": len(items) // 2}[where]
    items.insert(pos, (key if key not in state else key + "_2", text))
    return dict(items), where


async def build_f8(client: Client, rows: Sequence[Mapping], n: int, seed: str, teachers: Sequence[str] = LABELERS) -> tuple[list[dict], list[dict]]:
    lang = rows[0]["lang"] if rows else "en"
    # "A manager already decided X" can be real case evidence, so keeping the gold
    # unchanged would be wrong there; only unambiguous manipulation kinds are used.
    pool = [s for s in await injection_pool(client, lang) if s.get("kind") != "authority"]
    r = rng("f8", seed, lang)
    chosen = r.sample(list(rows), min(n, len(rows)))
    out = []
    for i, row in enumerate(chosen):
        field = r.choice(list(row["targets"]))
        gold = row["targets"][field]["label"]
        values = field_values(row)[field]
        wrong = [v for v in values if v != gold]
        if row["questions"][field]["type"] == "score":  # push to the far end of the scale
            wrong = [values[0] if values.index(gold) >= len(values) / 2 else values[-1]]
        snippet = pool[i % len(pool)]
        text = snippet["text"].replace("{field}", field).replace("{answer}", r.choice(wrong))
        state, where = inject(row["state"], text, snippet["json_key"], r)
        new = {**_base(row, "f8"), "state": state, "questions": row["questions"], "targets": row["targets"]}
        new["tags"] += [f"twin:f8-{row['id']}", f"inject:{snippet.get('kind', 'order')}", f"inject_at:{where}", f"inject_field:{field}"]
        new["note"] = {"injected": text, "field": field}
        out.append(_finish(new, "f8"))
    dists = await teacher_dists(client, out, teachers=teachers)
    for row in out:
        row["teachers"] = {t: {"model": TEACHERS[t].id, "dist": {f: {k: round(v, 4) for k, v in d.items()} for f, d in dists[row["id"]].get(t, {}).items() if d}} for t in teachers}
    return out, []


# ---------------------------------------------------------------- rule-based sets (F5, F6, F9, F10, F12)


async def names_pool(client: Client, lang: str) -> dict:
    return await _once(("names", lang), lambda: _names_pool(client, lang))


async def _names_pool(client: Client, lang: str) -> dict:
    path = _templates_dir(lang) / "names.json"
    if path.exists():
        return json.loads(path.read_text())
    writer = {"en": "ds", "tr": "glm", "de": "mistral"}[lang]

    def check(data: Any) -> dict:
        if not isinstance(data, Mapping) or any(len(data.get(k) or []) < 10 for k in ("people", "orgs", "villagers")):
            raise ValueError("need people, orgs and villagers lists")
        return {k: [str(x) for x in data[k]] for k in ("people", "orgs", "villagers")}

    msg = prompts.NAMES.format(language=LANG_NAMES[lang], country=prompts.COUNTRY[lang])
    pool, reply = await client.ask_json(writer, [{"role": "user", "content": msg}], check=check, temperature=0.8, max_tokens=4000, purpose="names")
    pool["writer"] = f"{TEACHERS[writer].provider}/{reply.model}"
    path.write_text(json.dumps(pool, ensure_ascii=False, indent=1))
    return pool


async def rule_templates(client: Client, lang: str, name: str, k: int = 3) -> dict:
    """Teacher-written variants for one rule, validated, cached as a file."""
    return await _once(("rule", lang, name), lambda: _rule_templates(client, lang, name, k))


async def _rule_templates(client: Client, lang: str, name: str, k: int) -> dict:
    path = _templates_dir(lang) / f"rule-{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    rule = RULES[name]
    writers = TEMPLATE_WRITERS[lang]
    writer = await _writer(client, writers[sorted(RULES).index(name) % len(writers)])
    rejected: list[str] = []

    def check(data: Any) -> dict:
        variants = data.get("variants") if isinstance(data, Mapping) else None
        if not isinstance(variants, list):
            raise ValueError('expected {"variants": [...]}')
        vocab = check_vocab(rule, data.get("vocab"))
        good, errors = [], []
        for v in variants:
            try:
                good.append(check_variant(rule, v))
            except ValueError as exc:
                errors.append(str(exc))
        if len(good) < max(1, k - 1):
            raise ValueError("; ".join(errors)[:600] or "no usable variant")
        rejected[:] = errors
        return {"variants": good, "vocab": vocab}

    data, reply = await client.ask_json(writer, template_messages(rule, lang, k), check=check, temperature=0.5, max_tokens=24000, purpose=f"tpl:{name}")
    data["writer"] = f"{TEACHERS[writer].provider}/{reply.model}"
    data["writer_teacher"] = writer
    data["rejected_variants"] = rejected
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return data


async def build_rule_set(client: Client, lang: str, fset: str, n: int, seed: str) -> tuple[list[dict], list[dict]]:
    names = await names_pool(client, lang)
    rule_names = SETS[fset]
    templates = {}
    drops = []
    async for name, got in bounded([_named(name, rule_templates(client, lang, name)) for name in rule_names], limit=8):
        if isinstance(got, Exception):
            drops.append({"stage": fset, "reason": "templates_failed", "detail": f"{name}: {got}"})
        else:
            templates[name] = got
    usable = [name for name in rule_names if name in templates]
    out = []
    for i in range(n):
        if not usable:
            break
        name = usable[i % len(usable)]
        rule, tpl = RULES[name], templates[name]
        v_index = (i // len(usable)) % len(tpl["variants"])
        r = rng(fset, lang, seed, i)
        invert = r.choice([q.key for q in rule.questions if q.kind == "noul"]) if fset == "f10" else None
        inst = instance(rule, tpl["variants"][v_index], tpl["vocab"], names, lang, r, invert=invert)
        tpl_id = f"{lang}-{name}-{v_index}"
        row = {
            "lang": lang, "source": f"prog:{FIX_NAME[fset]}", "license": GEN_LICENSE, "split": "train",
            "tags": [f"fix:{FIX_NAME[fset]}", f"rule:{name}", f"tpl:{tpl_id}", f"group:tpl-{tpl_id}", f"writer:{tpl['writer']}", f"locale:{inst['locale']}"],
            "state": inst["state"], "questions": inst["questions"],
        }
        if invert:
            row["tags"].append(f"inverted:{tpl['variants'][v_index]['questions'][invert]['field']}")
        values = field_values(row)
        row["targets"] = {f: target(one_hot(label, values[f])) for f, label in inst["gold"].items()}
        problems = check_row(row, min_q=1)
        if problems:
            drops.append({"stage": fset, "reason": "schema", "detail": "; ".join(problems), "tpl": tpl_id})
            continue
        out.append(_finish(row, fset))
    return out, drops


async def _named(name: str, coro):
    try:
        return name, await coro
    except (GiveUp, ValueError, RuntimeError) as exc:
        return name, exc


# Reasoning verifiers for templates the non-thinking ensemble disputes (date and number
# arithmetic trips non-thinking labelers); never the template writer's own lab.
VERIFIERS = ("glmt", "dst")


async def _verifier(client: Client, writer_id: str) -> str | None:
    """A reasoning verifier from a lab other than the template writer's, if one answers."""
    from .reasoning import lab_of

    for v in VERIFIERS:
        if TEACHERS[v].lab == lab_of(writer_id):
            continue
        if TEACHERS[v].provider == "ollama" and not await client.available("ollama"):
            continue
        return v
    return None


async def verify_programmatic(client: Client, rows: list[dict], *, teachers: Sequence[str] = LABELERS, reject: bool = True) -> tuple[list[dict], list[dict], dict]:
    """Label programmatic rows with the teacher ensemble (kept as a difficulty measure;
    rows keep their exact gold). A template is dropped only when the ensemble majority
    disagrees with the gold on at least half of its checks AND a reasoning verifier from
    another lab disagrees as well. ``reject`` False (F10) only measures."""
    dists = await teacher_dists(client, rows, teachers=teachers)
    per_tpl: dict[str, list[bool]] = {}
    for row in rows:
        row["teachers"] = {t: {"model": TEACHERS[t].id, "dist": {f: {k: round(v, 4) for k, v in d.items()} for f, d in dists[row["id"]].get(t, {}).items() if d}} for t in teachers}
        for f, t in row["targets"].items():
            votes = [argmax(row["teachers"][x]["dist"][f]) for x in teachers if f in row["teachers"][x]["dist"]]
            ok = sum(v == t["label"] for v in votes) >= 2
            row.setdefault("check", {})[f] = ok
            per_tpl.setdefault(has_tag(row, "tpl") or "", []).append(ok)
    stats = {tpl: {"agree": sum(oks), "n": len(oks)} for tpl, oks in per_tpl.items()}
    disputed = {tpl for tpl, oks in per_tpl.items() if reject and len(oks) >= 2 and sum(oks) / len(oks) < 0.5}
    bad = set()
    for tpl in sorted(disputed):
        tpl_rows = [row for row in rows if has_tag(row, "tpl") == tpl]
        verifier = await _verifier(client, has_tag(tpl_rows[0], "writer") or "")
        if verifier is None:
            stats[tpl]["verifier"] = "none available"
            continue
        second = await teacher_dists(client, tpl_rows, teachers=(verifier,))
        oks = [argmax(d) == row["targets"][f]["label"] for row in tpl_rows for f, d in second[row["id"]].get(verifier, {}).items() if d]
        stats[tpl].update(verifier=verifier, verifier_agree=sum(oks), verifier_n=len(oks))
        if not oks or sum(oks) / len(oks) < 0.5:
            bad.add(tpl)
    kept = [row for row in rows if has_tag(row, "tpl") not in bad]
    drops = [{"stage": "verify", "id": row["id"], "reason": "template_rejected_by_verifier", "tpl": has_tag(row, "tpl")} for row in rows if has_tag(row, "tpl") in bad]
    return kept, drops, stats


async def padding_keys(client: Client, lang: str) -> dict:
    """Wrapper keys for F7 (the case to decide, unrelated records), written by a teacher."""
    path = _templates_dir(lang) / "padding.json"
    if path.exists():
        return json.loads(path.read_text())

    def check(data: Any) -> dict:
        if not isinstance(data, Mapping) or not all(FIELD_RE.match(str(data.get(k, ""))) for k in ("current", "others")) or data["current"] == data["others"]:
            raise ValueError('need {"current": "<key>", "others": "<key>"} as two different ASCII snake_case keys')
        return {"current": data["current"], "others": data["others"]}

    keys, reply = await client.ask_json(HELPER, [{"role": "user", "content": prompts.PAD_KEYS.format(language=LANG_NAMES[lang])}], check=check, temperature=0.3, max_tokens=2000, purpose="f7:keys")
    keys["writer"] = f"{TEACHERS[HELPER].provider}/{reply.model}"
    path.write_text(json.dumps(keys, ensure_ascii=False, indent=1))
    return keys
