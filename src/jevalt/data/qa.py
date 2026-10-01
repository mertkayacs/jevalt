"""QA: schema, compilation, language, PII, dedup and statistics for a set of rows,
plus the markdown report that renders examples for reading."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from jevalt.format import render

from .common import argmax, compiled, data_dir, field_values, has_tag, read_jsonl, sha, text_of
from .validate import check_row, check_targets, language_check, pii_hits

RECORD_KEYS = ("id", "lang", "source", "license", "split", "tags", "state", "questions", "targets", "reasoning", "teachers", "abstain")
SPLITS = {"train", "calibration", "validation", "test"}


def strip(row: Mapping) -> dict:
    """Keep only the record fields of DESIGN.md section 3."""
    return {k: row[k] for k in RECORD_KEYS if k in row and row[k] not in (None, {}, "")}


def schema_problems(row: Mapping) -> list[str]:
    problems = [f"missing {k}" for k in RECORD_KEYS[:9] if k not in row]
    if problems:
        return problems
    if row["lang"] not in {"en", "tr", "de"}:
        problems.append("bad lang")
    if row["split"] not in SPLITS:
        problems.append("bad split")
    if not isinstance(row["tags"], list) or not all(isinstance(t, str) for t in row["tags"]):
        problems.append("tags must be strings")
    if "reasoning" in row and not isinstance(row["reasoning"], str):
        problems.append("reasoning must be a string")
    problems += check_row(row, min_q=1)
    if not problems:
        problems += check_targets(row)
    return problems


# ---------------------------------------------------------------- dedup


def state_key(row: Mapping) -> str:
    state = row["state"]
    text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
    return " ".join(text.lower().split())


def group_of(row: Mapping) -> str:
    return has_tag(row, "group") or row["id"]


def dedup(rows: Sequence[Mapping], *, threshold: float = 0.8, n: int = 5, num_perm: int = 128) -> tuple[list[Mapping], list[dict]]:
    """Exact then MinHash (character n-grams, Jaccard >= threshold) on state text.

    Rows of one group (a G row and the fix-set rows built from it, or the rows of
    one template) may share a state; duplicates across groups keep the first row.
    """
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    exact: dict[str, str] = {}
    groups: dict[str, str] = {}
    whole: dict[str, str] = {}
    kept, dups = [], []
    for row in rows:
        key, group = state_key(row), group_of(row)
        same = sha([key, row.get("questions"), row.get("targets")])
        if same in whole:  # an identical row, whatever its group (F2 and F3 can share a source)
            dups.append({"id": row["id"], "dup_of": whole[same], "kind": "identical_row"})
            continue
        whole[same] = row["id"]
        if key in exact and groups[exact[key]] != group:
            dups.append({"id": row["id"], "dup_of": exact[key], "kind": "exact"})
            continue
        mh = MinHash(num_perm=num_perm)
        for i in range(max(1, len(key) - n + 1)):
            mh.update(key[i : i + n].encode())
        near = [other for other in lsh.query(mh) if groups[other] != group]
        if near:
            dups.append({"id": row["id"], "dup_of": near[0], "kind": "minhash"})
            continue
        exact.setdefault(key, row["id"])
        if row["id"] not in groups:
            lsh.insert(row["id"], mh)
            groups[row["id"]] = group
        kept.append(row)
    return kept, dups


# ---------------------------------------------------------------- statistics


@lru_cache(maxsize=1)
def _tokenizer():
    """Intern-Decision-4B tokenizer (loads in under a second, about 200 MB)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = data_dir() / "models" / "tokenizer.json"
    if not path.exists():
        hf_hub_download("internlm/Intern-Decision-4B", "tokenizer.json", local_dir=str(path.parent))
    return Tokenizer.from_file(str(path))


def token_length(row: Mapping) -> int:
    c = compiled(row)
    if row.get("reasoning"):
        c.messages[2]["reasoning_content"] = row["reasoning"]
    try:
        return len(_tokenizer().encode(render(c)).ids)
    except Exception:  # tokenizer unavailable: DATA.md fallback estimate
        return int(len(render(c)) / 3.5)


def set_of(row: Mapping) -> str:
    if row["source"].startswith("gen:"):
        return "G"
    if row["source"].startswith("pub:"):
        return "P"
    fix = has_tag(row, "fix") or "?"
    return {"unknown": "F1", "negation": "F2", "noul_choice": "F3", "permutation": "F4", "dates": "F5", "numbers": "F6", "padding": "F7",
            "injection": "F8", "multihop": "F9", "inverted": "F10", "ordinal": "F11", "policy": "F12"}.get(fix, fix)


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:.1f}%" if b else "n/a"


def describe(values: Sequence[float]) -> str:
    if not values:
        return "n/a"
    v = sorted(values)
    p90 = v[min(len(v) - 1, int(0.9 * len(v)))]
    return f"mean {statistics.mean(v):.0f}, median {statistics.median(v):.0f}, p90 {p90:.0f}, max {v[-1]:.0f} (n={len(v)})"


def stats(rows: Sequence[Mapping]) -> dict:
    out: dict[str, Any] = {"rows": len(rows), "by_set": Counter(set_of(r) for r in rows)}
    qtypes, options, levels, qcounts = Counter(), Counter(), Counter(), Counter()
    noul_yes = defaultdict(lambda: [0, 0])
    gold_pos = defaultdict(Counter)
    for row in rows:
        qcounts[len(row["questions"])] += 1
        values = field_values(row)
        for field, q in row["questions"].items():
            qtypes[q["type"]] += 1
            gold = row["targets"][field]["label"]
            if q["type"] == "choice":
                n = len(q["criteria"])
                options["2-8" if n <= 8 else "9-30" if n <= 30 else "31-62"] += 1
                if gold in q["criteria"]:
                    gold_pos[set_of(row)][f"{list(q['criteria']).index(gold) / max(1, n - 1):.1f}"] += 1
            elif q["type"] == "score":
                levels[len(values[field]) - (1 if "unknown" in values[field] else 0)] += 1
            else:
                domain = has_tag(row, "domain") or set_of(row)
                noul_yes[domain][0] += gold == "yes"
                noul_yes[domain][1] += 1
    out.update(qtypes=qtypes, choice_options=options, score_levels=levels, questions_per_row=qcounts,
               noul_yes={d: f"{a}/{b}" for d, (a, b) in sorted(noul_yes.items())}, gold_position=gold_pos)
    return out


def label_balance(rows: Sequence[Mapping]) -> list[str]:
    """Gold-label shares above 60 percent: noul answers per domain, and labels of
    fields that recur within a domain (like game_npc next_action) at least 5 times."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        domain = has_tag(row, "domain") or set_of(row)
        for field, t in row["targets"].items():
            kind = row["questions"][field]["type"]
            if kind == "noul":
                counts[f"{domain} noul"][t["label"]] += 1
            counts[f"{domain} {field}"][t["label"]] += 1
    flags = []
    for key, c in sorted(counts.items()):
        total = sum(c.values())
        if total >= 5 and max(c.values()) / total > 0.6:
            flags.append(f"{key}: {dict(c.most_common(4))} of {total}")
    return flags


def api_usage(run: str) -> dict:
    usage: dict[str, Counter] = defaultdict(Counter)
    for rec in read_jsonl(data_dir() / "logs" / "calls.jsonl"):
        if rec.get("run") != run:
            continue
        key = f"{rec['provider']}/{rec['model']}"
        usage[key]["calls"] += 1
        usage[key]["prompt_tokens"] += rec.get("prompt_tokens") or 0
        usage[key]["completion_tokens"] += rec.get("completion_tokens") or 0
        usage[key]["seconds"] += rec.get("latency_s") or 0
    return usage


def quota_events(run: str) -> Counter:
    events = Counter()
    path = data_dir() / "logs" / "quota.log"
    for rec in read_jsonl(path):
        if rec.get("run") == run:
            events[f"{rec['provider']}:{rec['event']}:{rec.get('status')}"] += 1
    return events


# ---------------------------------------------------------------- rendering


def render_row(row: Mapping, n: int) -> str:
    lines = [f"### Example {n}: `{row['id']}` ({set_of(row)})", "", f"source `{row['source']}` · license `{row['license']}` · tags: {', '.join(f'`{t}`' for t in row['tags'])}", ""]
    lines += ["State:", "", "```", text_of(row["state"]), "```", ""]
    teachers = list((row.get("teachers") or {}).keys())
    values = field_values(row)
    for field, q in row["questions"].items():
        t = row["targets"][field]
        lines.append(f"**`{field}`** ({q['type']}): {text_of(q.get('instructions', ''))}")
        lines.append("")
        header = "| option | description | gold | " + " | ".join(teachers) + " |" if teachers else "| option | description | gold |"
        lines += [header, "|" + "---|" * (3 + len(teachers))]
        crit = q.get("criteria")
        for v in values[field]:
            if isinstance(crit, Mapping):
                desc = crit.get(v) if q["type"] != "noul" else next((val for k, val in crit.items() if str(k).lower() in ({"yes", "true", "1"} if v == "yes" else {"no", "false", "0"})), "")
            elif isinstance(crit, list) and v.isdigit() and int(v) < len(crit):
                desc = crit[int(v)]
            else:
                desc = "(unknown option)" if v == "unknown" else ""
            desc = text_of(desc).replace("\n", " ").replace("|", "/")[:160] if desc else ""
            mark = " **←**" if v == t["label"] else ""
            cells = [f"{row['teachers'][x]['dist'].get(field, {}).get(v, 0):.2f}" if field in row["teachers"][x]["dist"] else "-" for x in teachers]
            lines.append(f"| `{v}`{mark} | {desc} | {t['dist'][v]:.3f} | " + " | ".join(cells) + (" |" if teachers else ""))
        lines.append("")
    if row.get("reasoning"):
        lines += [f"Reasoning: _{row['reasoning']}_", ""]
    return "\n".join(lines)


def language_summary(rows: Iterable[Mapping]) -> dict:
    results = [language_check(r, include_questions=not has_tag(r, "qlang")) for r in rows]
    return {"checked": len(results), "pass": sum(r["ok"] for r in results), "folded": sum(bool(r.get("folded")) for r in results),
            "min_prob": min((r["prob"] for r in results), default=None), "fails": [r for r in results if not r["ok"]][:10]}


def gates(rows: Sequence[Mapping]) -> dict:
    """DATA.md section 8 checks that can run on a single language set."""
    schema = {r["id"]: p for r in rows if (p := schema_problems(r))}
    pii = {r["id"]: n for r in rows if (n := pii_hits(r["state"], r["lang"]) + pii_hits(r["questions"], r["lang"]))}
    lang = language_summary(rows)
    missing_abstain = [r["id"] for r in rows
                      if any(t.get("dist", {}).get("unknown", 0) > 0 for t in (r.get("targets") or {}).values())
                      and not r.get("abstain")]
    return {"schema_fail": schema, "pii_rows": pii, "language": lang,
            "argmax_ok": all(r["targets"][f]["label"] == argmax(r["targets"][f]["dist"]) for r in rows for f in r["targets"]),
            "abstain_missing": missing_abstain}
