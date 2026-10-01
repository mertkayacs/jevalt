"""Stage runners. Every stage reads and writes JSONL under ``data/<run>/<lang>/`` so
stages can be rerun alone; teacher replies come from the cache on reruns."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import fixsets, generate, label, qa, reasoning
from .common import data_dir, has_tag, read_jsonl, rng, write_jsonl
from .providers import HFR, Client
from .validate import check_row, check_targets, scrub

G_TARGET = 48
TOPUP_TARGET = 54  # G_TARGET plus room for rows the label ensemble drops
# Native edit share per language, plus every row that fails the language check. One editor call per row through
# Z.ai (4 concurrent) made 100% of Turkish a 7-hour stage, so 30% (2026-10-01); German rows come from a top German writer.
EDIT_SAMPLE = {"tr": 0.3, "de": 0.0}


def stage_dir(run: str, lang: str) -> Path:
    path = data_dir() / run / lang
    path.mkdir(parents=True, exist_ok=True)
    return path


def _drops(run: str, lang: str, stage: str, drops: Sequence[Mapping]) -> None:
    write_jsonl(stage_dir(run, lang) / f"drops.{stage}.jsonl", drops)


# ---------------------------------------------------------------- G


async def stage_gen(client: Client, lang: str, *, run: str, n: int = 56, seed: str = "pilot", npc_share: float = 0.25) -> list[dict]:
    d = stage_dir(run, lang)
    names = await fixsets.names_pool(client, lang)
    cards = generate.plan_cards(lang, n, npc_share=npc_share, seed=seed, names=names)
    rows, drops = await generate.generate(client, lang, cards)
    valid, val_drops = generate.validate_rows(rows)
    edit_sample = EDIT_SAMPLE.get(lang, 0.1)
    edited, lang_drops = await generate.native_edit(client, valid, sample=edit_sample, seed=seed)
    drops += val_drops + lang_drops
    # Top up short languages with fresh cards (a new seed leaves earlier batches and
    # their cache untouched).
    for k in range(1, 3):
        if len(edited) >= TOPUP_TARGET:
            break
        extra = generate.plan_cards(lang, 8, npc_share=npc_share, seed=f"{seed}-topup{k}", names=names)
        got, lost = await generate.write_cards(client, lang, generate.TOPUP_WRITER[lang], extra, 100 + k)
        v, vd = generate.validate_rows(got)
        e, ld = await generate.native_edit(client, v, seed=f"{seed}-topup{k}")
        cards += extra
        rows += got
        valid += v
        edited += e
        drops += lost + vd + ld
    (d / "cards.json").write_text(json.dumps(cards, ensure_ascii=False, indent=1))
    write_jsonl(d / "g.gen.jsonl", rows)
    write_jsonl(d / "g.valid.jsonl", valid)
    write_jsonl(d / "g.edit.jsonl", edited)
    _drops(run, lang, "gen", drops)
    return edited


async def stage_label(client: Client, lang: str, *, run: str, target_rows: int = G_TARGET) -> list[dict]:
    d = stage_dir(run, lang)
    rows = list(read_jsonl(d / "g.edit.jsonl"))

    # Two-teacher pairs: GLM plus one Ollama lab that did not write the row.
    # Writer distribution is a filter only (drops rows whose argmax disagrees
    # with the teachers' mean on any question).
    from .reasoning import lab_of

    pending: list[dict] = []  # no independent teacher available right now (GLM-only mode, GLM-written rows)
    pairs: dict[str, tuple[str, ...]] = {}
    for row in rows:
        writer_lab = lab_of(has_tag(row, "writer") or "")
        edit = row.get("edit") or {}
        editor_lab = lab_of(edit.get("editor") or "") if edit.get("status") == "changed" else None
        pair = label.pair_for(writer_lab, editor_lab, lang)
        if pair is None:
            pending.append(row)
            continue
        pairs[row["id"]] = pair

    labeled = [row for row in rows if row["id"] in pairs]
    api_pairs = {rid: tuple(t for t in pair if t != label.WRITER) for rid, pair in pairs.items()}
    dists = await label.dists_by_teacher(client, labeled, api_pairs)
    all_dists: dict[str, dict] = {}
    all_sets: dict[str, tuple[str, ...]] = {}
    for row in labeled:
        row_dists = dict(dists[row["id"]])
        if label.WRITER in pairs[row["id"]]:
            row_dists[label.WRITER] = generate.clean_writer_dist(row)
        all_dists[row["id"]] = row_dists
        all_sets[row["id"]] = pairs[row["id"]]

    kept, drops = [], []
    for row in rows:
        if row["id"] not in all_sets:
            continue
        teachers = all_sets[row["id"]]
        new, dropped = label.apply_labels(row, all_dists[row["id"]], teachers=teachers)
        drops += dropped
        if new:
            # Writer distribution filter: a question whose writer argmax differs from the teachers' mean is dropped,
            # so every kept question has writer and teachers agreeing; the row needs more than half of its questions.
            disagree = label.writer_argmax_disagrees(new, new.get("targets", {}), teachers)
            drops += [{"stage": "label", "id": row["id"], "field": f, "reason": "writer_argmax_disagrees"} for f in disagree]
            keep = [f for f in new["targets"] if f not in disagree]
            n = len(row["questions"])
            if not keep or n - len(keep) > n / 2:
                drops.append({"stage": "label", "id": row["id"], "reason": "row_lost_most_questions", "detail": f"kept {len(keep)} of {n} after the writer filter"})
                continue
            if disagree:
                new = label.keep_fields(new, keep)
            kept.append(new)
    picked = pick_g(kept, target_rows)
    chosen = {r["id"] for r in picked}
    drops += [{"stage": "label", "id": r["id"], "reason": "over_target"} for r in kept if r["id"] not in chosen]
    kept = picked
    drops += [{"stage": "label", "id": r["id"], "reason": "no_independent_teacher"} for r in pending]
    write_jsonl(d / "g.pending.jsonl", pending)  # labeled later, when a second lab has quota
    write_jsonl(d / "g.jsonl", kept)
    _drops(run, lang, "label", drops)
    return kept


def pick_g(rows: list[dict], n: int, min_npc: int = 10) -> list[dict]:
    """Keep n rows, covering as many domains as possible and at least ``min_npc`` game rows."""
    by_domain: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        by_domain.setdefault(has_tag(row, "domain") or "", []).append(row)
    picked = by_domain.get("game_npc", [])[:min_npc]
    rest = {d: rs[min_npc:] if d == "game_npc" else list(rs) for d, rs in by_domain.items()}
    while len(picked) < n and any(rest.values()):
        for bucket in rest.values():
            if bucket and len(picked) < n:
                picked.append(bucket.pop(0))
    return sorted(picked, key=lambda r: r["id"])


# ---------------------------------------------------------------- F


async def stage_fix(client: Client, lang: str, *, run: str, sets: Sequence[str], n: int = 16, seed: str = "pilot") -> dict[str, int]:
    d = stage_dir(run, lang)
    g_rows = list(read_jsonl(d / "g.jsonl"))
    # Without labeled G rows (GLM-only mode for English and German), the derived sets start from the public rows.
    base_rows = g_rows or public_base(lang)
    done: dict[str, int] = {}
    # Self-hosted teachers need no Ollama wait; otherwise the three-lab set waits for the Ollama plan.
    teachers = label.fix_teachers(lang)

    async def one(fset: str):
        drops: list[dict] = []
        if fset == "f1":
            rows, drops = await fixsets.build_f1(client, g_rows, n, seed, teachers)
        elif fset == "f2":
            rows, drops = await fixsets.build_f2(client, base_rows, n, seed, teachers)
        elif fset == "f3":
            rows, drops = await fixsets.build_f3(client, base_rows, n, seed, teachers)
        elif fset == "f4":
            rows = fixsets.build_f4(base_rows, n, seed)
        elif fset == "f7":
            rows = fixsets.build_f7(base_rows, await fixsets.padding_keys(client, lang), n, seed)
        elif fset == "f8":
            rows, drops = await fixsets.build_f8(client, base_rows, n, seed, teachers)
        elif fset == "f11":
            rows = fixsets.build_f11(base_rows, n)
        else:  # rule-based: f5 f6 f9 f10 f12, verified by the teacher ensemble
            rows, drops = await fixsets.build_rule_set(client, lang, fset, n, seed)
            rows, vdrops, per_tpl = await fixsets.verify_programmatic(client, rows, teachers=teachers, reject=fset != "f10")
            drops += vdrops
            (d / f"{fset}.templates.json").write_text(json.dumps(per_tpl, indent=1))
        rows, bad = post_check(rows, fset)
        write_jsonl(d / f"{fset}.jsonl", rows)
        _drops(run, lang, fset, drops + bad)
        done[fset] = len(rows)

    async def guarded(fset: str) -> None:
        try:
            await one(fset)
        except Exception as exc:  # one broken set must not stop the others
            _drops(run, lang, fset, [{"stage": fset, "reason": "crashed", "detail": f"{type(exc).__name__}: {exc}"[:500]}])
            done[fset] = 0

    await asyncio.gather(*[guarded(s) for s in sets])
    return done


def public_base(lang: str) -> list[dict]:
    """Public rows as a base for derived fix sets. Their dataset gold stands in as a single "gold" teacher so the
    builders can check which questions are clear-cut; derived rows keep the public source and license."""
    rows = []
    for path in sorted((data_dir() / "public" / lang).glob("*.sample.jsonl")):
        for row in read_jsonl(path):
            row = dict(row)
            row["teachers"] = {"gold": {"model": row["source"], "dist": {f: t["dist"] for f, t in row["targets"].items()}}}
            rows.append(row)
    return rows


def post_check(rows: list[dict], stage: str) -> tuple[list[dict], list[dict]]:
    """Every fix-set row gets the G-row checks too: PII scrub, schema, compile,
    prompt-copy guard and target shape. Teacher-written text can break any of them."""
    kept, drops = [], []
    for row in rows:
        row["state"], _ = scrub(row["state"], row["lang"])
        row["questions"], _ = scrub(row["questions"], row["lang"])
        problems = check_row(row, min_q=1) or check_targets(row)
        if problems:
            drops.append({"stage": stage, "id": row["id"], "reason": "post_check", "detail": "; ".join(problems)[:300]})
        else:
            kept.append(row)
    return kept, drops


async def prefetch_templates(client: Client, lang: str, sets: Sequence[str]) -> None:
    """Rule templates and the injection pool do not depend on G rows: fetch them early."""
    from .rules import SETS

    names = sorted({name for s in sets if s in SETS for name in SETS[s]})
    jobs = [fixsets.rule_templates(client, lang, name) for name in names]
    if "f8" in sets:
        jobs.append(fixsets.injection_pool(client, lang))
    await asyncio.gather(*jobs, return_exceptions=True)


# ---------------------------------------------------------------- R


async def stage_reason(client: Client, lang: str, *, run: str, plan: Mapping[str, int]) -> int:
    d = stage_dir(run, lang)
    by_set = {name: list(read_jsonl(d / f"{name}.jsonl")) for name in plan if (d / f"{name}.jsonl").exists()}
    plan = {name: k for name, k in plan.items() if name in by_set}
    up = {"zai"} | ({"hfr"} if HFR else set()) | ({"ollama"} if await client.available("ollama") else set()) | ({"kimi"} if client.usable("kimi") else set())
    picked = reasoning.pick_rows(by_set, plan, up)
    # Accepted traces from an earlier pass stay; only rows without one are traced (new G rows after a late label pass).
    done = {t["id"]: t for t in read_jsonl(d / "r.jsonl")} if (d / "r.jsonl").exists() else {}
    traces, logs = await reasoning.add_traces(client, [(r, w) for r, w in picked if r["id"] not in done])
    old_logs = list(read_jsonl(d / "r.log.jsonl")) if (d / "r.log.jsonl").exists() else []
    write_jsonl(d / "r.log.jsonl", old_logs + logs)
    write_jsonl(d / "r.jsonl", [*done.values(), *({"id": k, **v} for k, v in traces.items())])
    return len(done) + len(traces)


# ---------------------------------------------------------------- QA


def assemble(lang: str, *, run: str) -> tuple[list[dict], dict]:
    """Merge G, F and traces into final records; dedup; return rows and QA facts."""
    d = stage_dir(run, lang)
    traces = {t["id"]: t for t in read_jsonl(d / "r.jsonl")}
    rows = []
    fix_files = [p for p in d.glob("f*.jsonl") if p.stem[1:].isdigit()]  # f1..f12 only, never final.jsonl
    # n.jsonl: NPC rows from the village game's own requests (workspace/scripts/game-rows.py), when present.
    for path in [d / "g.jsonl", d / "n.jsonl"] + sorted(fix_files, key=lambda p: int(p.stem[1:])):
        for row in read_jsonl(path):
            if row["id"] in traces:
                row["reasoning"] = traces[row["id"]]["trace"]
                row["tags"] = row["tags"] + [f"reasoner:{traces[row['id']]['model']}"]
            # Signal the eval client to show the unknown option when the gold
            # distribution puts mass on it (F1 rows where the answer is unknown).
            if any(t.get("dist", {}).get("unknown", 0) > 0 for t in (row.get("targets") or {}).values()):
                row["abstain"] = True
            rows.append(row)
    final, failed = [], []
    for r in (qa.strip(r) for r in rows):  # the last gate: rows failing any row check never ship
        problems = qa.schema_problems(r)
        (failed if problems else final).append({"stage": "qa", "id": r["id"], "reason": "row_check", "detail": "; ".join(problems)[:300]} if problems else r)
    _drops(run, lang, "qa", failed)
    kept, dups = qa.dedup(final)
    write_jsonl(d / "final.jsonl", kept)
    return kept, {"dups": dups, "gates": qa.gates(kept)}


def drop_summary(run: str, lang: str) -> Counter:
    c: Counter = Counter()
    for path in stage_dir(run, lang).glob("drops.*.jsonl"):
        for rec in read_jsonl(path):
            c[f"{rec.get('stage')}: {rec.get('reason')}"] += 1
    return c


def sample_examples(rows: Sequence[Mapping], k: int = 12, seed: str = "report") -> list[Mapping]:
    """Examples across sets: G first (including game rows), then each fix set, traced rows included."""
    r = rng("examples", seed, len(rows))
    by_set: dict[str, list] = {}
    for row in rows:
        by_set.setdefault(qa.set_of(row), []).append(row)
    order = ["G", "G", "G", "G", "F1", "F2", "F3", "F5", "F6", "F8", "F10", "F12"]
    picked: list[Mapping] = []
    traced = [row for row in rows if row.get("reasoning")]
    npc = [row for row in by_set.get("G", []) if has_tag(row, "domain") == "game_npc"]
    if npc:
        picked.append(r.choice(npc))
    for name in order:
        pool = [row for row in by_set.get(name, []) if row not in picked]
        prefer = [row for row in pool if row in traced]
        if pool and len(picked) < k:
            picked.append(r.choice(prefer or pool) if name in {"F5", "F6", "F12"} else r.choice(pool))
    return picked[:k]


def fmt_counter(c: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0])))


# ---------------------------------------------------------------- report


def _verify_stats(rows: Sequence[Mapping]) -> str:
    """How often the teacher majority agrees with each fix set's gold."""
    by_set: dict[str, list[bool]] = {}
    for row in rows:
        s = qa.set_of(row)
        if s == "G":  # for G rows the teachers define the gold, so agreement is circular
            continue
        teachers = row.get("teachers") or {}
        for field, t in row["targets"].items():
            votes = [max(d["dist"][field], key=d["dist"][field].get) for d in teachers.values() if field in d.get("dist", {})]
            if len(votes) >= 2:
                by_set.setdefault(s, []).append(sum(v == t["label"] for v in votes) >= 2)
    return "\n".join(f"| {s} | {sum(v)}/{len(v)} ({qa.pct(sum(v), len(v))}) |" for s, v in sorted(by_set.items()))


def _injection_stats(rows: Sequence[Mapping]) -> str:
    flips = total = 0
    for row in rows:
        if qa.set_of(row) != "F8":
            continue
        field = has_tag(row, "inject_field")
        for d in (row.get("teachers") or {}).values():
            if field in d.get("dist", {}):
                total += 1
                flips += max(d["dist"][field], key=d["dist"][field].get) != row["targets"][field]["label"]
    return f"{flips}/{total} teacher answers moved off the gold on the injected field"


def write_report(lang: str, *, run: str, kimi: Mapping | None = None) -> Path:
    d = stage_dir(run, lang)
    rows = list(read_jsonl(d / "final.jsonl"))
    gen = list(read_jsonl(d / "g.gen.jsonl"))
    valid = list(read_jsonl(d / "g.valid.jsonl"))
    edited = list(read_jsonl(d / "g.edit.jsonl"))
    g_rows = [r for r in rows if qa.set_of(r) == "G"]
    agree = label.agreement_stats(g_rows)
    st = qa.stats(rows)
    gates = qa.gates(rows)
    tokens: dict[str, list[int]] = {}
    for row in rows:
        tokens.setdefault(qa.set_of(row), []).append(qa.token_length(row))
    usage = qa.api_usage(run)
    r_logs = list(read_jsonl(d / "r.log.jsonl"))
    adherence = generate.card_adherence(edited) if edited else {}
    lines = [f"# JevAlt data pilot: {lang}", "", f"Run `{run}`. Rows are in `{d}/final.jsonl`; every stage file sits next to it.", ""]
    lines += ["## Counts", "", "| Stage or set | Rows |", "|---|---|",
              f"| G cards requested / rows written | {len(json.loads((d / 'cards.json').read_text())) if (d / 'cards.json').exists() else '?'} / {len(gen)} |",
              f"| G after schema, leak and PII checks | {len(valid)} |",
              f"| G after native edit and language check | {len(edited)} |",
              f"| G labeled and kept (2 of 3 agreement, capped at {G_TARGET}) | {len(g_rows)} |"]
    for s, n in sorted(st["by_set"].items(), key=lambda kv: (kv[0][0], int(kv[0][1:]) if kv[0][1:].isdigit() else 0)):
        if s != "G":
            lines.append(f"| {s} | {n} |")
    lines += [f"| Reasoning traces accepted | {sum(1 for r in rows if r.get('reasoning'))} of {len(r_logs)} rows tried |",
              f"| Final rows after dedup | {len(rows)} |", ""]
    lines += ["## Drop reasons", "", "| Stage: reason | Count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in drop_summary(run, lang).most_common()] + [""]
    n_q = agree["questions"] or 1
    lines += ["## Teacher agreement (G, kept questions)", "",
              "- Teacher sets (rows): " + ", ".join(f"{k} {v}" for k, v in agree["sets"].items())
              + ". Rows labeled by glm+k3n are the reduced pilot fallback (teachers:2, both must agree) used when the full set was unavailable.",
              f"- Questions: {agree['questions']}; all teachers of the row share the argmax on {qa.pct(agree['unanimous'], n_q)}; "
              f"trivial (every teacher put at least 0.95 on the same option): {qa.pct(agree['trivial'], n_q)}.",
              "- Pairwise argmax agreement: " + ", ".join(f"{k} {qa.pct(v, agree['pair_n'][k])} (n={agree['pair_n'][k]})" for k, v in agree["pair"].items()) + ".",
              f"- Mean pairwise total variation distance: {sum(agree['tv']) / max(1, len(agree['tv'])):.3f}.",
              f"- Writer's own argmax matched the card target on {adherence.get('target_hit', 0)} of {adherence.get('targets', 0)} targeted questions; "
              f"option or level counts matched the card on {adherence.get('count_match', 0)} of {adherence.get('questions', 0)}; "
              f"state length / card length ratio: {qa.describe([x * 100 for x in adherence.get('len_ratio', [])])} (percent).", ""]
    lines += ["## Fix-set checks by the teacher ensemble", "", "| Set | Teacher majority agrees with gold |", "|---|---|", _verify_stats(rows), "",
              f"F8 injection: {_injection_stats(rows)}.", ""]
    edits = [r for r in edited if r.get("edit")]
    if lang == "en":
        lines += ["## Native edit pass", "", "Not applicable: DATA.md runs the native edit pass on Turkish and German rows only.", ""]
    else:
        lines += ["## Native edit pass", "", f"{len(edits)} of {len(edited)} rows went to the editor (a 10 percent sample plus every row the language check flagged): "
                  + fmt_counter(Counter(r['edit']['status'] for r in edits))
                  + ". 'deferred' means the editor was unavailable at the time; those rows kept the writer's text.", ""]
    for r in [r for r in edits if r["edit"]["status"] == "changed"][:4]:
        lines.append(f"- `{r['id']}` by {r['edit']['editor']}: " + "; ".join(str(c) for c in r["edit"].get("changes", [])[:4]))
    lines += ["", "## Language check", "",
              f"- At validation: {sum(r['checks']['lang']['ok'] for r in valid)}/{len(valid)} G rows passed first time "
              f"({sum(bool(r['checks']['lang'].get('folded')) for r in valid)} ASCII-folded).",
              f"- Final rows: {gates['language']['pass']}/{gates['language']['checked']} pass (fastText lid.176, plus the diacritics check); "
              f"lowest top-language probability {gates['language']['min_prob']}.", ""]
    for fail in gates["language"]["fails"][:5]:
        lines.append(f"  - {fail}")
    lines += ["## Gates", "", f"- Schema and compile failures: {len(gates['schema_fail'])}", f"- Rows with PII pattern hits: {len(gates['pii_rows'])}",
              "- Duplicates removed: " + (fmt_counter(Counter(x["kind"] for x in assemble_dups(run, lang))) or "none")
              + " ('identical_row': the same state, question and target reached two fix sets, usually an F2 original and an F3 Noul twin built from the same G question).", ""]
    lines += ["## Token lengths (Intern-Decision-4B tokenizer, full rendered chat)", "", "| Set | Tokens |", "|---|---|"]
    lines += [f"| {s} | {qa.describe(v)} |" for s, v in sorted(tokens.items())] + [""]
    lines += ["## Label and shape distribution", "",
              f"- Question types: {fmt_counter(st['qtypes'])}", f"- Questions per row: {fmt_counter(st['questions_per_row'])}",
              f"- Choice option counts: {fmt_counter(st['choice_options'])}", f"- Score levels: {fmt_counter(st['score_levels'])}",
              f"- Noul gold = yes, per domain or set: {fmt_counter(st['noul_yes'])}",
              "- Choice gold position (0 = first option, 1 = last): " + "; ".join(f"{s}: {fmt_counter(c)}" for s, c in st["gold_position"].items()),
              "- Shares above 60 percent: " + ("; ".join(qa.label_balance(rows)) or "none"), ""]
    lines += ["## API use in this run", "", "| Provider/model | Calls | Prompt tokens | Completion tokens | Seconds |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['calls']} | {v['prompt_tokens']} | {v['completion_tokens']} | {v['seconds']:.0f} |" for k, v in sorted(usage.items())]
    lines += ["", "Calls are cumulative for the whole pilot run tag (all three languages, every restart; cached replies are not counted). "
              "Quota events: " + (fmt_counter(qa.quota_events(run)) or "none") + "."]
    if kimi:
        lines += [f"Kimi Code usage meter: before {kimi.get('before')}, after {kimi.get('after')}."]
    lines += ["", "## Examples", ""]
    for i, row in enumerate(sample_examples(rows), 1):
        lines += [qa.render_row(row, i), ""]
    critique = d / "critique.md"
    lines += ["## Critique", "", critique.read_text() if critique.exists() else "(pending)", ""]
    out = data_dir() / "reports" / f"{run}-{lang}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def assemble_dups(run: str, lang: str) -> list:
    path = stage_dir(run, lang) / "dups.json"
    return json.loads(path.read_text()) if path.exists() else []
