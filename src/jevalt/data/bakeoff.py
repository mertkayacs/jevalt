"""Native writer bake-off: TR and DE writers rated blind by nativefeel.

Usage: python -m jevalt.data bakeoff --lang tr,de --run pilot
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from . import generate
from .common import LANG_NAMES, data_dir, text_of
from .providers import TEACHERS, Client, _load_keys

# Writers in the bake-off (provider keys from TEACHERS).
BAKEOFF_WRITERS = {
    "tr": ["glmt", "k3", "dst", "mistral", "gemma"],
    "de": ["glmt", "k3", "dst", "mistral", "gemma"],
}
# Per writer+lang, raters from different model families.
RATERS: dict[str, dict[str, list[str]]] = {
    "tr": {
        "glmt": ["mistral", "k3n"],
        "k3": ["mistral", "ds"],
        "dst": ["mistral", "k3n"],
        "mistral": ["k3n", "ds"],
        "gemma": ["mistral", "k3n"],
    },
    "de": {
        "glmt": ["mistral", "ds"],
        "k3": ["mistral", "ds"],
        "dst": ["mistral", "k3n"],
        "mistral": ["ds", "k3n"],
        "gemma": ["mistral", "ds"],
    },
}

CARDS_PER_LANG = 12


def extract_texts(rows: list[dict]) -> list[str]:
    """Concatenate state and every question's instructions for rating."""
    texts = []
    for row in rows:
        parts = [text_of(row.get("state", ""))]
        for q in (row.get("questions") or {}).values():
            parts.append(text_of(q.get("instructions", "")))
        texts.append("\n".join(parts))
    return texts


async def write_with_writer(client: Client, lang: str, writer: str, cards: list[dict]) -> tuple[list[dict], list[dict]]:
    """Write cards with a specific writer."""
    batches = [cards[i : i + generate.ROWS_PER_CALL] for i in range(0, len(cards), generate.ROWS_PER_CALL)]
    rows, drops = [], []
    for b, batch in enumerate(batches):
        got, dropped = await generate.write_batch(client, lang, writer, batch, b)
        rows += got
        drops += dropped
    rows.sort(key=lambda r: r["id"])
    return rows, drops


def _rate_sync(texts: list[str], lang: str, base_url: str, model: str, key_env: str) -> dict:
    """Synchronous nativefeel rating, called from a thread."""
    from jevoss.nativefeel import rate

    return rate(texts, lang, base_url=base_url, model=model, key_env=key_env, delay=0.3)


async def bakeoff(lang: str, run: str, concurrency: int = 2, keys: dict | None = None) -> dict:
    """Run the bake-off for one language and return per-writer results."""
    d = data_dir() / run / lang
    cards = json.loads((d / "cards.json").read_text())

    # Pick 12 diverse cards.
    r = generate.rng("bakeoff", "pilot", lang)
    by_domain: dict[str, list[int]] = {}
    for i, c in enumerate(cards):
        by_domain.setdefault(c["domain"], []).append(i)
    picked: list[int] = []
    for indices in by_domain.values():
        if indices:
            picked.append(r.choice(indices))
    # Fill to CARDS_PER_LANG with random draws.
    need = CARDS_PER_LANG - len(picked)
    if need > 0:
        rest = [i for i in range(len(cards)) if i not in picked]
        picked += r.sample(rest, min(need, len(rest)))
    picked = sorted(picked[:CARDS_PER_LANG])
    bc = [cards[i] for i in picked]
    (d / "bakeoff_cards.json").write_text(json.dumps(bc, ensure_ascii=False, indent=1))

    writers = BAKEOFF_WRITERS[lang]
    results: dict[str, dict] = {}

    # Set API keys in the environment for the nativefeel module (uses urllib + env vars).
    key_map = keys or _load_keys()
    env_map = {"kimi": "KIMI_API_KEY", "zai": "ZAI_API_KEY", "ollama": "OLLAMA_API_KEY"}
    for prov, env_key in env_map.items():
        if key_map.get(prov):
            os.environ.setdefault(env_key, key_map[prov])

    async with Client(run=f"{run}-bakeoff", concurrency=concurrency, wait_on_quota=True) as client:
        for writer_key in writers:
            t = TEACHERS[writer_key]
            print(f"  {lang} {writer_key} ({t.id})...", flush=True)
            try:
                rows, drops = await write_with_writer(client, lang, writer_key, bc)
                texts = extract_texts(rows)
                print(f"    wrote {len(rows)}, drops {len(drops)}", flush=True)
            except Exception as exc:
                print(f"    FAILED: {exc}", flush=True)
                results[writer_key] = {"model": t.id, "error": str(exc), "rows": 0, "scores": {}}
                continue

            results[writer_key] = {"model": t.id, "rows": len(rows), "drops": len(drops), "scores": {}}

            rater_keys = RATERS.get(lang, {}).get(writer_key, [])
            for rater_key in rater_keys:
                if results[writer_key].get("error"):
                    continue
                rt = TEACHERS[rater_key]
                print(f"    rating with {rater_key} ({rt.id})...", flush=True)
                try:
                    score = await asyncio.to_thread(
                        _rate_sync, texts, lang, rt.base_url, rt.model, env_map[rt.provider]
                    )
                    results[writer_key]["scores"][rater_key] = score
                    print(f"      mean={score['mean']:.2f} n={score['n']} dist={score['distribution']}", flush=True)
                except Exception as exc:
                    print(f"      rating FAILED: {exc}", flush=True)
                    results[writer_key]["scores"][rater_key] = {"error": str(exc)}

    return results


def write_report(lang: str, results: dict, out_dir: Path) -> None:
    """Write the bake-off report section for one language."""
    lines = [f"# Native writer bake-off: {lang}", "",
             f"12 cards per writer. Two raters per writer, each from a different model family.",
             f"RUBRIC_VERSION: native-feel-v1. Score: 1=clearly translated/wrong, 3=understandable but stiff, 5=native.",
             ""]
    lines += ["## Per-writer results", ""]
    lines += ["| Writer | Model | Rows | Rater 1 | Rater 2 | Mean |",
              "|---|---|---|---|---|---|"]
    combined: dict[str, float] = {}
    for writer_key in sorted(results):
        r = results[writer_key]
        if r.get("error"):
            lines.append(f"| {writer_key} | {r.get('model', '?')} | FAILED: {r['error']} | | | |")
            continue
        means = []
        cells = []
        for rk in sorted(r.get("scores", {})):
            sc = r["scores"][rk]
            if "error" in sc:
                cells.append(f"{rk}: err")
            else:
                m = sc["mean"]
                means.append(m)
                cells.append(f"{rk}: {m:.2f}")
        while len(cells) < 2:
            cells.append("")
        comb = round(sum(means) / len(means), 2) if means else 0
        t = TEACHERS[writer_key]
        lines.append(f"| {writer_key} ({t.id}) | | {r['rows']} | {cells[0]} | {cells[1]} | **{comb:.2f}** |")
        combined[writer_key] = comb

    lines += [""]
    if combined:
        winner = max(combined, key=combined.get)
        tw = TEACHERS[winner]
        lines += [f"**Winner: {winner} ({tw.id})** with combined mean {combined[winner]:.2f}", ""]

    lines += ["## Typical issues per writer", ""]
    for writer_key in sorted(results):
        r = results[writer_key]
        if r.get("error"):
            continue
        all_issues = []
        for rk, sc in sorted(r.get("scores", {}).items()):
            all_issues += sc.get("issues", [])
        if all_issues:
            model_name = TEACHERS[writer_key].id if writer_key in TEACHERS else writer_key
            lines += [f"### {writer_key} ({model_name})", ""]
            for issue in all_issues[:8]:
                lines.append(f"- {issue}")
            lines += [""]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bakeoff.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    lines_path = out_dir / "bakeoff.md"
    lines_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {lines_path}", flush=True)


async def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="tr,de")
    p.add_argument("--run", default="pilot")
    p.add_argument("--concurrency", type=int, default=2)
    args = p.parse_args()

    langs = args.lang.split(",")
    out_dir = data_dir() / "reports"
    for lang in langs:
        print(f"Bake-off: {lang}", flush=True)
        results = await bakeoff(lang, args.run, args.concurrency)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"native-bakeoff-{lang}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"  raw: {out_dir / f'native-bakeoff-{lang}.json'}", flush=True)
    # Combined report.
    combined = {}
    for lang in langs:
        path = out_dir / f"native-bakeoff-{lang}.json"
        if path.exists():
            combined[lang] = json.loads(path.read_text())
    if combined:
        write_combined_report(combined, out_dir)


def write_combined_report(all_results: dict, out_dir: Path) -> None:
    lines = ["# Native writer bake-off", "",
             f"12 cards per language, per writer. Two raters per writer from different model families.",
             f"RUBRIC_VERSION: native-feel-v1. 1=clearly translated/wrong, 3=understandable but stiff, 5=native.",
             ""]
    for lang, results in sorted(all_results.items()):
        lines += [f"## {LANG_NAMES[lang]} ({lang})", ""]
        lines += ["| Writer | Rows | Rater 1 | Rater 2 | Mean |",
                  "|---|---|---|---|---|"]
        combined = {}
        for writer_key in sorted(results):
            r = results[writer_key]
            if r.get("error"):
                lines.append(f"| {writer_key} | FAILED: {r['error']} | | | |")
                continue
            means = []
            cells = []
            for rk in sorted(r.get("scores", {})):
                sc = r["scores"][rk]
                if "error" in sc:
                    cells.append(f"{rk}: err")
                else:
                    m = sc["mean"]
                    means.append(m)
                    cells.append(f"{rk}: {m:.2f}")
            while len(cells) < 2:
                cells.append("")
            comb = round(sum(means) / len(means), 2) if means else 0
            t = TEACHERS[writer_key]
            lines.append(f"| {writer_key} ({t.id}) | {r['rows']} | {cells[0]} | {cells[1]} | **{comb:.2f}** |")
            combined[writer_key] = comb
        lines += [""]
        if combined:
            winner = max(combined, key=combined.get)
            tw = TEACHERS[winner]
            lines += [f"**Winner for {lang}: {winner} ({tw.id})** with combined mean {combined[winner]:.2f}", ""]

        for writer_key in sorted(results):
            r = results[writer_key]
            if r.get("error"):
                continue
            all_issues = []
            for rk, sc in sorted(r.get("scores", {}).items()):
                all_issues += sc.get("issues", [])
            if all_issues:
                model_name = TEACHERS[writer_key].id if writer_key in TEACHERS else writer_key
                lines += [f"### {lang}: {writer_key} ({model_name})", ""]
                for issue in all_issues[:6]:
                    lines.append(f"- {issue}")
                lines += [""]

    out = out_dir / "native-bakeoff.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"  combined: {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())