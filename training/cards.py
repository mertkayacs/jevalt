"""Build Hugging Face model cards from measured results.

Every number in a card comes from a results file produced by training/jobs/.
Nothing is typed by hand, so a card can never drift from the evidence.
"""

from __future__ import annotations

import argparse
import json

REPO = "https://github.com/mertkayacs/jevalt"
PLAYGROUND = "https://github.com/mertkayacs/jevoss"
SPACE = "https://huggingface.co/spaces/mertkayacs/JevAlt"

BIBTEX = """```bibtex
@software{kaya2026jevalt,
  author = {Mert Kaya},
  title = {JevAlt: Open Decision Models with the Jev API},
  year = {2026},
  license = {Apache-2.0},
  url = {https://github.com/mertkayacs/jevalt}
}
```"""

MODELS = {
    "en": {"name": "Deem-4B", "repo": "Deem-4B", "language": "English", "native": None},
    "tr": {
        "name": "Karar-4B",
        "repo": "Karar-4B",
        "language": "Turkish",
        "native": (
            "<details>\n<summary><b>Türkçe özet</b></summary>\n\n"
            "Karar-4B, Türkçe yazılmış kararlar üzerinde eğitilmiş açık bir modeldir. Bir durum ve "
            "Choice, Score ya da Noul sorusu gönderirsiniz; her seçenek için kalibre edilmiş olasılık "
            "alırsınız. İsterseniz model önce kısa bir Türkçe gerekçe yazar, sonra karar verir. "
            "Q4_K_M sürümü yaklaşık 3 GB RAM ile kendi bilgisayarınızda çalışır, veriler dışarı çıkmaz.\n\n</details>\n"
        ),
    },
    "de": {
        "name": "Wähler-4B",
        "repo": "Wahler-4B",
        "language": "German",
        "native": (
            "<details>\n<summary><b>Deutsche Zusammenfassung</b></summary>\n\n"
            "Wähler-4B ist ein offenes Entscheidungsmodell, das auf deutschsprachigen Entscheidungen "
            "trainiert wurde. Sie senden einen Zustand und Choice-, Score- oder Noul-Fragen und erhalten "
            "für jede Option eine kalibrierte Wahrscheinlichkeit. Auf Wunsch schreibt das Modell zuerst "
            "eine kurze Begründung auf Deutsch und entscheidet dann. Die Q4_K_M-Version läuft mit etwa 3 GB RAM "
            "auf dem eigenen Rechner, die Daten bleiben lokal.\n\n</details>\n"
        ),
    },
}

PROBE_NAMES = {
    "permutation": ("Option order (flip rate)", "pct"),
    "injection": ("Prompt injection (attack success)", "pct"),
    "distractors": ("Distractors (accuracy lost)", "pts"),
    "noul-vs-choice": ("Noul vs Choice (mean gap)", "dec"),
    "determinism": ("Determinism (largest change)", "dec"),
}


def fmt(x, digits=3):
    return "-" if x is None else f"{x:.{digits}f}"


def _pct(x):
    return "-" if x is None else f"{x * 100:.1f}%"


def _pts(x):
    return "-" if x is None else f"{x * 100:.1f} pts"


def _dec(x):
    return "-" if x is None else f"{x:.3f}"


def _probe_fmt(value, probe_id):
    _, kind = PROBE_NAMES.get(probe_id, (probe_id, "dec"))
    if kind == "pct":
        return _pct(value)
    if kind == "pts":
        return _pts(value)
    return _dec(value)


def results_table(ours: dict, base: dict, suites: list[str], label_ours: str, label_base: str) -> str:
    rows = [f"| Suite | Decisions | {label_base} acc / Brier | {label_ours} acc / Brier | Δ acc | Δ Brier |", "|---|---|---|---|---|---|"]
    by = lambda res: {(r["suite"], r.get("reasoning", "off")): r["overall"] for r in res["results"]}  # noqa: E731
    o, b = by(ours), by(base)
    for suite in suites:
        key = (suite, "off")
        if key not in o or key not in b:
            continue
        x, y = b[key], o[key]
        rows.append(
            f"| {suite} | {y['n']} | {fmt(x['accuracy'])} / {fmt(x['brier'])} | {fmt(y['accuracy'])} / {fmt(y['brier'])} "
            f"| {y['accuracy'] - x['accuracy']:+.3f} | {y['brier'] - x['brier']:+.3f} |"
        )
    return "\n".join(rows)


def _probe_table(probe_results: dict | None, probe_baseline: dict | None, label_ours: str, label_base: str) -> str:
    if not probe_results and not probe_baseline:
        return ""
    ours = probe_results.get("probes", {}) if probe_results else {}
    base = probe_baseline.get("probes", {}) if probe_baseline else {}
    rows = [
        f"| Probe | {label_ours} | {label_base} |",
        "|---|---|---|",
    ]
    for pid, (display, _) in PROBE_NAMES.items():
        o_val = ours.get(pid, {}).get("value") if ours else None
        b_val = base.get(pid, {}).get("value") if base else None
        rows.append(f"| {display} | {_probe_fmt(o_val, pid)} | {_probe_fmt(b_val, pid)} |")
    return "\n".join(rows)


SUITE_NAMES = {
    "test-en": "English held-out test",
    "test-tr": "Turkish held-out test",
    "test-de": "German held-out test",
    "typed-decisions": "typed-decisions",
    "jevbench-hard": "JevBench-hard",
    "turkish-mmlu": "TurkishMMLU",
    "germeval2017": "GermEval 2017",
    "gnad10": "10kGNAD",
}
COMPARE_ROWS = {
    "en": ["test-en", "typed-decisions", "jevbench-hard", "test-tr", "test-de"],
    "tr": ["test-tr", "turkish-mmlu", "test-en", "typed-decisions", "jevbench-hard"],
    "de": ["test-de", "germeval2017", "gnad10", "test-en", "typed-decisions", "jevbench-hard"],
}
COMPARE_PROBES = [  # probe id, row label, format, short name for sentences; lower is better for all three
    ("injection", "An instruction hidden in the state flips the answer", "pct", "hidden instructions"),
    ("permutation", "Reordering the options flips the answer", "pct", "option order"),
    ("distractors-600w", "Accuracy lost to 600 words of padding", "pts", "long padding"),
]
OTHER_MODELS = ["intern-decision-4b", "kev-4b", "laya"]
OTHER_NAMES = {"intern-decision-4b": "Intern-Decision-4B", "kev-4b": "Kev-4B", "laya": "Laya"}
COMPARE_FILES = "https://huggingface.co/datasets/mertkayacs/jevalt-bench/tree/main/results/comparison"
VIDEOS = "https://huggingface.co/datasets/mertkayacs/emberwick-videos"
GIF_CAPTION = "Emberwick: every villager asks Deem-4B what to do next. Nothing is scripted."
FILM_CAPTION = "The one-minute film, sound on: three mistakes small decision models make and how JevAlt fixes each one."
# Turkish and German pages offer the clip and the film in their own language next to the English ones.
NATIVE = {
    "tr": ("Türkçe sürüm", "Emberwick Türkçe: her köylü bir sonraki adımını Karar-4B'ye soruyor.", "Bir dakikalık film, sesi açın.",
           "GIF (4K)", "hafif GIF"),
    "de": ("Deutsche Version", "Emberwick auf Deutsch: Jeder Dorfbewohner fragt Wähler-4B, was als Nächstes zu tun ist.",
           "Der einminütige Film, mit Ton.", "GIF (4K)", "leichtes GIF"),
}


def _video(lang: str) -> str:
    film = f"{VIDEOS}/resolve/main/film/jevalt-film-{lang}"
    return f'<video controls playsinline preload="none" poster="{film}.jpg" src="{film}-1080p.mp4"></video>'


def native_block(lang: str) -> str:
    """The Turkish or German clip and film behind a toggle, so the page opens in English."""
    if lang not in NATIVE:
        return ""
    label, caption, film, gif4k, gif960 = NATIVE[lang]
    g = f"{VIDEOS}/resolve/main/gifs"
    return (f"<details>\n<summary><b>{label}</b></summary>\n\n{_video(lang)}\n\n*{film}* {caption} "
            f"[{gif4k}]({g}/emberwick-{lang}.gif) · [{gif960}]({g}/960/emberwick-{lang}.gif)\n\n</details>\n\n")


def media_block(lang: str) -> str:
    """The English village clip on top (4K GIF), the English film, the clip and film in the model's own
    language behind a toggle."""
    return (f"![{GIF_CAPTION}]({VIDEOS}/resolve/main/gifs/emberwick-en.gif)\n\n"
            f"*{GIF_CAPTION}* [More clips]({VIDEOS})\n\n"
            f"{_video('en')}\n\n*{FILM_CAPTION}*\n\n"
            f"{native_block(lang)}")


def _bold_best(values: list[float | None], cells: list[str], higher: bool) -> list[str]:
    """Bold every cell that ties for the best value in its row."""
    known = [v for v in values if v is not None]
    if not known:
        return cells
    best = max(known) if higher else min(known)
    return [f"**{c}**" if v is not None and abs(v - best) < 1e-9 else c for v, c in zip(values, cells)]


def comparison_section(lang: str, comp: dict | None) -> str:
    """Kev-4B and Laya next to this model and the start checkpoint, from results/comparison.json."""
    if not comp:
        return ""
    key = {"en": "deem-4b", "tr": "karar-4b", "de": "wahler-4b"}[lang]
    name = MODELS[lang]["name"]
    cols = [key] + OTHER_MODELS
    names = [name] + [OTHER_NAMES[m] for m in OTHER_MODELS]
    lines = ["| Accuracy, higher is better | " + " | ".join(names) + " |", "|---" * (len(cols) + 1) + "|"]
    rows = 0
    wins = {"kev-4b": 0, "laya": 0}
    behind = {"kev-4b": [], "laya": []}
    for suite in COMPARE_ROWS[lang]:
        row = comp["suites"].get(suite, {})
        if any(m not in row for m in cols):
            continue
        vals = [row[m]["accuracy"] for m in cols]
        cells = _bold_best(vals, [_pct(v) for v in vals], higher=True)
        lines.append(f"| {SUITE_NAMES[suite]} ({row[key]['n']:,} decisions) | " + " | ".join(cells) + " |")
        rows += 1
        for other in wins:
            if vals[0] > row[other]["accuracy"]:
                wins[other] += 1
            elif vals[0] < row[other]["accuracy"]:
                behind[other].append(SUITE_NAMES[suite])
    lines.append("| **Lower is better** | | | | |")
    for probe, label, kind, short in COMPARE_PROBES:
        row = comp["probes"].get(probe, {})
        if any(m not in row for m in cols):
            continue
        vals = [row[m] for m in cols]
        cells = _bold_best(vals, [_pct(v) if kind == "pct" else _pts(v) for v in vals], higher=False)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
        rows += 1
        for other in wins:
            if vals[0] < row[other]:
                wins[other] += 1
            elif vals[0] > row[other]:
                behind[other].append(short)

    weak = ["| Weakness, held-out rows | JevAlt | Intern-Decision-4B | Kev-4B | Laya |", "|---|---|---|---|---|"]
    for row in comp["weaknesses"].values():
        if any(m not in row for m in ["jevalt"] + OTHER_MODELS):
            continue
        vals = [row[m]["accuracy"] for m in ["jevalt"] + OTHER_MODELS]
        cells = _bold_best(vals, [_pct(v) for v in vals], higher=True)
        weak.append(f"| {row['label']} ({row['jevalt']['n']} decisions) | " + " | ".join(cells) + " |")

    def where(items):
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    both = [x for x in behind["kev-4b"] if x in behind["laya"]]
    only_kev = [x for x in behind["kev-4b"] if x not in both]
    only_laya = [x for x in behind["laya"] if x not in both]
    others = []
    if both:
        others.append(f"Kev-4B and Laya both do better than {name} on {where(both)}")
    if only_kev:
        others.append(f"Kev-4B does better on {where(only_kev)}")
    if only_laya:
        others.append(f"Laya does better on {where(only_laya)}")
    others.append("Laya is far smaller and faster (322M to 421M parameters, about 33 ms per request on a T4 GPU, by its card)")

    return f"""## Compared with Kev-4B and Laya

[Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) and [Laya](https://huggingface.co/convaiinnovations/laya) are open models that answer the same Jev requests. All four models ran through one client on the same items; Kev-4B and Laya ran on their own servers with their shipped calibration. {name} beats Kev-4B on {wins['kev-4b']} of these {rows} rows and Laya on {wins['laya']}. Bold marks the best score in each row.

{chr(10).join(lines)}

The held-out tests come from the same pipeline as JevAlt's training rows, so they favour JevAlt; JevAlt also trained on the typed-decisions train split (scores use its test split). Kev-4B and Laya received every row in the shapes the TypeSafe docs use (Noul criteria keyed `true`/`false`, Score levels as a list). Rows that need the `unknown` option are left out of every column, because Kev-4B and Laya do not offer it. Probes use 100 typed-decisions items.

**Problems these models share.** Each held-out row below tests one weakness. The JevAlt column uses each language's own model (Deem-4B on English rows, Karar-4B on Turkish, Wähler-4B on German).

{chr(10).join(weak)}

Where the others do better: {'; '.join(others)}. Result files and every decision: [results/comparison]({COMPARE_FILES}).

"""


JEV_NOTES = "https://docs.typesafe.ai/model-jaggedness/jev-1.13"
JEV_AUDIT = "https://github.com/jujumilk3/jev-calibration-audit/blob/main/FINDINGS.md"
CHART_ALT = {
    "langs": "Accuracy on English, Turkish and German decisions and on typed-decisions: JevAlt, Intern-Decision-4B, Kev-4B and Laya",
    "fixes": "Hidden instructions, option order, long policies and negated questions: JevAlt against Intern-Decision-4B, Kev-4B and Laya",
    "jev": "What Jev 1.13 lacks and JevAlt has: thinking when unsure, an unknown answer, coverage sets, native Turkish and German, open weights, repeatable answers",
}


def charts_section(lang: str, comp: dict, notes: str = "") -> str:
    """Results as three charts in the model's language (assets/<chart>.png in the model repo), one line on
    method, where the other models do better, and the significance notes folded away."""
    key = {"en": "deem-4b", "tr": "karar-4b", "de": "wahler-4b"}[lang]
    name, repo = MODELS[lang]["name"], MODELS[lang]["repo"]
    behind = {"kev-4b": [], "laya": []}
    for suite in COMPARE_ROWS[lang]:
        row = comp["suites"].get(suite, {})
        for other in behind:
            if key in row and other in row and row[other]["accuracy"] > row[key]["accuracy"]:
                behind[other].append(SUITE_NAMES[suite])
    for probe, _, _, short in COMPARE_PROBES:
        row = comp["probes"].get(probe, {})
        for other in behind:
            if key in row and other in row and row[other] < row[key]:
                behind[other].append(short)

    def where(items):
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    both = [x for x in behind["kev-4b"] if x in behind["laya"]]
    others = []
    if both:
        others.append(f"Kev-4B and Laya both do better on {where(both)}")
    if only := [x for x in behind["kev-4b"] if x not in both]:
        others.append(f"Kev-4B does better on {where(only)}")
    if only := [x for x in behind["laya"] if x not in both]:
        others.append(f"Laya does better on {where(only)}")
    others.append("Laya is far smaller and faster")
    chart = lambda c: f"![{CHART_ALT[c]}](https://huggingface.co/mertkayacs/{repo}/resolve/main/assets/{c}.png)"  # noqa: E731
    charts = "\n\n".join(chart(c) for c in ("langs", "fixes"))
    folded = f"\n<details>\n<summary>Significance and caveats</summary>\n\n{chart('jev')}\n\n{notes}\n\n</details>\n"
    return f"""## Results

{charts}

Same items and client for every model, each as shipped: [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) r10 and [Laya](https://huggingface.co/convaiinnovations/laya) 0.3.22 on their own servers with their own calibration. Jev 1.13 rows come from [TypeSafe's notes]({JEV_NOTES}) and an [independent audit]({JEV_AUDIT}). The held-out tests come from JevAlt's own data pipeline, so they favour JevAlt. {'; '.join(others)}. Every number and every decision: [results/comparison]({COMPARE_FILES}).
{folded}
"""


def _training_section(training: dict | None) -> str:
    if not training:
        return ""
    lines = ["<details>", "<summary><b>How it was trained</b></summary>", ""]
    base = training.get("base_model", "internlm/Intern-Decision-4B")
    rank = training.get("lora_rank")
    alpha = training.get("lora_alpha")
    epochs = training.get("epochs")
    teachers = training.get("teachers", [])
    lines.append(f"- Base: {base} (Qwen3.5-4B)")
    if rank is not None:
        lines.append(f"- LoRA rank: {rank}" + (f", alpha: {alpha}" if alpha is not None else ""))
    if epochs is not None:
        lines.append(f"- Epochs: {epochs}")
    if teachers:
        lines.append(f"- Teachers: {', '.join(teachers)}")
    sources = training.get("sources")
    if sources:
        lines.append("")
        lines.append("| Source | English | Turkish | German |")
        lines.append("|---|---|---|---|")
        for src in sources:
            name = src.get("source", "-")
            en = src.get("en", "-")
            tr = src.get("tr", "-")
            de = src.get("de", "-")
            en_s = f"{en:,}" if isinstance(en, int) else "-"
            tr_s = f"{tr:,}" if isinstance(tr, int) else "-"
            de_s = f"{de:,}" if isinstance(de, int) else "-"
            lines.append(f"| {name} | {en_s} | {tr_s} | {de_s} |")
    return "\n".join(lines) + "\n\n</details>\n"


def card(
    lang: str,
    ours: dict,
    base: dict,
    suites: list[str],
    gguf_report: dict | None = None,
    training: dict | None = None,
    probe_results: dict | None = None,
    probe_baseline: dict | None = None,
    notes: str = "",
    comparison: dict | None = None,
    media: bool = False,
    examples: str = "",
    examples_url: str = "",
) -> str:
    m = MODELS[lang]
    name = m["name"]
    ram = ""
    if gguf_report and "memory" in gguf_report:
        peak = gguf_report["memory"].get("Q4_K_M", {}).get("peak_rss_gb")
        size = gguf_report.get("sizes_gb", {}).get("Q4_K_M")
        if peak:
            ram = f"| Q4_K_M file / peak RAM | {size:.2f} GB / {peak:.2f} GB (measured, 4k context) |\n"
    front = {
        "license": "apache-2.0",
        "language": ["en", "tr", "de"] if lang == "en" else [lang, "en"],
        "base_model": "internlm/Intern-Decision-4B",
        "library_name": "transformers",
        "pipeline_tag": "text-classification",
        "datasets": ["mertkayacs/jevalt-data"],
        "tags": ["decision-model", "calibration", "conformal-prediction", "uncertainty", "reasoning", "routing", "triage",
                 "jev", "typesafe", "qwen3.5", m["language"].lower()],
    }
    head = "---\n" + "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in front.items()) + "\n---\n\n"

    probe_tbl = _probe_table(probe_results, probe_baseline, name, "Intern-Decision-4B")
    probe_sec = f"## Probes\n\nOn 100 typed-decisions items, same client for both models.\n\n{probe_tbl}\n\n" if probe_tbl else ""
    results = (f"## Results\n\nHeld-out suites, same prompts and client for both models, each at its fitted calibration.\n\n"
               f"{results_table(ours, base, suites, name, 'Intern-Decision-4B')}\n\n{notes}\n\n{probe_sec}")
    if comparison:  # charts carry the start checkpoint, Kev-4B, Laya and Jev 1.13; the tables stay in the result files
        results = charts_section(lang, comparison, notes)
    article = "An" if m["language"][0] in "AEIOU" else "A"
    training_sec = _training_section(training)

    try_it = (f"## Try it\n\nOpen the [Space]({SPACE}), pick an example and press Decide, or write your own situation, question and options. "
              f"These are the Space's examples in {m['language']} with {name}'s answers on 1 October 2026:\n\n{examples}\n\n"
              f"Every probability of every run, in all three languages: [space-examples.json]({examples_url}).\n\n") if examples else \
             f"## Try it\n\n[Hugging Face Space]({SPACE}), no install needed.\n\n"

    return head + f"""# {name}

{article} {m['language']} decision model with the Jev API. You send a state and typed questions (Choice, Score, Noul) and get a calibrated probability for every option. It can think before it answers, it can say "unknown", and the Q4_K_M build runs on your own machine in about 3 GB of RAM.

**[Try it](#try-it) · [Run it](#run-it) · [Results](#results) · [Use and limits](#use-and-limits) · [Code and links](#code-and-links)**

{media_block(lang) if media else ''}{try_it}| | |
|---|---|
| Start checkpoint | [internlm/Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) (Qwen3.5-4B) |
| Languages | {m['language']} first, the others still work |
| API | TypeSafe's `POST /v1/systemone`, request and response unchanged |
| Extras | `reasoning` off / on / auto, `abstain`, `coverage` (conformal sets) |
{ram}| License | Apache-2.0 |

{m['native'] or ''}
## Run it

```bash
pip install "jevalt[serve,gguf] @ git+https://github.com/mertkayacs/jevalt"
jevalt serve --model mertkayacs/{m["repo"]}-GGUF --file {m["repo"]}-Q4_K_M.gguf
```

Any TypeSafe client works against it:

```python
from typesafe_sdk import TypeSafeClient, Choice, Noul
client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8000")
```

{results}{training_sec}
## Use and limits

- Good for routing, tagging, triage and moderation at volume, and for automated decisions that need calibrated probabilities.
- Runs on-device or on-prem with the GGUF build, so the data stays with you.
- Knowledge is bounded by a 4B model, and the context is 8k tokens, so it is no tool for general questions or long summaries.
- Probabilities are calibrated on our held-out data. Refit with `jevoss calibrate` on yours before you set thresholds.
- Reasoning traces add little on our test rows (see the results), and there is no image input.

## Citation

<details>
<summary>BibTeX</summary>

{BIBTEX}

</details>

## Code and links

- Code, server and training: [{REPO}]({REPO})
- Playground, probes and recipes: [{PLAYGROUND}]({PLAYGROUND})
- Try it online: [Space]({SPACE})
- The village game: [Emberwick](https://emberwick.mertkayacs.com)
- Project site: [jevalt.mertkayacs.com](https://jevalt.mertkayacs.com)

If this is useful to you, a star on [GitHub]({REPO}) helps other people find it.
"""


def _gguf_row(quant, export_report, memory_report, repo_name):
    """Build one table row for a quant file."""
    sizes = export_report.get("sizes_gb", {}) if export_report else {}
    parity = export_report.get("parity", {}) if export_report else {}
    memory = export_report.get("memory", {}) if export_report else {}

    size = sizes.get(quant)
    p = parity.get(quant, {})
    agree = p.get("argmax_agreement")
    gap = p.get("max_prob_gap")
    sec = p.get("sec_per_request")
    peak = memory.get(quant, {}).get("peak_rss_gb")
    load_setting = "no mmap, repack on (jevalt default)"

    # Memory job has more detail
    if memory_report:
        fname = f"{repo_name}-{quant}.gguf"
        runs = memory_report.get("runs", {}).get(fname, {})
        for cfg in ["mmap+repack", "nommap+repack", "mmap+norepack", "nommap+norepack"]:
            r = runs.get(cfg, {})
            if r:
                peak = r.get("peak_rss_gb", peak)
                sec = r.get("sec_per_request", sec)
                load_setting = cfg
                break

    size_s = f"{size:.2f} GB" if size is not None else "-"
    agree_s = f"{agree:.1%}" if agree is not None else "-"
    gap_s = f"{gap:.4f}" if gap is not None else "-"
    sec_s = f"{sec:.2f} s" if sec is not None else "-"
    peak_s = f"{peak:.2f} GB" if peak is not None else "-"

    return f"| {quant} | {size_s} | {agree_s} | {gap_s} | {sec_s} | {peak_s} | {load_setting} |"


def gguf_card(lang: str, export_report: dict | None, memory_report: dict | None) -> str:
    m = MODELS[lang]
    name = m["name"]
    repo = m["repo"]

    front = {
        "license": "apache-2.0",
        "language": ["en", "tr", "de"] if lang == "en" else [lang, "en"],
        "base_model": f"mertkayacs/{repo}",
        "base_model_relation": "quantized",
        "quantized_by": "mertkayacs",
        "pipeline_tag": "text-classification",
        "tags": ["gguf", "llama.cpp", "decision-model", "calibration", "local-ai", m["language"].lower()],
    }
    head = "---\n" + "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in front.items()) + "\n---\n\n"

    quants = ["Q4_K_M", "Q5_K_M", "Q8_0"]
    rows = [
        "| File | Size | Argmax agreement | Max prob gap | Sec/request | Peak RAM | Load setting |",
        "|---|---|---|---|---|---|---|",
    ]
    for q in quants:
        rows.append(_gguf_row(q, export_report, memory_report, repo))

    table = "\n".join(rows)

    return head + f"""# {name} GGUF

Quantized GGUF files for {name}, the {m['language']} decision model, for CPUs and small machines. The Q4_K_M file is the default; Q5_K_M and Q8_0 are higher fidelity at the cost of speed and memory.

**[Files](#files) · [Use with jevalt](#use-with-jevalt) · [Code and links](#code-and-links)** · Examples and results: [{name} card](https://huggingface.co/mertkayacs/{repo}#try-it) · Try it: [Space]({SPACE})

![{GIF_CAPTION}]({VIDEOS}/resolve/main/gifs/emberwick-en.gif)

{_video('en')}

*{FILM_CAPTION}*

{native_block(lang)}## Files

{table}

Measured by the export job on an HF cpu-upgrade machine (8 vCPU): argmax agreement and probability gap against the full-precision model on held-out validation requests, seconds per request with 8 threads, peak RAM of a fresh process serving the file at 4k context with 4 threads. The load setting column shows the llama.cpp configuration behind the RAM figure.

## Use with jevalt

```bash
pip install "jevalt[serve,gguf] @ git+https://github.com/mertkayacs/jevalt"
jevalt serve --model mertkayacs/{repo}-GGUF --file {repo}-Q4_K_M.gguf
```

Then send the Jev request body to `http://127.0.0.1:8000/v1/systemone`. The answers come from the model's next-token probabilities at each decision marker, which the jevalt server reads through llama-cpp-python. llama.cpp's own `llama-server` and LM Studio can load the file, but their chat endpoints return generated text, so they give you neither the Jev API nor calibrated probabilities.

## Code and links

- Full-precision weights: [mertkayacs/{repo}](https://huggingface.co/mertkayacs/{repo})
- How it compares with Jev 1.13, Kev-4B and Laya: [charts on the model card](https://huggingface.co/mertkayacs/{repo}#results)
- Try it online: [Space]({SPACE})
- Code and training: [{REPO}]({REPO})

If this is useful to you, a star on [GitHub]({REPO}) helps other people find it.
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("model")
    m.add_argument("--lang", required=True)
    m.add_argument("--ours", required=True)
    m.add_argument("--base", required=True)
    m.add_argument("--gguf-report", default="")
    m.add_argument("--suites", nargs="+", required=True)
    m.add_argument("--training-summary", default="")
    m.add_argument("--probe-results", default="")
    m.add_argument("--probe-baseline", default="")
    m.add_argument("--out", required=True)

    g = sub.add_parser("gguf")
    g.add_argument("--lang", required=True)
    g.add_argument("--export-report", required=True)
    g.add_argument("--memory-report", default="")
    g.add_argument("--out", required=True)

    a = p.parse_args()
    if a.cmd == "model":
        gguf = json.load(open(a.gguf_report)) if a.gguf_report else None
        training = json.load(open(a.training_summary)) if a.training_summary else None
        probes = json.load(open(a.probe_results)) if a.probe_results else None
        baseline = json.load(open(a.probe_baseline)) if a.probe_baseline else None
        open(a.out, "w").write(card(a.lang, json.load(open(a.ours)), json.load(open(a.base)), a.suites, gguf, training, probes, baseline))
    elif a.cmd == "gguf":
        exp = json.load(open(a.export_report)) if a.export_report else None
        mem = json.load(open(a.memory_report)) if a.memory_report else None
        open(a.out, "w").write(gguf_card(a.lang, exp, mem))
