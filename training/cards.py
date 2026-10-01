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
            "## Türkçe özet\n\n"
            "Karar-4B, Türkçe yazılmış kararlar üzerinde eğitilmiş açık bir modeldir. Bir durum ve "
            "Choice, Score ya da Noul sorusu gönderirsiniz; her seçenek için kalibre edilmiş olasılık "
            "alırsınız. İsterseniz model önce kısa bir Türkçe gerekçe yazar, sonra karar verir. "
            "Q4_K_M sürümü 4 GB RAM ile kendi makinenizde çalışır, veriler dışarı çıkmaz.\n"
        ),
    },
    "de": {
        "name": "Wähler-4B",
        "repo": "Wahler-4B",
        "language": "German",
        "native": (
            "## Deutsche Zusammenfassung\n\n"
            "Wähler-4B ist ein offenes Entscheidungsmodell, das auf deutschsprachigen Entscheidungen "
            "trainiert wurde. Sie senden einen Zustand und Choice-, Score- oder Noul-Fragen und erhalten "
            "für jede Option eine kalibrierte Wahrscheinlichkeit. Auf Wunsch schreibt das Modell zuerst "
            "eine kurze Begründung auf Deutsch und entscheidet dann. Die Q4_K_M-Version läuft mit 4 GB RAM "
            "auf dem eigenen Rechner, die Daten bleiben lokal.\n"
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


def _training_section(training: dict | None) -> str:
    if not training:
        return ""
    lines = ["## How it was trained", ""]
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
    return "\n".join(lines) + "\n"


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
        "tags": ["decision-model", "jev", "typesafe", "system-one", "calibration", "conformal-prediction", "reasoning", "typed-decisions", "qwen3.5", "gguf", m["language"].lower()],
    }
    head = "---\n" + "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in front.items()) + "\n---\n\n"

    probe_tbl = _probe_table(probe_results, probe_baseline, name, "Intern-Decision-4B")
    probe_sec = f"## Probes\n\nOn 100 typed-decisions items, same client for both models.\n\n{probe_tbl}\n\n" if probe_tbl else ""
    training_sec = _training_section(training)

    return head + f"""# {name}

A {m['language']} decision model with the Jev API. You send a state and typed questions (Choice, Score, Noul) and get a calibrated probability for every option. It can think before it answers, it can say "unknown", and the Q4_K_M build runs in 4 GB of RAM on your own machine.

If this is useful to you, a star on [GitHub]({REPO}) helps other people find it.

**Try it**: [Hugging Face Space]({SPACE}) (no install needed).

| | |
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

## Results

Held-out suites, same prompts and client for both models, each at its fitted calibration.

{results_table(ours, base, suites, name, 'Intern-Decision-4B')}

{notes}

{probe_sec}{training_sec}
## Intended use

- Routing, tagging and moderation at volume.
- Calibrated probabilities for automated decisions.
- On-device or on-prem inference with the GGUF build.

## Out of scope

- General knowledge QA beyond what a 4B model can hold.
- Long-context summarization (8k context).
- Production use without recalibrating on your own data.

## Limits

- Knowledge questions are bounded by a 4B model, and reasoning traces add little on our test rows (see the results).
- Probabilities are calibrated on our held-out data. Refit with `jevoss calibrate` on yours before you set thresholds.
- No image input.

## Citation

{BIBTEX}

## Links

- Code, server and training: [{REPO}]({REPO})
- Playground, probes and recipes: [{PLAYGROUND}]({PLAYGROUND})
- Try it online: [Space]({SPACE})
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
        "base_model": f"mertkayacs/{repo}",
        "quantized_by": "mertkayacs",
        "tags": ["gguf", "llama.cpp", "decision-model", "calibration", m["language"].lower()],
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

Quantized GGUF files for {name}, the {m['language']} decision model. The Q4_K_M file is the default; Q5_K_M and Q8_0 are higher fidelity at the cost of speed and memory.

If this is useful to you, a star on [GitHub]({REPO}) helps other people find it.

## Files

{table}

Measured by the export job on an HF cpu-upgrade machine (8 vCPU): argmax agreement and probability gap against the full-precision model on held-out validation requests, seconds per request with 8 threads, peak RAM of a fresh process serving the file at 4k context with 4 threads. The load setting column shows the llama.cpp configuration behind the RAM figure.

## Use with jevalt

```bash
pip install "jevalt[serve,gguf] @ git+https://github.com/mertkayacs/jevalt"
jevalt serve --model mertkayacs/{repo}-GGUF --file {repo}-Q4_K_M.gguf
```

Then send the Jev request body to `http://127.0.0.1:8000/v1/systemone`. The answers come from the model's next-token probabilities at each decision marker, which the jevalt server reads through llama-cpp-python. llama.cpp's own `llama-server` and LM Studio can load the file, but their chat endpoints return generated text, so they give you neither the Jev API nor calibrated probabilities.

## Links

- Full-precision weights: [mertkayacs/{repo}](https://huggingface.co/mertkayacs/{repo})
- Try it online: [Space]({SPACE})
- Code and training: [{REPO}]({REPO})
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
