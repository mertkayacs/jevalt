# JevAlt

![Emberwick: every villager asks Deem-4B what to do next](https://raw.githubusercontent.com/mertkayacs/jevalt/media/emberwick-en.gif)

*Emberwick, a village game where every villager asks Deem-4B what to do next. Nothing is scripted; the card at the bottom shows the choice and how sure the model was.* Also in [Türkçe](https://huggingface.co/datasets/mertkayacs/emberwick-videos/resolve/main/gifs/emberwick-tr.gif) and [Deutsch](https://huggingface.co/datasets/mertkayacs/emberwick-videos/resolve/main/gifs/emberwick-de.gif).

Watch the one-minute film with sound: [English](https://huggingface.co/datasets/mertkayacs/emberwick-videos/resolve/main/film/jevalt-film-en-1080p.mp4), [Türkçe](https://huggingface.co/datasets/mertkayacs/emberwick-videos/resolve/main/film/jevalt-film-tr-1080p.mp4), [Deutsch](https://huggingface.co/datasets/mertkayacs/emberwick-videos/resolve/main/film/jevalt-film-de-1080p.mp4).

![JevAlt: open decision models with honest odds, in English, Turkish and German](https://raw.githubusercontent.com/mertkayacs/jevalt/media/jevalt-card.png)

Open decision models with the Jev API. Calibrated Choice, Score and Noul answers, reasoning when unsure, native Turkish and German, runs offline in about 3 GB of RAM.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-mertkayacs-yellow)](https://huggingface.co/mertkayacs)
[![Docs](https://img.shields.io/badge/docs-mertkayacs.github.io/jevalt-green)](https://mertkayacs.github.io/jevalt/)

If this is useful to you, a star on GitHub helps other people find it.

## What it is

JevAlt is a family of open 4B decision models that speak TypeSafe's `POST /v1/systemone` API. You send a state and typed questions (Choice, Score, Noul); it returns a calibrated probability for every option. It can reason before it answers, it can say "unknown", and the Q4_K_M build runs on your own machine in 4 GB of RAM (measured peak: 3.0 GB at 4k context). A request that works with Jev works here without changes.

## Models

| Model | Language | Weights | GGUF |
|---|---|---|---|
| Deem-4B | English | [mertkayacs/Deem-4B](https://huggingface.co/mertkayacs/Deem-4B) | [mertkayacs/Deem-4B-GGUF](https://huggingface.co/mertkayacs/Deem-4B-GGUF) |
| Karar-4B | Turkish | [mertkayacs/Karar-4B](https://huggingface.co/mertkayacs/Karar-4B) | [mertkayacs/Karar-4B-GGUF](https://huggingface.co/mertkayacs/Karar-4B-GGUF) |
| Wähler-4B | German | [mertkayacs/Wahler-4B](https://huggingface.co/mertkayacs/Wahler-4B) | [mertkayacs/Wahler-4B-GGUF](https://huggingface.co/mertkayacs/Wahler-4B-GGUF) |

The same Q4_K_M files are on [Kaggle](https://www.kaggle.com/models/mertilovski/jevalt) with a [CPU quickstart notebook](https://www.kaggle.com/code/mertilovski/jevalt-quickstart-calibrated-decisions-on-cpu), and all three models answer in the [Space](https://huggingface.co/spaces/mertkayacs/JevAlt). Each model has a Q4_K_M GGUF file (for example `Deem-4B-Q4_K_M.gguf`) that the `jevalt` server runs on a CPU. The server reads the model's next-token probabilities at each decision, so it is the piece that speaks the Jev API; a plain chat server can load the file but returns text.

## Quickstart

Python 3.11 or newer. A GPU is optional.

```bash
pip install "jevalt[serve,gguf] @ git+https://github.com/mertkayacs/jevalt"
jevalt serve
```

`jevalt serve` downloads Deem-4B on the first run and listens on `http://127.0.0.1:8000`.

Ask a question from a JSON file:

```bash
jevalt ask request.json
```

Or call the server with curl:

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Hi, my payouts have been failing for 3 days.",
    "questions": {
      "is_urgent": {
        "type": "noul",
        "instructions": "Does this convey urgency?"
      }
    }
  }'
```

To serve a specific model:

```bash
jevalt serve --model mertkayacs/Karar-4B-GGUF --file Karar-4B-Q4_K_M.gguf
```

Request extensions (`reasoning`, `abstain`, `coverage`) and the full API are in the [docs](https://mertkayacs.github.io/jevalt/).

## Results

![Accuracy on English, Turkish and German decisions and on typed-decisions: JevAlt, Intern-Decision-4B, Kev-4B and Laya](docs/assets/charts/langs.png)

![Hidden instructions, option order, long policies and negated questions: JevAlt against Intern-Decision-4B, Kev-4B and Laya](docs/assets/charts/fixes.png)

![What Jev 1.13 lacks and JevAlt has: thinking when unsure, an unknown answer, coverage sets, native Turkish and German, open weights, repeatable answers](docs/assets/charts/jev.png)

Same items and client for every model, each as shipped: [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) r10 and [Laya](https://huggingface.co/convaiinnovations/laya) 0.3.22 ran on their own servers with their own calibration. Jev 1.13 rows come from [TypeSafe's notes](https://docs.typesafe.ai/model-jaggedness/jev-1.13) and an [independent audit](https://github.com/jujumilk3/jev-calibration-audit/blob/main/FINDINGS.md). The held-out tests come from JevAlt's own data pipeline, so they favour JevAlt.

Where the others lead: Kev-4B on 10kGNAD, and on GermEval 2017 against Wähler-4B; Kev-4B and Laya lose less accuracy under 600 words of padding; the start checkpoint stays ahead on JevBench-hard; Laya is far smaller and faster. Every number with Brier and ECE, and every decision: [results](https://huggingface.co/datasets/mertkayacs/jevalt-bench/tree/main/results), [comparison](https://huggingface.co/datasets/mertkayacs/jevalt-bench/tree/main/results/comparison).

<details>
<summary>Significance and caveats</summary>

A paired bootstrap (2,000 resamples) puts every held-out gain well above zero: +4.4, +5.1 and +11.5 accuracy points in each model's own language. Off the training distribution the picture is flatter. Wähler-4B gains 4.75 points on 10kGNAD and Karar-4B 4.0 on GermEval, both significant; TurkishMMLU, typed-decisions and JevBench-hard show no significant accuracy change, and Brier gets slightly worse on typed-decisions and JevBench-hard. JevAlt trained on the typed-decisions train split; its scores use the test split.

Kev-4B and Laya received every row in the shapes the TypeSafe docs use (Noul criteria keyed `true`/`false`, Score levels as a list). Rows that need the `unknown` option are left out of every model's score, because Kev-4B and Laya do not offer it. Probes use 100 typed-decisions items.

Long noisy states are still a weak spot. Reasoning helps less than we hoped: with the fitted thresholds, `reasoning: "auto"` moved Deem-4B from 0.761 to 0.769 on the English date, number and policy test rows, left Karar-4B unchanged and made Wähler-4B's Brier worse ([details](https://mertkayacs.github.io/jevalt/reasoning/)).

</details>

## Reproduce

Training and evaluation code is in `training/`:

- `training/mix.py` composes a training mix from canonical JSONL files.
- `training/train.py` runs LoRA fine-tuning of Intern-Decision-4B on one A100 80 GB (HF Jobs flavor a100-large) with gradient checkpointing.
- `training/cards.py` builds Hugging Face model cards from measured results.
- `training/jobs/evaluate.py` loads a checkpoint (optionally with a LoRA adapter) in-process with transformers and scores suites and probes. Does not use a server.
- `training/jobs/eval_http.py` evaluates any running `/v1/systemone` server (suites and probes).
- `training/jobs/export.py` merges the adapter, converts to GGUF, quantizes (Q4_K_M, Q5_K_M, Q8_0), checks parity against full precision and measures peak RAM.
- `training/jobs/bakeoff.py` evaluates candidate start checkpoints on the same suites.

See the [docs](https://mertkayacs.github.io/jevalt/) for the API, reasoning modes, and local setup.

## License

Apache-2.0, for the code and the weights.

## Citation

```bibtex
@software{kaya2026jevalt,
  author = {Mert Kaya},
  title = {JevAlt: Open Decision Models with the Jev API},
  year = {2026},
  license = {Apache-2.0},
  url = {https://github.com/mertkayacs/jevalt}
}
```

## Acknowledgements

JevAlt starts from [internlm/Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) (Qwen3.5-4B), which is Apache-2.0. The API follows TypeSafe's public `/v1/systemone` specification, so existing clients keep working. JevAlt is an independent project with no affiliation to TypeSafe AI. Jev is a TypeSafe AI model.
