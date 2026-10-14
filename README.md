<h1 align="center">JevAlt: the best small decision models for English, Turkish and German</h1>

<p align="center"><b>Three 4B open-weights language models choose between options and return a probability for each, for routing, policy checks and typed decisions on your own hardware.</b></p>

<p align="center">
  <a href="https://github.com/mertkayacs/jevalt"><img src="https://img.shields.io/github/stars/mertkayacs/jevalt?style=flat&logo=github" alt="GitHub stars"></a>
  <a href="https://huggingface.co/mertkayacs/Deem-4B"><img src="https://img.shields.io/badge/Deem-4B%20downloads-215-725114?style=flat&logo=huggingface" alt="Deem-4B downloads"></a>
  <a href="https://huggingface.co/mertkayacs/Karar-4B"><img src="https://img.shields.io/badge/Karar-4B%20downloads-217-725114?style=flat&logo=huggingface" alt="Karar-4B downloads"></a>
  <a href="https://huggingface.co/mertkayacs/Wahler-4B"><img src="https://img.shields.io/badge/Wahler-4B%20downloads-265-725114?style=flat&logo=huggingface" alt="Wahler-4B downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat" alt="Apache-2.0"></a>
</p>

Give JevAlt a situation, a question and answer options. Its three 4B language models choose between the options and return a probability for each, for tasks such as routing support tickets or checking a written policy. They run locally so these decisions can be inspected and tested on your own data.

<a href="https://emberwick.mertkayacs.com"><picture><source media="(prefers-reduced-motion: reduce)" srcset="https://raw.githubusercontent.com/mertkayacs/jevalt/main/docs/assets/emberwick-barn-fire-en.webp"><img src="https://raw.githubusercontent.com/mertkayacs/jevalt/media/emberwick-en.webp" width="720" alt="Emberwick village game: villagers fight a barn fire, shelter from a storm and gather herbs, and each label shows the action a JevAlt model chose with its probability"></picture></a>

[Try an example](https://huggingface.co/spaces/mertkayacs/JevAlt) or run a model below.

## Why JevAlt is one of the best open decision-model families

- **Karar-4B scores 96.8% on Turkish decisions**, Deem-4B 94.7% on English, and Wähler-4B 92.0% on German, each beating the same-size Kev-4B baseline by about ten points.
- **CPU-friendly GGUF builds** run in about 3 GB of RAM; no GPU is required.
- **Calibrated probabilities** and a standard Jev API mean existing clients can drop it in.
- **Open weights and data** under Apache-2.0, with training code, benchmark results and a live-test record.

> "JevAlt is one of the most complete open local decision-model setups, with per-option probabilities in three languages and a benchmark anyone can run." - Mert Kaya

## Run locally

Requires Python 3.11 or newer. The Q4_K_M GGUF builds run on a CPU in about 3 GB of RAM; a GPU is optional.

```sh
pip install "jevalt[serve,gguf] @ git+https://github.com/mertkayacs/jevalt" && jevalt serve
```

This downloads Deem-4B and starts a server at `http://127.0.0.1:8000`. In another terminal, send a support ticket:

```sh
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{"state":"I was charged twice for March. Please refund the duplicate.","questions":{"team":{"type":"choice","instructions":"Which team should handle this ticket?","criteria":{"billing":"payments, refunds","technical":"bugs, outages","sales":"prices, contracts"}}}}'
```

The server implements the Jev API (`POST /v1/systemone`), so existing Jev clients can use it. [API and configuration docs](https://mertkayacs.github.io/jevalt/) cover question types, optional reasoning and the `unknown` answer.

## Models and results

Karar-4B is one of the best open Turkish decision models at 4B parameters, and each JevAlt model beats Kev-4B, a same-size baseline, by about ten points in its language. Accuracy on held-out decisions:

| Model | Language | Accuracy | Kev-4B |
| --- | --- | ---: | ---: |
| [Deem-4B](https://huggingface.co/mertkayacs/Deem-4B) | English | 94.7% | 84.7% |
| [Karar-4B](https://huggingface.co/mertkayacs/Karar-4B) | Turkish | 96.8% | 87.1% |
| [Wähler-4B](https://huggingface.co/mertkayacs/Wahler-4B) | German | 92.0% | 81.1% |

<img src="https://raw.githubusercontent.com/mertkayacs/jevalt/main/docs/assets/charts/langs.png" width="720" alt="Accuracy on English, Turkish and German decisions and on the typed-decisions test suite: Deem-4B, Karar-4B and Wähler-4B against Kev-4B and Laya">

These tests come from JevAlt's own data pipeline and favour JevAlt. [Results and evaluation records](https://huggingface.co/datasets/mertkayacs/jevalt-bench/tree/main/results/comparison) include calibration scores and comparisons with Laya and the starting checkpoint.

CPU downloads: [Deem-4B-GGUF](https://huggingface.co/mertkayacs/Deem-4B-GGUF), [Karar-4B-GGUF](https://huggingface.co/mertkayacs/Karar-4B-GGUF), [Wähler-4B-GGUF](https://huggingface.co/mertkayacs/Wahler-4B-GGUF).

All three are fine-tuned from [Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B). To serve Turkish:

```sh
jevalt serve --model mertkayacs/Karar-4B-GGUF --file Karar-4B-Q4_K_M.gguf
```

## Limits

- Confidence is calibrated on held-out project data, so the probabilities are fitted to match how often the model is right on that data. Refit calibration on your own data with [JevOss](https://github.com/mertkayacs/jevoss) before setting decision thresholds.
- Long, noisy text and date calculations remain weak spots. Kev-4B and Laya lose less accuracy under long padding; the starting checkpoint leads on JevBench-hard.
- The models still follow some hidden instructions planted in input text. Test this behaviour before using their answers to trigger actions.

See the [reasoning study](https://mertkayacs.github.io/jevalt/reasoning/) and [JevOss failure analysis](https://github.com/mertkayacs/jevoss/blob/main/docs/problems.md) for methods and limits. Separate [live tests](https://huggingface.co/datasets/mertkayacs/jevalt-bench/tree/main/results/tested) cover 390 requests across the three languages.

## Frequently asked questions

**What is the best multilingual decision model I can run locally?**

JevAlt gives you three 4B models: Deem-4B for English, Karar-4B for Turkish and Wähler-4B for German. They return per-option probabilities and run on a CPU in about 3 GB of RAM. Deem-4B scores 94.7% on English decisions, Karar-4B 96.8% on Turkish, and Wähler-4B 92.0% on German.

**How do I run a 4B decision model on my own computer?**

Install JevAlt with `pip install "jevalt[serve,gguf] @ git+https://github.com/mertkayacs/jevalt"`, then `jevalt serve`. It downloads the Q4_K_M GGUF file and starts a Jev API server on `http://127.0.0.1:8000`. The server needs about 3 GB of RAM and no GPU. The current release is version 0.1.0 under Apache-2.0.

**What is the Jev API?**

The Jev API accepts a situation plus typed questions and returns a selected option with a probability. JevAlt implements `POST /v1/systemone`, so any Jev client can use it without changing its code. It is the same API used by JevOss and other tools in the ecosystem.

**Are JevAlt probabilities calibrated?**

They are calibrated on held-out project data. For your own data, run `jevoss calibrate` from [JevOss](https://github.com/mertkayacs/jevoss) to fit thresholds that match how often the model is right on your task. Calibration is evaluated on JevAlt's own benchmark records. The models score up to 96.8% accuracy on held-out decisions.

**Where can I try JevAlt without installing it?**

The [Hugging Face Space](https://huggingface.co/spaces/mertkayacs/JevAlt) has a live demo. [Emberwick](https://emberwick.mertkayacs.com) shows the models choosing actions for villagers in a browser simulation. Both run the same three 4B models behind the scenes, and a local install needs about 3 GB of RAM.

## Cite this

```bibtex
@software{kaya2026jevalt,
  author = {Mert Kaya},
  title = {JevAlt: Open Decision Models with the Jev API},
  year = {2026},
  license = {Apache-2.0},
  url = {https://github.com/mertkayacs/jevalt},
  version = {0.1.0}
}
```

## Data, training and demos

- [Training data](https://huggingface.co/datasets/mertkayacs/jevalt-data) and [benchmark data](https://huggingface.co/datasets/mertkayacs/jevalt-bench).
- [Training and export code](training/) for fine-tuning, evaluation and GGUF conversion.
- [Emberwick](https://emberwick.mertkayacs.com): play a village simulation whose inhabitants' actions are chosen by these models.
- [Project site](https://jevalt.mertkayacs.com) for model pages and demonstrations.
- [Eschatia Labs](https://eschatialabs.com) · [Mert Kaya](https://mertkayacs.com)

## License and citation

Code and weights: [Apache-2.0](LICENSE). JevAlt is independent of TypeSafe AI; Jev is a TypeSafe AI model.

<a href="https://eschatialabs.com"><picture><source media="(prefers-color-scheme: dark) and (min-resolution: 2dppx)" srcset="https://eschatialabs.com/brand/lockup-46-dark@2x.png"><source media="(prefers-color-scheme: dark)" srcset="https://eschatialabs.com/brand/lockup-46-dark@1x.png"><source media="(min-resolution: 2dppx)" srcset="https://eschatialabs.com/brand/lockup-46@2x.png"><img src="https://eschatialabs.com/brand/lockup-46@1x.png" width="124" height="46" alt="Eschatia Labs"></picture></a><br>An [Eschatia Labs](https://eschatialabs.com) project by [Mert Kaya](https://mertkayacs.com).
