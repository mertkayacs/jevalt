# JevAlt: small decision models for English, Turkish and German

Give JevAlt a situation, a question and answer options. Its three 4B language models choose between the options and return a probability for each, for tasks such as routing support tickets or checking a written policy. They run locally so these decisions can be inspected and tested on your own data.

<a href="https://emberwick.mertkayacs.com"><picture><source media="(prefers-reduced-motion: reduce)" srcset="https://raw.githubusercontent.com/mertkayacs/jevalt/main/docs/assets/emberwick-barn-fire-en.webp"><img src="https://raw.githubusercontent.com/mertkayacs/jevalt/media/emberwick-en.webp" width="720" alt="Emberwick village game: villagers fight a barn fire, shelter from a storm and gather herbs, and each label shows the action a JevAlt model chose with its probability"></picture></a>

[Try an example](https://huggingface.co/spaces/mertkayacs/JevAlt) or run a model below.

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

## Data, training and demos

- [Training data](https://huggingface.co/datasets/mertkayacs/jevalt-data) and [benchmark data](https://huggingface.co/datasets/mertkayacs/jevalt-bench).
- [Training and export code](training/) for fine-tuning, evaluation and GGUF conversion.
- [Emberwick](https://emberwick.mertkayacs.com): play a village simulation whose inhabitants' actions are chosen by these models.
- [Project site](https://jevalt.mertkayacs.com) for model pages and demonstrations.

## License and citation

Code and weights: [Apache-2.0](LICENSE). JevAlt is independent of TypeSafe AI; Jev is a TypeSafe AI model.

<details>
<summary>BibTeX</summary>

```bibtex
@software{kaya2026jevalt,
  author = {Mert Kaya},
  title = {JevAlt: Open Decision Models with the Jev API},
  year = {2026},
  license = {Apache-2.0},
  url = {https://github.com/mertkayacs/jevalt}
}
```

</details>

<a href="https://eschatialabs.com"><picture><source media="(prefers-color-scheme: dark) and (min-resolution: 2dppx)" srcset="https://eschatialabs.com/brand/lockup-46-dark@2x.png"><source media="(prefers-color-scheme: dark)" srcset="https://eschatialabs.com/brand/lockup-46-dark@1x.png"><source media="(min-resolution: 2dppx)" srcset="https://eschatialabs.com/brand/lockup-46@2x.png"><img src="https://eschatialabs.com/brand/lockup-46@1x.png" width="124" height="46" alt="Eschatia Labs"></picture></a><br>An [Eschatia Labs](https://eschatialabs.com) project by [Mert Kaya](https://mertkayacs.com).
