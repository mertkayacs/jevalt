# JevAlt

JevAlt is a family of open decision models that speak TypeSafe's Jev API. Give it a state and typed questions; it returns a calibrated probability for every option. It runs on your own machine in 4 GB of RAM (measured peak 3.0 GB for the Q4_K_M builds at 4k context).

| Model | Language | Hugging Face |
|---|---|---|
| Deem-4B | English | [mertkayacs/Deem-4B](https://huggingface.co/mertkayacs/Deem-4B) |
| Karar-4B | Turkish | [mertkayacs/Karar-4B](https://huggingface.co/mertkayacs/Karar-4B) |
| Wähler-4B | German | [mertkayacs/Wahler-4B](https://huggingface.co/mertkayacs/Wahler-4B) |

Each model has a GGUF build next to it (`-GGUF`). The `jevalt` server runs it on a CPU and answers the Jev API; it reads the model's next-token probabilities at each decision, which plain chat servers do not expose.

## Three question types

- **Choice** picks one option from a set you define and returns a probability for each.
- **Score** rates the state on an ordered rubric and returns the expected level plus the distribution.
- **Noul** answers a yes/no question with the probability of yes.

Your code acts on the numbers. A 0.97 can go straight through; a 0.55 can go to a person.

## What JevAlt adds

![What Jev 1.13 lacks and JevAlt has: thinking when unsure, an unknown answer, coverage sets, native Turkish and German, open weights, repeatable answers](assets/charts/jev.png)

![Hidden instructions, option order, long policies and negated questions: JevAlt against Intern-Decision-4B, Kev-4B and Laya](assets/charts/fixes.png)

Turn the extras on per request with `reasoning: "auto"`, `abstain: true` and `coverage: 0.9`. The Jev 1.13 rows come from [TypeSafe's notes](https://docs.typesafe.ai/model-jaggedness/jev-1.13) and an [independent audit](https://github.com/jujumilk3/jev-calibration-audit/blob/main/FINDINGS.md). The measured numbers behind each row are on the model cards and in the [JevOss](https://github.com/mertkayacs/jevoss) reports.

## Start

```bash
pip install "jevalt[serve,gguf] @ git+https://github.com/mertkayacs/jevalt"
jevalt serve
```

Then send any Jev request to `http://127.0.0.1:8000/v1/systemone`. The [API page](api.md) has the full contract, [Reasoning](reasoning.md) explains the modes, and [Run locally](local.md) covers memory and speed.

If JevAlt saves you time, a star on [GitHub](https://github.com/mertkayacs/jevalt) helps others find it.
