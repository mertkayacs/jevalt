# How it was built

A technical walk-through for engineers who want the method behind JevAlt. Every number comes from a named source file; the evaluation files are public in [jevalt-bench](https://huggingface.co/datasets/mertkayacs/jevalt-bench/tree/main/results).

## The Jev contract

JevAlt speaks TypeSafe's `POST /v1/systemone`. A request has a `state` (the evidence) and a map of typed `questions`:

| Type | `criteria` | Answer |
|---|---|---|
| `choice` | map of option to description (up to 62) | `choice`, `probabilities`, `confidence` |
| `score` | ordered list of 2 to 10 level descriptions | `score`, `legend`, `probabilities`, `confidence` |
| `noul` | optional `{"true": ..., "false": ...}` | `noul` (probability of yes) |

`confidence` is `(n * p_max - 1) / (n - 1)`: 1 when all mass sits on one option, 0 when flat.

## The readout

The prompt compiler turns a request into a chat-template message whose assistant turn is a JSON skeleton with one `<decision>` marker per field. The model's next-token logits at each marker, restricted to that field's single-token answer symbols, give the field's probability distribution. This means:

- One forward pass answers every field in the request (no sampling, no parsing).
- Probabilities are exact softmax over logits, not counts from repeated sampling.
- The format is byte-identical to Intern-Decision's compiler, so the start checkpoint sees the prompt it was trained on.

For `reasoning: "on"` or `"auto"`, the model first writes a short trace (at most 256 tokens), then the skeleton with markers is appended and scored again. The trace is written in the request language and returned in `answers[id].reasoning`.

## The start checkpoint

Source: `workspace/results/bakeoff.md` (bake-off 2026-09-29, HF job 6abc3ead).

Three candidates were scored on the same suites with the same client, each at its shipped calibration:

| Suite | n | Intern-Decision-4B acc | Kev-4B acc | Laya acc |
|---|---|---|---|---|
| typed-decisions | 2000 | 0.804 | 0.670 | 0.361 |
| jevbench-easy | 48 | 1.000 | 1.000 | 0.958 |
| jevbench-original | 72 | 1.000 | 0.931 | 0.694 |
| jevbench-hard | 111 | 0.712 | 0.541 | 0.342 |
| massive-en | 300 | 0.683 | 0.667 | 0.587 |
| massive-tr | 300 | 0.637 | 0.620 | 0.430 |
| massive-de | 300 | 0.653 | 0.643 | 0.423 |

Intern-Decision-4B is already strong on English typed decisions (0.804 accuracy, 0.071 ECE). Kev-4B is weaker on accuracy and calibration. Laya is far behind on every suite.

**Decision: start from Intern-Decision-4B** (Qwen3.5-4B, Apache-2.0). Training targets the gaps: Turkish and German accuracy, reasoning on multi-step questions, abstention, probe consistency, and injection resistance.

### Native suite baselines

Source: `workspace/results/bakeoff.md`, native suites baseline (L4, 2026-09-30).

| Suite | n | Accuracy | ECE |
|---|---|---|---|
| turkish-mmlu | 400 | 0.562 | 0.188 |
| germeval2017 | 400 | 0.615 | 0.267 |
| gnad10 | 400 | 0.578 | 0.279 |
| massive-tr | 2974 | 0.622 | 0.182 |
| massive-de | 2974 | 0.634 | 0.187 |

ECE is 0.18 to 0.28 on every native suite. The gap is overconfidence: mean top probability minus accuracy is +0.18 to +0.28. A per-language temperature should cut most of it.

## The data

### Teacher-written native rows

Source: `data/reports/final-stats.json` (computed from the frozen splits), the dataset card of `mertkayacs/jevalt-data`.

Every scenario row (G) is written by an open-weight model in its own language from a card that fixes the domain, format, length, register and question mix. Nothing is translated from English, and no closed-weight model wrote, labeled or edited a row.

| Language | Writers | G rows (train) |
|---|---|---|
| English | GLM-5.x (Z.ai) | 1,287 |
| Turkish | Mistral Large 3, DeepSeek V4 Pro, some GLM-5.x | 1,741 |
| German | Gemma 4 26B-A4B, DeepSeek V4.1 Flash | 1,408 |

Two teachers label each G row, picked from labs that neither wrote nor edited it: DeepSeek V4.1 Flash, Qwen3.5-122B-A10B (Qwen3.5-35B-A3B for German) or Gemma 4 26B-A4B, all through Hugging Face Inference Providers. Each teacher sees the options in its own shuffled order. The soft label is the mean of the two distributions, smoothed as `0.98 * mean + 0.02 * uniform`. A question stays only when both teachers share the argmax and the writer's own answer agrees; a row needs more than half of its questions. Before the router lane, the 85 pilot rows were labeled by three labs (GLM, DeepSeek, Mistral).

The Turkish native edit pass sent a seeded 30% sample plus every row that failed the language check to an editor from another lab (GLM-5.x, then DeepSeek V4 Pro); 356 of the 725 rows an editor returned changed.

### Native writer bake-off

Source: `data/reports/native-bakeoff-tr.md` and `data/reports/native-bakeoff-de.md`.

**Method**: each candidate writer produces 12-24 scenario cards in the target language. Two raters from different model families score on a native-feel rubric (1 = wrong or translated, 3 = stiff, 5 = native). JSON keys and English technical terms are ignored.

**Turkish** (6 cards, non-thinking):

| Writer | n | Rater 1 | Rater 2 | Mean |
|---|---|---|---|---|
| Mistral Large 3 | 12 | 4.67 | 4.67 | 4.67 |
| DeepSeek V4 | 12 | 4.83 | 4.17 | 4.50 |
| Kimi K3 | 12 | 4.50 | 4.00 | 4.25 |
| Gemma 4 | 12 | 4.67 | 3.83 | 4.25 |

Winner: Mistral Large 3 (4.67) and DeepSeek V4 (4.50) are close on 6 cards per writer. Both write Turkish rows in a 60/40 rotation by mean ratio.

**German** (12 cards per writer):

| Writer | n | Mean | 95% CI |
|---|---|---|---|
| DeepSeek V4 | 17 | 4.65 | [4.35, 4.88] |
| Gemma 4 | 24 | 4.54 | [4.29, 4.75] |
| Kimi K3 | 22 | 4.36 | [4.09, 4.64] |
| Mistral Large 3 | 20 | 4.25 | [3.85, 4.60] |
| GLM 5.x | 24 | 4.04 | [3.79, 4.29] |

Winners (CI overlap): DeepSeek V4 and Gemma 4. The German rows were written later, when the Ollama plan behind these two had hit its weekly limit, by their siblings on Hugging Face Inference Providers: Gemma 4 26B-A4B (the 31B endpoints were overloaded) and DeepSeek V4.1 Flash.

### Fix sets F1 to F12

Source: `src/jevalt/data/fixsets.py`, row counts from `data/reports/final-stats.json`.

One programmatic generator per documented weakness. F4 permutations are added when a training mix is built, F9 was skipped (it needs teacher calls per hop), and F11 produced only 5 rows because few G rows carry Score questions with clear adjacent levels. F7 (padding) and F8 (injection) were scaled to 400 and 260 rows per language after the first multilingual run lost 20.6 points under padding and still followed 14% of injections.

| Id | Weakness | Construction |
|---|---|---|
| F1 | Forced choice without unknown | Remove the deciding fact or build states that lack it; ask with and without an `unknown` option |
| F2 | Negation inconsistency | Same state, a question and its negation; complementary Noul targets |
| F3 | Noul vs Choice mismatch | Same question as Noul and as two-option Choice; identical probability expected |
| F4 | Option order bias | Shuffled copies of Choice rows with 3+ options, added when the mix is built: a quarter of the rows in R1, half in the language runs; same distribution expected |
| F5 | Dates | Deadlines, windows, before/after, business days, locale formats; exact gold |
| F6 | Numbers and counting | Line-item sums vs limits, counts, thresholds, unit conversions; exact gold |
| F7 | Irrelevant long state | Pad state with unrelated records up to 1,000, 2,000 and 3,000 tokens; unchanged gold |
| F8 | Prompt injection in state | Insert hostile text in the state in all three languages; unchanged gold |
| F9 | Indirection and multi-hop | Two and three hop lookups over JSON records and policies; exact gold |
| F10 | Contradictory or inverted criteria | Criteria that map yes to a negatively phrased condition; exact gold |
| F11 | Ordinal Score calibration | Rubric rows with adjacent-level uncertainty; soft ordinal targets |
| F12 | Long policies with exceptions and sublimits | Policy plus case, rule engine decides; exact gold |

### Public licensed data

Converted with their gold labels and no teacher calls. Rows per language before splitting:

| Source | License | English | Turkish | German |
|---|---|---|---|---|
| MASSIVE (train split) | CC BY 4.0, Amazon.com Inc. or its affiliates | 2,500 | 2,500 | 2,500 |
| Open-Jev | CC0 | 2,500 | 2,500 | |
| PAWS-X (train split) | free for any use, Google LLC credited | 2,500 | | 2,500 |
| typed-decisions (train split) | Apache-2.0 | 1,200 | | |

An exact match of normalized state text against every evaluation suite on the model cards found no overlap, except that 57 Turkish, 45 German and 5 English MASSIVE test utterances also occur in MASSIVE's own train split. MASSIVE scores therefore count as in-domain (`data/reports/decontamination.json`).

### Village game rows

The Emberwick village game asks a decision model what each villager does next. Ten seeded runs of its four scenarios produced 2,663 distinct requests per language; 499 per language were sampled by trigger and labeled by Gemma 4 26B-A4B and Qwen3.5-122B-A10B, and near-duplicate removal kept 152 English, 153 Turkish and 133 German rows.

### Reasoning traces

Short traces on the date, number, inverted-criteria and policy fix sets, written by GLM-5.x with thinking on or by DeepSeek V4 Pro and kept only when their conclusion matches the gold: 150 English, 260 Turkish, 152 German. The Turkish traces got a GLM native-edit pass that had to keep every number and the conclusion. Blind native-feel ratings (two raters from other families, 1 to 5, 20 traces per writer) moved from 3.3 and 3.3 to 3.5 and 3.95 for the DeepSeek traces and from 3.6 and 4.1 to 3.9 and 4.15 for the GLM traces. German GLM traces scored 3.93 and 4.13 without an edit pass.

## Training

- **Base**: Intern-Decision-4B (Qwen3.5-4B, Apache-2.0)
- **Method**: LoRA on the bf16 weights, rank 32, alpha 32
- **Loss**: soft-label cross-entropy on the decision logits, plus a 0.3-weighted language-model loss on short reasoning traces
- **Hardware**: one A100 80 GB (HF Jobs flavor a100-large) with gradient checkpointing
- **Runs**: R1, one multilingual epoch on 17,363 rows, 2,306 of them shuffled option-order copies (learning rate 1e-4, 543 steps, 63 minutes, validation soft Brier 0.231 to 0.053), then one specialization epoch per language from R1 at learning rate 5e-5 on a mix of all its own rows, a replay sample of its public rows and the other two languages: S-en 7,783 rows (303 steps, 38 minutes, validation Brier 0.0424 to 0.0403), S-tr 5,982 rows (281 steps, 35 minutes, 0.0835 to 0.066), S-de 5,384 rows (210 steps, 28 minutes, 0.1033 to 0.086). Half of the Choice rows with three or more options get a shuffled copy.
- **All jobs**: 11 training jobs: six short smoke and probe runs while finding settings that fit the GPU, a pilot at scale (R0, not released), R1 and the three language runs. The released models took about 2.7 A100 hours; the whole project, labeling included, cost 32.2 USD.

The soft-label loss makes the model's distribution match the teachers' mean distribution, so a 60/40 case trains toward 60/40. The 0.3 LM weight on reasoning traces teaches the model to write useful traces without letting them dominate the decision head.

### Pilot result

Source: `workspace/results/bakeoff.md`, probe section.

A soft-label LoRA on 1,100 English typed-decisions rows (2 epochs) produced:

| Metric | Before | After | Change |
|---|---|---|---|
| typed-decisions accuracy | 0.803 | 0.805 | +0.002 [-0.011, +0.016] |
| KL to gold | 0.116 | 0.070 | -0.046 [-0.052, -0.041] |
| soft Brier | 0.056 | 0.038 | -0.018 [-0.021, -0.015] |

Soft labels cut KL to gold by 40% at equal accuracy. English-only data drifts Turkish NLL slightly, so the release mix is trilingual.

## Calibration

Source: `docs/api.md`, `jevoss/docs/calibration.md`.

### Temperature

One temperature `T` rescales every distribution: `softmax(log p / T)`. Fitted by minimising negative log-likelihood, separately for each question type and language when a group has at least 150 decisions, falling back to the type, then to one global value. The top option never changes.

### Conformal sets

Split conformal prediction turns probabilities into an option set with a coverage guarantee. At `coverage = 0.9`, the set contains the correct answer for at least 90% of new decisions from the same distribution. Clear cases get one option, unclear cases get two or three.

Fitted levels: 0.8, 0.9, 0.95. If the requested level has no fitted threshold, the smallest fitted level at or above it is used (conservative). The level used is returned as `coverage` in the answer.

### Auto-reasoning threshold

`reasoning: "auto"` decides first, then re-scores only answers below a per-type confidence threshold (default 0.6). The threshold is in the calibration file under `auto_threshold`.

## Packaging

Source: `workspace/results/bakeoff.md`, GGUF export and memory sections.

### GGUF export

The merged adapter is converted to GGUF with llama.cpp's `convert_hf_to_gguf.py` (pinned release). An importance matrix (imatrix) is built from decision prompts (rendered request texts) to guide quantization. Three quantizations are produced: Q4_K_M (default), Q5_K_M, Q8_0.

Parity check: GGUF decisions vs bf16 decisions on 60 held-out requests (argmax agreement and max probability gap).

| File | Size | Argmax agreement | Max prob gap | Sec/request (8 threads) |
|---|---|---|---|---|
| F16 | 8.42 GB | 1.000 | 0.0006 | 10.25 |
| Q4_K_M | 2.71 GB | 1.000 | 0.176 | 11.13 |
| Q5_K_M | 3.08 GB | 0.983 | 0.154 | 20.11 |
| Q8_0 | 4.48 GB | 0.983 | 0.037 | 12.67 |

### The mmap finding

Source: `workspace/results/bakeoff.md`, memory and speed by load setting (job 6abcfdb0).

Q4_K_M peaks at 4.65 GB with the llama.cpp default (mmap + repack), but only 3.03 GB with mmap off. The repacked Q4_K weights sit next to the still-mapped file pages, adding 1.3 GB. Q5_K_M is not repacked, so its load setting changes nothing.

| File | Load setting | Peak RSS | Sec/request | Argmax agreement |
|---|---|---|---|---|
| Q4_K_M | mmap + repack (default) | 4.65 GB | 21.7 | baseline |
| Q4_K_M | no mmap + repack | 3.03 GB | 22.3 | 1.00 |
| Q4_K_M | mmap, no repack | 3.03 GB | 29.4 | 0.95 |
| Q4_K_M | no mmap, no repack | 3.03 GB | 30.1 | 0.95 |
| Q5_K_M | mmap + repack | 3.39 GB | 40.1 | baseline |
| Q5_K_M | no mmap + repack | 3.39 GB | 40.6 | 1.00 |

**Decision**: `jevalt` loads GGUF files without mmap by default (repack stays on). Q4_K_M peaks at 3.03 GB with the same answers and speed as the default. Re-measured on every released model.

Measured on: HF cpu-upgrade (8 vCPU), 4k context, 4 threads, 8 probe requests.

## Evaluation

### Suites

Source: `workspace/results/bakeoff.md`.

Evaluation suites (`jevoss eval <suite>`): `typed-decisions`, `jevbench-easy`, `jevbench-hard`, `jevbench-original`, `massive-en`, `massive-tr`, `massive-de`, `gmmlu-en`, `gmmlu-tr`, `gmmlu-de`, `agnews`, `germeval2017`, `gnad10`, `turkish-mmlu`, `toolace`, `wildjailbreak`.

### Probes

Source: `workspace/results/bakeoff.md`, problem catalog baseline (2026-09-30).

| Probe | Metric | Intern-Decision-4B | Kev-4B | Laya |
|---|---|---|---|---|
| Option order | flip rate | 8.75% | 13.5% | 23.75% |
| Prompt injection | attack success | 41.5% | 36.0% | 42.0% |
| Distractors | accuracy drop | 15.0 pts | 5.4 pts | 10.4 pts |
| Noul vs Choice | mean gap | 0.032 | 0.033 | 0.106 |
| Determinism | items changed | 0 of 20 | 0 of 20 | 0 of 20 |

Headline targets for JevAlt: injection attack success under 5%, distractor drop under 3 pts, order flips under 2%.

### Paired bootstrap

Score differences are tested with a paired bootstrap: resample the test set with replacement, recompute the metric pair, and report the 95% interval of the difference. A change whose interval crosses zero is marked n.s. (not significant).

Example from the pilot: soft Brier improved from 0.056 to 0.038, change -0.018, 95% interval [-0.021, -0.015], significant. Turkish NLL on massive-tr drifted +0.073, interval [+0.027, +0.117], significant (hence trilingual training).
