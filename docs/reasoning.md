# Reasoning modes

Jev answers in one pass. That is fast, and TypeSafe's own notes say where it hurts: counting, date comparisons, negated questions, and anything that needs two or three steps. JevAlt keeps the one-pass answer and adds an optional short trace in front of it.

| `reasoning` | What happens | When to use it |
|---|---|---|
| `off` | One forward pass. Same speed as a plain decision. | Routing, tagging, moderation at volume. |
| `on` | The model writes a short trace (at most 256 tokens), then scores the options with the trace in view. | Policies with exceptions, dates, sums, multi-step questions. |
| `auto` | Decide first. Only answers with confidence below a per-type threshold get a second pass with a trace. | A good default when most requests are easy and a few are not. |

The trace is written in the language of the request: English for Deem-4B, Turkish for Karar-4B, German for Wähler-4B. It comes back in `answers[id].reasoning` so you can log it or show it to a reviewer.

A request to Deem-4B (the output below is real, from the public Space):

```json
{
  "state": {"policy": "Refunds need a receipt and a purchase within 30 days.", "purchase_date": "2026-08-20", "today": "2026-09-30", "receipt": true},
  "questions": {"refund": {"type": "noul", "instructions": "Does the policy allow a refund?"}},
  "reasoning": "on"
}
```

```json
{
  "refund": {
    "type": "noul",
    "noul": 0.019,
    "reasoning": "The policy requires a receipt and a purchase within 30 days. The receipt is present, but the purchase date of 2026-08-20 is 41 days before 2026-09-30, exceeding the 30-day limit. Since the time requirement is not met, the refund is not allowed. The answer is no."
  }
}
```

With `"reasoning": "off"` the same request returns `noul` 0.23: the answer is already no, with less certainty. `"auto"` returns the `off` answer here, because Deem-4B's fitted threshold for Noul questions is 0 (see below).

## How the trace is used

The probabilities still come from the decision head, the same way as in `off` mode. The trace only changes what the model has read before it scores the options. So every answer keeps the Jev shape and its calibration, and the trace never turns into free text you have to parse.

## What the thresholds say

`auto` uses a confidence threshold per question type, fitted on held-out calibration rows: below it, the answer is redone with a trace. A threshold of 0 means a trace never helped on those rows, so `auto` behaves like `off` for that type. The fitted values, from 60 calibration rows with reasoning on (20 per language):

| Model | Choice | Noul | Score |
|---|---|---|---|
| Deem-4B | 0.5 | 0 | 0 |
| Karar-4B | 0 | 0.05 | 0.2 |
| Wähler-4B | 0.25 | 0 | 0 |

On those rows a trace raised Deem-4B's Choice accuracy from 0.901 to 0.930 when used below 0.5 confidence (on 13% of Choice answers), and lowered its Noul accuracy when used everywhere (0.946 to 0.893). On the held-out date, number and policy rows, `auto` moved Deem-4B from 0.761 to 0.769 accuracy, left Karar-4B unchanged, and made Wähler-4B's Brier worse by 0.11. Refit the thresholds on your own traffic with JevOss before you rely on `auto`.

## Cost

A trace adds generated tokens. On one L4 GPU with the transformers backend, the 134 English fix-set test decisions took 9.0 s with `off` and 16.9 s with `auto`; the 38 German ones took 2.8 s and 32.7 s. `auto` pays that price only for the answers below its threshold.
