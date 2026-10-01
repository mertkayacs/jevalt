# API

JevAlt speaks TypeSafe's `POST /v1/systemone`. A request that works with Jev works here without changes. A few optional fields add what Jev does not have.

## Request

```json
{
  "state": "Hi, we were billed twice for March. Please refund the duplicate today.",
  "model": "jevalt",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "Invoices, payments, refunds", "technical": "Bugs and outages"}
    },
    "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["Can wait", "Today", "Now"]},
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"}
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `state` | string, object or array | The evidence. Only put what the questions need in it. |
| `questions` | map of id to question | Answers come back under the same ids. |
| `model` | string | Accepted and ignored by a local server; the response names the model that answered. |

Question types:

| Type | `criteria` | Answer |
|---|---|---|
| `choice` | map of option to description (up to 62 options) | `choice`, `probabilities`, `confidence` |
| `score` | ordered list of 2 to 10 level descriptions | `score` (probability-weighted level), `legend`, `probabilities`, `confidence` |
| `noul` | optional `{"true": ..., "false": ...}` | `noul`, the probability of yes |

`confidence` is `(n * p_max - 1) / (n - 1)`: 1 when all mass sits on one option, 0 when the distribution is flat. It matches the worked examples in TypeSafe's documentation.

## When instructions and criteria disagree

The criteria win. If a Noul asks "Was it on time?" but its criteria say `yes` means "arrived after the deadline", JevAlt answers by the criteria. Jev 1.13 gets confused in this case (TypeSafe lists it as a known weakness); JevAlt is trained on such pairs so the rule holds.

## Extensions

All optional. Leave them out and the response has exactly Jev's shape.

| Field | Values | What it adds |
|---|---|---|
| `reasoning` | `"off"` (default), `"on"`, `"auto"` | `on`: the model writes a short reasoning trace, then decides. `auto`: it decides first and thinks again only for answers below a confidence threshold. Answers that used a trace carry it in `reasoning`. |
| `abstain` | `true` | Adds an `unknown` option to every Choice and Score. Its probability comes back as `unknown`. |
| `coverage` | `0.8`, `0.9`, `0.95` | Adds `set`: the smallest group of options that contains the right answer with at least this probability, from conformal thresholds fitted on held-out data. If the requested level has no fitted threshold, the smallest fitted level at or above it is used. The level used is returned as `coverage` in the answer. If no fitted level is at or above the request, no set is returned and `coverage` is null. |
| `language` | `"en"`, `"tr"`, `"de"` | Language for the `unknown` wording and the reasoning trace. Detected from the state when missing. |

## Response

```json
{
  "model": "Deem-4B",
  "answers": {
    "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.93, "technical": 0.07}, "confidence": 0.86},
    "urgency": {"type": "score", "score": 1.4, "legend": {"0": "Can wait", "1": "Today", "2": "Now"}, "probabilities": {"0": 0.05, "1": 0.5, "2": 0.45}, "confidence": 0.25},
    "refund": {"type": "noul", "noul": 0.97}
  },
  "usage": {"input_tokens": 212, "output_tokens": 3}
}
```

## Errors

| Status | Meaning |
|---|---|
| 401 | The server was started with `JEVALT_API_KEY` and the request did not send it as a bearer token. |
| 422 | The body is not valid: missing `state`, empty `questions`, an unknown type, or too many options. The message names the problem. |
