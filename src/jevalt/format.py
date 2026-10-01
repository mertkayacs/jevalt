"""Decision prompt compiler.

Turns a Jev-style request (state + typed questions) into chat messages whose
assistant turn is a JSON skeleton with one ``<decision>`` marker per field.
The model's next-token logits just before each marker, restricted to that
field's answer symbols, give the field's probability distribution.

With default settings the output is byte-identical to Intern-Decision's
compiler, so the start checkpoint sees exactly the format it was trained on.
Extensions: an optional reasoning trace inside ``<think>`` and an optional
``unknown`` option for Choice and Score questions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DECISION_TOKEN = "<decision>"
ANSWER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
MAX_OPTIONS = len(ANSWER_SYMBOLS)
UNKNOWN = "unknown"

SYSTEM_PROMPT = (
    "You are a careful decision assistant. Use the state and decision schema in "
    "the user message to make the requested decisions. For every field, choose "
    "exactly one answer symbol (e.g. A, B, C, ...) from its listed options and "
    "return one valid JSON object mapping each field name to its chosen symbol. "
    "Use the field names and symbols exactly as given. Do not include "
    "explanations, Markdown, or extra text."
)

NOUL_YES = "The answer is yes (affirmative, or align with the claim)."
NOUL_NO = "The answer is no (negative, or disagree with the claim)."

UNKNOWN_TEXT = {
    "en": "The state does not contain enough information to decide.",
    "tr": "Durum, karar vermek için yeterli bilgiyi içermiyor.",
    "de": "Der Zustand enthält nicht genügend Informationen, um zu entscheiden.",
}


@dataclass(frozen=True)
class Field:
    name: str
    kind: str  # choice | score | noul
    values: tuple[str, ...]  # option values in prompt order
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class Compiled:
    messages: list[dict[str, Any]]
    fields: tuple[Field, ...]


def _text(value: Any) -> str:
    """Strings pass through; structured values render as compact JSON."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def options(question: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Ordered (value, description) pairs for a question."""
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, Mapping) or not criteria:
            raise ValueError("choice criteria must be a non-empty object")
        return [(str(k), _text(v)) for k, v in criteria.items()]
    if kind == "score":
        if isinstance(criteria, list) and len(criteria) >= 2:
            return [(str(i), _text(v)) for i, v in enumerate(criteria)]
        if isinstance(criteria, Mapping) and len(criteria) >= 2:
            return [(str(k), _text(v)) for k, v in criteria.items()]
        raise ValueError("score criteria must list at least two levels")
    if kind == "noul":
        described = criteria if isinstance(criteria, Mapping) else {}
        yes = next((_text(described[k]) for k in described if str(k).lower() in {"yes", "true", "1"}), NOUL_YES)
        no = next((_text(described[k]) for k in described if str(k).lower() in {"no", "false", "0"}), NOUL_NO)
        return [("no", no), ("yes", yes)]
    raise ValueError(f"unsupported question type: {kind!r}")


def compile_request(
    request: Mapping[str, Any],
    *,
    reasoning: str | None = None,
    abstain: bool = False,
    language: str = "en",
) -> Compiled:
    questions = request.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("questions must be a non-empty object")

    fields: list[Field] = []
    lines: list[str] = []
    for name, question in questions.items():
        if not isinstance(question, Mapping):
            raise ValueError(f"question {name!r} must be an object")
        opts = options(question)
        kind = question["type"]
        if abstain and kind in {"choice", "score"} and UNKNOWN not in {v for v, _ in opts}:
            opts.append((UNKNOWN, UNKNOWN_TEXT.get(language, UNKNOWN_TEXT["en"])))
        if len(opts) > MAX_OPTIONS:
            raise ValueError(f"question {name!r} has {len(opts)} options; the limit is {MAX_OPTIONS}")
        symbols = tuple(ANSWER_SYMBOLS[: len(opts)])
        fields.append(Field(str(name), kind, tuple(v for v, _ in opts), symbols))
        lines.append(f"{name}: {_text(question.get('instructions', ''))}")
        for symbol, (value, description) in zip(symbols, opts):
            lines.append(f"    {symbol} = {value}: {description}" if description else f"    {symbol} = {value}")

    state = json.dumps(request.get("state"), ensure_ascii=False, indent=2)
    user = (
        "Return one answer for every field using the supplied answer symbols.\n\n"
        "## State\n" + state + "\n## Decision schema\n" + "\n".join(lines)
    )
    if DECISION_TOKEN in user:
        raise ValueError("the reserved <decision> marker appears in the input")
    skeleton = json.dumps(dict.fromkeys((f.name for f in fields), DECISION_TOKEN), ensure_ascii=False, indent=4)
    assistant: dict[str, Any] = {"role": "assistant", "content": skeleton}
    if reasoning:
        assistant["reasoning_content"] = reasoning.strip()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        assistant,
    ]
    return Compiled(messages=messages, fields=tuple(fields))


def render(compiled: Compiled, *, close: bool = True) -> str:
    """Render the chat exactly like the Qwen3.5 template with enable_thinking=False.

    ``close=False`` stops right after ``<think>\\n`` so a backend can generate a
    reasoning trace, then append ``render_tail``.
    """
    system, user, assistant = compiled.messages
    head = (
        f"<|im_start|>system\n{system['content']}<|im_end|>\n"
        f"<|im_start|>user\n{user['content']}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    if not close:
        return head
    return head + render_tail(compiled, assistant.get("reasoning_content", ""))


def render_tail(compiled: Compiled, reasoning: str = "") -> str:
    skeleton = compiled.messages[2]["content"]
    return f"{reasoning.strip()}\n</think>\n\n{skeleton}<|im_end|>\n"


def confidence(probabilities: list[float]) -> float:
    """TypeSafe-style confidence: 1 when all mass is on one option, 0 when flat."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    return min(1.0, max(0.0, (n * max(probabilities) - 1) / (n - 1)))
