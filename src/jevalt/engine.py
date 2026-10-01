"""Backend-agnostic decision engine: request in, Jev-compatible response out.

A backend only has to do two things: return logits for each field's answer
symbols given a rendered prompt, and (for reasoning) continue a prompt until
``</think>``. Everything else, calibration, reasoning modes and the response
shape, lives here so every backend behaves the same.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .format import Compiled, UNKNOWN, compile_request, confidence, render

REASONING_MODES = {"off", "on", "auto"}


class Backend(Protocol):
    name: str

    def field_logits(self, text: str, compiled: Compiled) -> list[list[float]]: ...

    def continue_reasoning(self, prefix: str, max_tokens: int) -> str: ...

    def count_tokens(self, text: str) -> int: ...


TR_WORDS = r"\b(ve|bir|bu|için|icin|değil|degil|lütfen|lutfen|çok|cok|ama|gibi|ile|mi|mı|mu|mü|ben|biz|siz|istiyorum|gelmedi|yok|var)\b"
DE_WORDS = r"\b(und|nicht|der|die|das|ist|ich|sie|mit|für|auf|bitte|ein|eine|zu|wir|kein|keine|wurde)\b"


def detect_language(text: str) -> str:
    """Cheap hint for localized option text; callers can always pass `language`."""
    lower = text.lower()
    tr = 2 * len(re.findall(r"[ğış]", lower)) + len(re.findall(TR_WORDS, lower))
    de = 2 * len(re.findall(r"[äß]", lower)) + len(re.findall(DE_WORDS, lower))
    if tr > de:
        return "tr"
    if de > tr:
        return "de"
    return "en"


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    top = max(logits)
    weights = [math.exp((x - top) / temperature) for x in logits]
    total = sum(weights)
    return [w / total for w in weights]


class DecisionEngine:
    def __init__(self, backend: Backend, *, model: str, calibration: Mapping[str, Any] | None = None, max_reasoning_tokens: int = 256):
        self.backend = backend
        self.model = model
        self.calibration = dict(calibration or {})
        self.max_reasoning_tokens = max_reasoning_tokens

    def temperature(self, kind: str, lang: str) -> float:
        temps = self.calibration.get("temperatures", {})
        return float(temps.get(f"{kind}:{lang}", temps.get(kind, temps.get("default", 1.0))))

    def conformal_set(self, kind: str, lang: str, table: dict[str, float], coverage: float) -> tuple[list[str], float] | None:
        thresholds = self.calibration.get("conformal", {})
        group = thresholds.get(f"{kind}:{lang}") or thresholds.get(kind) or thresholds.get("default")
        if not group:
            return None
        fitted = sorted(float(k) for k in group)
        level = next((l for l in fitted if l >= coverage), None)
        if level is None:
            return None
        q = float(group[str(level)])
        ranked = sorted(table, key=lambda v: -table[v])
        members = [v for v in ranked if 1.0 - table[v] <= q]
        return (members or ranked[:1], level)

    def auto_threshold(self, kind: str) -> float:
        return float(self.calibration.get("auto_threshold", {}).get(kind, 0.6))

    def _score(self, request: Mapping[str, Any], compiled: Compiled, lang: str) -> dict[str, dict[str, Any]]:
        coverage = request.get("coverage")
        text = render(compiled)
        logits = self.backend.field_logits(text, compiled)
        answers: dict[str, dict[str, Any]] = {}
        for field, row in zip(compiled.fields, logits):
            probs = softmax(row, self.temperature(field.kind, lang))
            table = dict(zip(field.values, probs))
            answer: dict[str, Any] = {"type": field.kind}
            if field.kind == "noul":
                answer["noul"] = table["yes"]
            else:
                answer["probabilities"] = table
                answer["confidence"] = confidence(probs)
                if field.kind == "choice":
                    answer["choice"] = max(field.values, key=lambda v: (table[v], -field.values.index(v)))
                else:
                    question = request["questions"][field.name]
                    levels = [v for v in field.values if v != UNKNOWN]
                    mass = sum(table[v] for v in levels) or 1.0
                    # Expected level position (0-based): list criteria are numbered, named levels count in their given order.
                    answer["score"] = sum(i * table[v] for i, v in enumerate(levels)) / mass
                    criteria = question["criteria"]
                    answer["legend"] = {v: (criteria[int(v)] if isinstance(criteria, list) else criteria[v]) for v in levels}
                if UNKNOWN in table:
                    answer["unknown"] = table[UNKNOWN]
                if coverage:
                    result = self.conformal_set(field.kind, lang, table, float(coverage))
                    if result is not None:
                        answer["set"], answer["coverage"] = result
                    else:
                        answer["coverage"] = None
            answers[field.name] = answer
        return answers

    def _uncertain(self, answers: Mapping[str, Mapping[str, Any]]) -> list[str]:
        out = []
        for name, a in answers.items():
            conf = a.get("confidence")
            if conf is None:  # noul: distance from a coin flip
                conf = abs(2 * a["noul"] - 1)
            if conf < self.auto_threshold(a["type"]):
                out.append(name)
        return out

    def predict(self, request: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        mode = request.get("reasoning", "off") or "off"
        if mode not in REASONING_MODES:
            raise ValueError(f"reasoning must be one of {sorted(REASONING_MODES)}")
        abstain = bool(request.get("abstain", False))
        state_text = request["state"] if isinstance(request.get("state"), str) else str(request.get("state"))
        lang = request.get("language") or detect_language(state_text + " " + str(request.get("questions")))
        compiled = compile_request(request, abstain=abstain, language=lang)
        answers = self._score(request, compiled, lang)

        trace = None
        redo = list(answers) if mode == "on" else self._uncertain(answers) if mode == "auto" else []
        if redo:
            trace = self.backend.continue_reasoning(render(compiled, close=False), self.max_reasoning_tokens).strip()
            thought = compile_request(request, reasoning=trace, abstain=abstain, language=lang)
            second = self._score(request, thought, lang)
            for name in redo:
                answers[name] = {**second[name], "reasoning": trace}

        text = render(compiled)
        response: dict[str, Any] = {
            "model": self.model,
            "answers": answers,
            "usage": {"input_tokens": self.backend.count_tokens(text), "output_tokens": len(answers) + (self.backend.count_tokens(trace) if trace else 0)},
        }
        if mode != "off":
            response["reasoning_mode"] = mode
            response["reasoned_fields"] = redo
        response["timing_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return response
