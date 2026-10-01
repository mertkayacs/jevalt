import math

import pytest

from jevalt.engine import DecisionEngine, detect_language
from jevalt.format import Compiled


class StubBackend:
    """First symbol always wins by one logit unless the prompt carries reasoning."""

    name = "stub"

    def __init__(self):
        self.reasoning_calls = 0

    def field_logits(self, text: str, compiled: Compiled):
        boost = 3.0 if "because" in text else 0.0
        return [[1.0 + boost] + [0.0] * (len(f.symbols) - 1) for f in compiled.fields]

    def continue_reasoning(self, prefix: str, max_tokens: int) -> str:
        self.reasoning_calls += 1
        return "The payouts failed because the bank rejected them."

    def count_tokens(self, text: str) -> int:
        return len(text.split())


REQUEST = {
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs", "sales": "deals"}},
        "anger": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "upset", "furious"]},
        "urgent": {"type": "noul", "instructions": "Urgent?"},
    },
}


def test_response_shape_matches_jev():
    out = DecisionEngine(StubBackend(), model="jevalt-test").predict(REQUEST)
    team, anger, urgent = (out["answers"][k] for k in ("team", "anger", "urgent"))
    assert team["choice"] == "billing" and set(team["probabilities"]) == {"billing", "tech", "sales"}
    assert math.isclose(sum(team["probabilities"].values()), 1.0)
    assert anger["legend"] == {"0": "calm", "1": "upset", "2": "furious"}
    expected = sum(float(k) * p for k, p in anger["probabilities"].items())
    assert math.isclose(anger["score"], expected)
    assert set(urgent) == {"type", "noul"} and 0 < urgent["noul"] < 1
    assert "reasoning_mode" not in out


def test_temperature_softens_probabilities():
    raw = DecisionEngine(StubBackend(), model="m").predict(REQUEST)["answers"]["team"]["probabilities"]["billing"]
    hot = DecisionEngine(StubBackend(), model="m", calibration={"temperatures": {"choice:en": 2.0}}).predict(REQUEST)
    assert hot["answers"]["team"]["probabilities"]["billing"] < raw


def test_abstain_reports_unknown_and_keeps_score_on_levels():
    out = DecisionEngine(StubBackend(), model="m").predict({**REQUEST, "abstain": True})
    assert "unknown" in out["answers"]["team"] and "unknown" in out["answers"]["anger"]
    assert "unknown" not in out["answers"]["anger"]["legend"]


def test_reasoning_on_rescoring_uses_trace():
    backend = StubBackend()
    out = DecisionEngine(backend, model="m").predict({**REQUEST, "reasoning": "on"})
    assert backend.reasoning_calls == 1
    assert out["answers"]["team"]["reasoning"].startswith("The payouts failed")
    assert out["reasoned_fields"] == ["team", "anger", "urgent"]


def test_reasoning_auto_only_touches_uncertain_fields():
    backend = StubBackend()
    engine = DecisionEngine(backend, model="m", calibration={"auto_threshold": {"choice": 0.99, "score": 0.0, "noul": 0.0}})
    out = engine.predict({**REQUEST, "reasoning": "auto"})
    assert out["reasoned_fields"] == ["team"]
    assert "reasoning" not in out["answers"]["anger"]


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        DecisionEngine(StubBackend(), model="m").predict({**REQUEST, "reasoning": "maybe"})


@pytest.mark.parametrize("text,lang", [("Kargo gelmedi, iade istiyorum", "tr"), ("Die Lieferung ist nicht angekommen", "de"), ("My parcel never arrived", "en"), ("Müşteri ığdır şehrinde", "tr")])
def test_language_hint(text, lang):
    assert detect_language(text) == lang


def test_conformal_set_uses_fitted_threshold():
    cal = {"conformal": {"choice": {"0.9": 0.9}}}
    out = DecisionEngine(StubBackend(), model="m", calibration=cal).predict({**REQUEST, "coverage": 0.9})
    team = out["answers"]["team"]
    assert team["set"][0] == team["choice"] and all(1 - team["probabilities"][v] <= 0.9 for v in team["set"])
    assert "set" not in DecisionEngine(StubBackend(), model="m").predict({**REQUEST, "coverage": 0.9})["answers"]["team"]


def test_named_score_levels_count_by_position():
    named = {**REQUEST, "questions": {"stage": {"type": "score", "instructions": "How far?", "criteria": {"niedrig": "low", "mittel": "mid", "hoch": "high"}}}}
    anger = DecisionEngine(StubBackend(), model="m").predict(REQUEST)["answers"]["anger"]
    stage = DecisionEngine(StubBackend(), model="m").predict(named)["answers"]["stage"]
    assert stage["legend"] == {"niedrig": "low", "mittel": "mid", "hoch": "high"}
    assert math.isclose(stage["score"], sum(i * stage["probabilities"][k] for i, k in enumerate(("niedrig", "mittel", "hoch"))))
    assert math.isclose(stage["score"], anger["score"])  # same logits, same positions
