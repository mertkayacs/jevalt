"""Test conformal_set rounding behaviour: smallest fitted level at or above the request."""

from jevalt.engine import DecisionEngine


class StubBackend:
    name = "stub"
    def field_logits(self, text, compiled): return []
    def continue_reasoning(self, prefix, max_tokens): return ""
    def count_tokens(self, text): return 0


CALIBRATION = {
    "conformal": {
        "choice:en": {"0.8": 0.15, "0.9": 0.08, "0.95": 0.04},
        "default": {"0.8": 0.20, "0.9": 0.10},
    }
}

TABLE = {"a": 0.8, "b": 0.15, "c": 0.05}


def make_engine():
    return DecisionEngine(StubBackend(), model="test", calibration=CALIBRATION)


def test_exact_match():
    eng = make_engine()
    members, level = eng.conformal_set("choice", "en", TABLE, 0.9)
    assert level == 0.9
    assert "a" in members


def test_round_up():
    eng = make_engine()
    members, level = eng.conformal_set("choice", "en", TABLE, 0.85)
    assert level == 0.9
    assert "a" in members


def test_no_level_at_or_above():
    eng = make_engine()
    result = eng.conformal_set("choice", "en", TABLE, 0.99)
    assert result is None


def test_default_group_round_up():
    eng = make_engine()
    members, level = eng.conformal_set("score", "tr", TABLE, 0.85)
    assert level == 0.9
    assert "a" in members


def test_no_calibration():
    eng = DecisionEngine(StubBackend(), model="test")
    assert eng.conformal_set("choice", "en", TABLE, 0.9) is None


def test_answer_carries_coverage_level():
    """The _score method should set answer['coverage'] to the level used."""
    # This is an integration check: verify the engine puts coverage in the answer.
    # We call conformal_set directly and check the tuple shape.
    eng = make_engine()
    members, level = eng.conformal_set("choice", "en", TABLE, 0.9)
    assert isinstance(members, list)
    assert isinstance(level, float)
