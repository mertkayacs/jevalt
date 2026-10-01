"""The compiler must reproduce Intern-Decision's prompt byte for byte.

Fixtures: ``intern_schema.py`` is Intern-Decision's compiler and
``chat_template.jinja`` is the Intern-Decision-4B (Qwen3.5) chat template,
both Apache-2.0.
"""

import importlib.util
import sys
from pathlib import Path

import jinja2
import jinja2.sandbox
import pytest

from jevalt.format import ANSWER_SYMBOLS, compile_request, confidence, render

FIXTURES = Path(__file__).parent / "fixtures"


def _intern():
    spec = importlib.util.spec_from_file_location("intern_schema", FIXTURES / "intern_schema.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _template(messages):
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, **kw: __import__("json").dumps(x, ensure_ascii=False, **kw)
    template = env.from_string((FIXTURES / "chat_template.jinja").read_text())
    return template.render(messages=messages, add_generation_prompt=False, enable_thinking=False)


REQUESTS = [
    {
        "state": "Help! My payouts have been failing for 3 days.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages", "sales": "Pricing"},
            },
            "frustration": {"type": "score", "instructions": "How frustrated?", "criteria": ["Calm", "Frustrated", "Very angry"]},
            "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        },
    },
    {
        "state": {"ticket": {"id": 7, "body": "Kargo gelmedi, iade istiyorum.\nAcil!"}, "tags": ["tr", "shipping"]},
        "questions": {
            "iade": {
                "type": "noul",
                "instructions": "Müşteri iade istiyor mu?",
                "criteria": {"true": "Açıkça iade istiyor", "false": "İade istemiyor"},
            }
        },
    },
]


@pytest.mark.parametrize("request_", REQUESTS)
def test_messages_match_intern(request_):
    theirs = _intern().compile_row(request_, include_targets=False)
    ours = compile_request(request_)
    assert ours.messages == theirs.messages
    assert tuple(f.name for f in ours.fields) == theirs.fields
    assert {f.name: f.symbols for f in ours.fields} == theirs.symbols


@pytest.mark.parametrize("request_", REQUESTS)
def test_render_matches_chat_template(request_):
    compiled = compile_request(request_)
    assert render(compiled) == _template(compiled.messages)


def test_reasoning_renders_inside_think():
    compiled = compile_request(REQUESTS[0], reasoning="The payouts failed for three days.")
    rendered = render(compiled)
    assert rendered == _template(compiled.messages)
    assert "<think>\nThe payouts failed for three days.\n</think>\n\n{" in rendered


def test_open_render_stops_after_think():
    compiled = compile_request(REQUESTS[0])
    assert render(compiled, close=False).endswith("<|im_start|>assistant\n<think>\n")


def test_abstain_adds_localized_unknown_to_choice_and_score_only():
    compiled = compile_request(REQUESTS[0], abstain=True, language="tr")
    kinds = {f.name: f for f in compiled.fields}
    assert kinds["department"].values[-1] == "unknown"
    assert kinds["frustration"].values[-1] == "unknown"
    assert kinds["is_urgent"].values == ("no", "yes")
    assert "Durum, karar vermek için yeterli bilgiyi içermiyor." in compiled.messages[1]["content"]


def test_option_limit():
    too_many = {"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {f"o{i}": None for i in range(len(ANSWER_SYMBOLS) + 1)}}}}
    with pytest.raises(ValueError):
        compile_request(too_many)


def test_null_description_has_no_trailing_colon():
    req = {"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": "desc"}}}}
    user = compile_request(req).messages[1]["content"]
    assert "    A = a\n" in user and "    B = b: desc" in user


def test_confidence_matches_typesafe_examples():
    assert round(confidence([0.88, 0.12, 0.0]), 2) == 0.82
    assert round(confidence([0.0, 0.95, 0.05]), 3) == 0.925
    assert confidence([1 / 3] * 3) == pytest.approx(0.0)
