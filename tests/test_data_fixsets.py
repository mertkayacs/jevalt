"""F1-F3 builders end to end with a fake teacher client (no network)."""

import asyncio

import pytest

from jevalt.data import fixsets
from jevalt.format import UNKNOWN

TEACHERS = ("glm", "ds", "mistral")


def g_row(field="refund", kind="noul", yes=0.9):
    q = {"type": kind, "instructions": "Does the customer ask for a refund?"}
    if kind == "choice":
        q = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "shipping": "Parcels", "tech": "Bugs"}}
        dist = {"billing": 0.9, "shipping": 0.05, "tech": 0.05}
    else:
        dist = {"no": 1 - yes, "yes": yes}
    top = max(dist, key=dist.get)
    return {
        "id": f"en-g-{field}", "lang": "en", "source": "gen:dst", "license": "apache-2.0", "split": "train",
        "tags": ["domain:customer_support", f"group:en-g-{field}"],
        "state": "Hello, my parcel never arrived and I want my money back for order 4411, please.",
        "questions": {field: q}, "targets": {field: {"label": top, "dist": dist}},
        "teachers": {t: {"model": t, "dist": {field: dist}} for t in TEACHERS},
    }


class FakeClient:
    def __init__(self, reply):
        self.reply = reply

    async def ask_json(self, teacher, messages, *, check=None, **kw):
        data = self.reply
        return (check(data) if check else data), type("Reply", (), {"model": "fake"})()

    async def available(self, provider):
        return True


def fake_dists(answer):
    async def dists(client, rows, *, teachers=TEACHERS, unknown=False, **kw):
        return {row["id"]: {t: {f: answer(row, f, unknown) for f in row["questions"]} for t in teachers} for row in rows}
    return dists


def test_f2_builds_complementary_twins(monkeypatch):
    client = FakeClient({"items": [{"id": "q1", "field": "no_refund_request", "instructions": "Is it true that the customer does not ask for a refund?"}]})
    monkeypatch.setattr(fixsets, "teacher_dists", fake_dists(lambda row, f, u: {"no": 0.85, "yes": 0.15}))
    rows, drops = asyncio.run(fixsets.build_f2(client, [g_row()], 16, "t", TEACHERS))
    assert not drops and len(rows) == 2
    pos, neg = rows
    assert pos["targets"]["refund"]["dist"] == pytest.approx({"no": 0.1, "yes": 0.9})
    assert neg["targets"]["no_refund_request"]["dist"] == pytest.approx({"no": 0.9, "yes": 0.1})
    assert {t for t in pos["tags"] if t.startswith("twin:")} == {t for t in neg["tags"] if t.startswith("twin:")}


def test_f2_drops_a_negation_the_teachers_do_not_flip(monkeypatch):
    client = FakeClient({"items": [{"id": "q1", "field": "refund_again", "instructions": "Does the customer want a refund?"}]})
    monkeypatch.setattr(fixsets, "teacher_dists", fake_dists(lambda row, f, u: {"no": 0.1, "yes": 0.9}))
    rows, drops = asyncio.run(fixsets.build_f2(client, [g_row()], 16, "t", TEACHERS))
    assert not rows and drops[0]["reason"] == "negation_not_complementary"


def test_f3_builds_a_choice_twin_with_the_same_probability(monkeypatch):
    client = FakeClient({"items": [{"id": "q1", "field": "request_kind", "instructions": "What does the customer want?",
                                    "options": {"refund_requested": "Asks for money back", "no_refund": "Does not ask"}, "yes_option": "refund_requested"}]})
    monkeypatch.setattr(fixsets, "teacher_dists", fake_dists(lambda row, f, u: {"refund_requested": 0.88, "no_refund": 0.12}))
    rows, drops = asyncio.run(fixsets.build_f3(client, [g_row()], 16, "t", TEACHERS))
    assert not drops and len(rows) == 2
    choice = rows[1]
    assert choice["targets"]["request_kind"]["dist"]["refund_requested"] == pytest.approx(0.9)
    assert choice["questions"]["request_kind"]["type"] == "choice"


def test_f1_builds_unknown_prior_and_control_rows(monkeypatch):
    client = FakeClient({"feasible": True, "deciding_facts": ["parcel never arrived"], "state": "Hello, I have a question about order 4411, please get back to me.", "removed": "the problem"})

    def answer(row, f, unknown):
        if unknown:
            return {"billing": 0.1, "shipping": 0.1, "tech": 0.1, UNKNOWN: 0.7}
        return {"billing": 0.4, "shipping": 0.35, "tech": 0.25}

    monkeypatch.setattr(fixsets, "teacher_dists", fake_dists(answer))
    monkeypatch.setattr(fixsets, "language_check", lambda row: {"ok": True})
    rows, drops = asyncio.run(fixsets.build_f1(client, [g_row("team", "choice")], 16, "t", TEACHERS))
    assert not drops and len(rows) == 3
    removed_unknown, removed_prior, control = rows
    assert removed_unknown["targets"]["team"]["label"] == UNKNOWN and "abstain" in removed_unknown["tags"]
    assert removed_prior["targets"]["team"]["label"] == "billing" and "abstain" not in removed_prior["tags"]
    assert control["state"] == g_row("team", "choice")["state"] and control["targets"]["team"]["dist"][UNKNOWN] < 0.01


def test_f1_skips_holistic_questions(monkeypatch):
    client = FakeClient({"feasible": False})
    monkeypatch.setattr(fixsets, "teacher_dists", fake_dists(lambda row, f, u: {}))
    rows, drops = asyncio.run(fixsets.build_f1(client, [g_row("team", "choice")], 16, "t", TEACHERS))
    assert not rows and drops[0]["reason"] == "infeasible"


def prog_row(i, tpl, label="yes"):
    return {"id": f"en-f5-{i}", "lang": "en", "source": "prog:dates", "tags": [f"tpl:{tpl}", "writer:ollama/deepseek-v4-pro:0813"],
            "state": "x" * 30, "questions": {"on_time": {"type": "noul", "instructions": "On time?"}},
            "targets": {"on_time": {"label": label, "dist": {"no": 0.01, "yes": 0.99} if label == "yes" else {"no": 0.99, "yes": 0.01}}}}


def test_disputed_template_needs_the_reasoning_verifier_to_agree_before_it_is_dropped(monkeypatch):
    rows = [prog_row(i, "t-good") for i in range(2)] + [prog_row(i + 2, "t-bad") for i in range(2)]
    wrong = {"no": 0.9, "yes": 0.1}

    async def dists(client, rows_, *, teachers=TEACHERS, **kw):
        # the ensemble gets every row wrong; the verifier (glmt) is right on t-good only
        verifier_run = teachers == ("glmt",)
        return {r["id"]: {t: {"on_time": ({"no": 0.1, "yes": 0.9} if verifier_run and "good" in r["tags"][0] else wrong)} for t in teachers} for r in rows_}

    monkeypatch.setattr(fixsets, "teacher_dists", dists)
    kept, drops, stats = asyncio.run(fixsets.verify_programmatic(FakeClient(None), rows, teachers=TEACHERS))
    assert {r["id"] for r in kept} == {"en-f5-0", "en-f5-1"}
    assert {d["tpl"] for d in drops} == {"t-bad"} and stats["t-good"]["verifier"] == "glmt"
