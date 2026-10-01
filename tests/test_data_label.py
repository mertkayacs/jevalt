"""Teacher label parsing, shuffling and aggregation (DATA.md section 4)."""

import pytest

from jevalt.data.label import aggregate, apply_labels, display_order, parse_dist

VALUES = ("billing", "shipping", "tech")


def test_parse_renormalizes_within_tolerance_and_fills_zeros():
    dist, problem = parse_dist({"billing": 0.7, "`Shipping`": 0.29}, VALUES)
    assert problem is None
    assert dist == pytest.approx({"billing": 0.7 / 0.99, "shipping": 0.29 / 0.99, "tech": 0.0})


@pytest.mark.parametrize(
    "raw, problem",
    [
        ({"billing": 0.5, "shipping": 0.4}, "sum"),
        ({"billing": 0.9, "sales": 0.1}, "unknown option"),
        ({"billing": "high"}, "non-numeric"),
        ({"billing": 1.2}, "out of range"),
        (None, "no probabilities"),
    ],
)
def test_parse_rejects(raw, problem):
    dist, got = parse_dist(raw, VALUES)
    assert dist is None and problem in got


def test_display_order_is_a_seeded_permutation_per_teacher():
    many = [f"o{i}" for i in range(12)]
    a = display_order("row1", "f", many, "k3o")
    assert sorted(a) == sorted(many)
    assert a == display_order("row1", "f", many, "k3o")
    assert a != display_order("row1", "f", many, "glm") or a != display_order("row1", "f", many, "ds")


def test_unanimous_gold_is_smoothed_mean():
    dists = {"a": {"billing": 0.9, "shipping": 0.1, "tech": 0.0}, "b": {"billing": 0.8, "shipping": 0.2, "tech": 0.0}, "c": {"billing": 1.0, "shipping": 0.0, "tech": 0.0}}
    gold, reason = aggregate(VALUES, dists)
    assert reason is None
    assert gold["billing"] == pytest.approx(0.98 * 0.9 + 0.02 / 3)
    assert gold["tech"] == pytest.approx(0.02 / 3)
    assert sum(gold.values()) == pytest.approx(1)


def test_two_of_three_majority_is_kept():
    dists = {"a": {"billing": 0.6, "shipping": 0.4, "tech": 0}, "b": {"billing": 0.7, "shipping": 0.3, "tech": 0}, "c": {"billing": 0, "shipping": 0, "tech": 1.0}}
    gold, reason = aggregate(VALUES, dists)
    assert reason is None and max(gold, key=gold.get) == "billing"


def test_no_majority_and_missing_teacher_are_dropped():
    split = {"a": {"billing": 1.0}, "b": {"shipping": 1.0}, "c": {"tech": 1.0}}
    assert aggregate(VALUES, split) == (None, "no_majority")
    missing = {"a": {"billing": 1.0}, "b": {"billing": 1.0}, "c": None}
    assert aggregate(VALUES, missing) == (None, "teacher_missing")


def test_mean_that_contradicts_the_majority_is_dropped():
    # Two teachers lean billing at 0.6, one is certain of shipping: the mean says shipping.
    dists = {"a": {"billing": 0.6, "shipping": 0.4, "tech": 0}, "b": {"billing": 0.6, "shipping": 0.4, "tech": 0}, "c": {"billing": 0, "shipping": 1.0, "tech": 0}}
    assert aggregate(VALUES, dists) == (None, "mean_disagrees_with_majority")


def test_tied_argmax_counts_for_both_labels():
    dists = {"a": {"billing": 0.5, "shipping": 0.5, "tech": 0}, "b": {"billing": 0.9, "shipping": 0.1, "tech": 0}, "c": {"tech": 1.0}}
    gold, reason = aggregate(VALUES, dists)
    assert reason is None and max(gold, key=gold.get) == "billing"


def _row(n):
    q = {"type": "noul", "instructions": "Refund requested?"}
    return {"id": "r1", "lang": "en", "state": "The customer wants the money back.", "questions": {f"q{i}": q for i in range(n)}, "tags": []}


def _dists(agree):
    """Questions marked False lose a teacher answer, so they must be dropped."""
    yes = {"no": 0.1, "yes": 0.9}
    out = {"glm": {}, "ds": {}, "mistral": {}}
    for field, ok in agree.items():
        out["glm"][field] = yes
        out["ds"][field] = yes
        out["mistral"][field] = yes if ok else None
    return out


def test_row_losing_more_than_half_is_dropped():
    row, drops = apply_labels(_row(3), _dists({"q0": True, "q1": False, "q2": False}))
    assert row is None and any(d["reason"] == "row_lost_most_questions" for d in drops)


def test_row_losing_exactly_half_keeps_the_rest():
    row, drops = apply_labels(_row(2), _dists({"q0": True, "q1": False}))
    assert row is not None and list(row["questions"]) == ["q0"] and list(row["targets"]) == ["q0"]
    assert set(row["teachers"]) == {"glm", "ds", "mistral"} and "q1" not in row["teachers"]["glm"]["dist"]
    assert row["targets"]["q0"]["label"] == "yes" and "teachers:3" in row["tags"]


def test_two_teachers_must_both_answer_and_agree():
    both = {"glm": {"billing": 0.8, "shipping": 0.2, "tech": 0}, "k3n": {"billing": 0.6, "shipping": 0.4, "tech": 0}}
    gold, reason = aggregate(VALUES, both)
    assert reason is None and max(gold, key=gold.get) == "billing"
    split = {"glm": {"billing": 1.0}, "k3n": {"shipping": 1.0}}
    assert aggregate(VALUES, split) == (None, "no_majority")
    assert aggregate(VALUES, {"glm": {"billing": 1.0}, "k3n": None}) == (None, "teacher_missing")


def test_fallback_pair_is_tagged():
    yes = {"no": 0.1, "yes": 0.9}
    row, _ = apply_labels(_row(2), {"glm": {"q0": yes, "q1": yes}, "k3n": {"q0": yes, "q1": yes}}, teachers=("glm", "k3n"))
    assert "teachers:2" in row["tags"] and set(row["teachers"]) == {"glm", "k3n"}


def test_trace_writer_policy():
    from jevalt.data.reasoning import candidates, reasoner_for

    labeled_tr = {"lang": "tr", "source": "gen:k3ot", "tags": [], "teachers": {"glm": {}, "ds": {}, "mistral": {}}}
    assert candidates(labeled_tr) == ["k3", "minimax"]  # labeler labs excluded; K3 allowed for TR
    fallback_tr = {**labeled_tr, "teachers": {"glm": {}, "k3n": {}}}
    assert "k3" not in candidates(fallback_tr) and "glmt" not in candidates(fallback_tr)
    assert reasoner_for(fallback_tr, {"zai", "kimi"}) is None  # only Ollama labs remain
    labeled_en = {**labeled_tr, "lang": "en"}
    assert "k3" not in candidates(labeled_en)  # Kimi writes TR/DE traces only
    prog_de = {"lang": "de", "source": "prog:dates", "tags": ["writer:ollama/deepseek-v4-pro:0813"], "teachers": {"glm": {}, "k3n": {}}}
    assert candidates(prog_de)[:2] == ["glmt", "k3"] and "dst" not in candidates(prog_de)


def test_single_probability_on_a_two_option_question_implies_the_complement():
    dist, problem = parse_dist({"yes": 0.95}, ("no", "yes"))
    assert problem is None and dist == pytest.approx({"no": 0.05, "yes": 0.95})
    dist, problem = parse_dist({"billing": 0.95}, VALUES)  # three options: the missing mass is ambiguous
    assert dist is None and "sum" in problem


def test_router_pairs_never_include_the_writer_or_editor_lab(monkeypatch):
    from jevalt.data import label
    from jevalt.data.providers import TEACHERS

    monkeypatch.setattr(label, "HFR", True)
    monkeypatch.setattr(label, "GLM_ONLY", False)
    for writer in ("zhipu", "mistral", "deepseek", "google", "alibaba", "moonshot", ""):
        for editor in (None, "zhipu", "deepseek"):
            pair = label.pair_for(writer, editor)
            free = [t for t in label.ROUTER_LABELERS if TEACHERS[t].lab not in {writer, editor}]
            if len(free) < 2:
                assert pair is None  # the row waits rather than getting a dependent label
                continue
            labs = {TEACHERS[t].lab for t in pair}
            assert len(labs) == 2 and not labs & {writer, editor} and all(TEACHERS[t].provider == "hfr" for t in pair)
    assert label.pair_for("zhipu") == ("dsf-r", "qwen35-r")
    assert label.pair_for("mistral", "deepseek") == ("qwen35-r", "gemma26-n")
    assert {TEACHERS[t].lab for t in label.fix_teachers("tr")} == {"google", "alibaba"}


def test_router_traces_prefer_deepseek_v4_pro(monkeypatch):
    from jevalt.data import reasoning

    monkeypatch.setattr(reasoning, "HFR", True)
    row = {"lang": "tr", "source": "gen:mistral", "tags": [], "teachers": {"gemma26-n": {}, "qwen35-r": {}}}
    assert reasoning.candidates(row) == ["ds-r", "glmt"]
    assert reasoning.reasoner_for(row, {"zai"}) == "glmt"  # router down: GLM thinking
    labeled_by_deepseek = {**row, "teachers": {"dsf-r": {}, "qwen35-r": {}}}
    assert reasoning.candidates(labeled_by_deepseek) == ["glmt"]


def test_writer_filter_reads_target_records():
    from jevalt.data.label import writer_argmax_disagrees

    row = {"state": "s", "questions": {"team": {"type": "choice", "criteria": {"billing": "b", "tech": "t"}}, "urgent": {"type": "noul"}},
           "writer": {"dist": {"team": {"billing": 0.8, "tech": 0.2}, "urgent": {"yes": 0.3, "no": 0.7}}}}
    targets = {"team": {"label": "billing", "dist": {"billing": 0.9, "tech": 0.1}}, "urgent": {"label": "yes", "dist": {"yes": 0.6, "no": 0.4}}}
    assert writer_argmax_disagrees(row, targets, ("a", "b")) == ["urgent"]
    assert writer_argmax_disagrees({**row, "writer": {}}, targets, ("a", "b")) == []
