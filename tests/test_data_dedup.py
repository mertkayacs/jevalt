"""Exact and MinHash dedup on state text, group-aware; writer reply parsing."""

from jevalt.data.generate import split_blocks
from jevalt.data.qa import dedup

TEXT = (
    "Guten Tag, ich habe am 12. September eine Waschmaschine bestellt und sie ist bis heute nicht geliefert worden. "
    "Die Sendungsverfolgung zeigt seit einer Woche denselben Status. Bitte teilen Sie mir mit, wann ich mit der Lieferung "
    "rechnen kann, sonst möchte ich vom Kauf zurücktreten und mein Geld zurück."
)


def row(i, state, group=None):
    """Rows differ in their question so only the state can make them duplicates."""
    return {"id": f"r{i}", "state": state, "tags": [f"group:{group or i}"], "questions": {"q": {"type": "noul", "instructions": f"Question {i}?"}}}


def test_exact_duplicate_across_groups_is_dropped():
    kept, dups = dedup([row(1, TEXT), row(2, "  " + TEXT.upper() + " ")])
    assert [r["id"] for r in kept] == ["r1"] and dups[0]["kind"] == "exact"


def test_same_group_may_share_a_state():
    kept, dups = dedup([row(1, TEXT, "g"), row(2, TEXT, "g")])
    assert len(kept) == 2 and not dups


def test_near_duplicate_is_caught_by_minhash():
    near = TEXT.replace("12. September", "13. September").replace("einer Woche", "acht Tagen")
    kept, dups = dedup([row(1, TEXT), row(2, near)])
    assert [r["id"] for r in kept] == ["r1"] and dups[0]["kind"] == "minhash"


def test_different_states_are_kept():
    other = "Merhaba, faturamda iki kez aynı abonelik ücreti görünüyor. Fazla çekilen tutarın iadesini rica ediyorum, teşekkürler."
    kept, dups = dedup([row(1, TEXT), row(2, other)])
    assert len(kept) == 2 and not dups


def test_json_states_compare_by_content_not_key_order():
    a = {"ticket": {"id": 7, "body": TEXT}, "tags": ["x"]}
    b = {"tags": ["x"], "ticket": {"body": TEXT, "id": 7}}
    kept, dups = dedup([row(1, a), row(2, b)])
    assert len(kept) == 1 and dups[0]["kind"] == "exact"


def test_split_blocks_keeps_good_rows_when_one_is_broken():
    reply = '### card 1\n{"state": "a", "questions": {}}\n### card 2\n{"state": "say "hi"", "questions": {}}\n### card 3\n```json\n{"state": "c", "questions": {}}\n```'
    blocks = split_blocks(reply)
    assert blocks[1]["state"] == "a" and blocks[3]["state"] == "c"
    assert isinstance(blocks[2], ValueError)  # an unescaped quote cannot be repaired


def test_split_blocks_repairs_a_missing_brace_and_unnests_writer_dist():
    reply = '### card 1\n{"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}, "writer_dist": {"q": {"yes": 0.9, "no": 0.1}}}'
    row = split_blocks(reply)[1]
    assert list(row["questions"]) == ["q"] and row["writer_dist"] == {"q": {"yes": 0.9, "no": 0.1}}


def test_split_blocks_never_returns_an_inner_object():
    reply = '### card 1\n{"state": [1, 2, "x"}, "questions": {"q": {"type": "noul"}}}'
    got = split_blocks(reply)[1]
    assert isinstance(got, ValueError) or "state" in got


def test_identical_rows_are_dropped_even_within_a_group():
    q = {"refund": {"type": "noul", "instructions": "Refund?"}}
    a = {**row(1, TEXT, "g"), "questions": q, "targets": {"refund": {"label": "yes"}}}
    b = {**row(2, TEXT, "g"), "questions": q, "targets": {"refund": {"label": "yes"}}}
    kept, dups = dedup([a, b])
    assert [r["id"] for r in kept] == ["r1"] and dups[0]["kind"] == "identical_row"
