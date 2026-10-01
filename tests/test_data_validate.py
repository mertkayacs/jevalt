"""Row validation: schema, leaks, language checks and PII scrubbing."""

import pytest

from jevalt.data.validate import check_row, check_targets, folded, language_check, leaks, scrub, scrub_text


def row(**questions):
    return {"lang": "en", "state": "The customer says the parcel never arrived and wants a refund.", "questions": questions}


CHOICE = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "shipping": "Deliveries", "tech": None}}
NOUL = {"type": "noul", "instructions": "Refund requested?", "criteria": {"yes": "Asks for money back", "no": "Does not"}}
SCORE = {"type": "score", "instructions": "How upset?", "criteria": ["Calm", "Annoyed", "Angry"]}


def test_valid_row_passes():
    assert check_row(row(team=CHOICE, refund=NOUL, mood=SCORE), min_q=2) == []


@pytest.mark.parametrize(
    "question, problem",
    [
        ({**CHOICE, "criteria": {"billing": "a", "Billing": "b"}}, "duplicate option labels"),
        ({**CHOICE, "criteria": {"billing": "a", "unknown": "b"}}, "reserved"),
        ({**CHOICE, "criteria": {"billing": "a"}}, "choice needs"),
        ({**CHOICE, "criteria": {"has space": "a", "b": "b"}}, "bad option label"),
        ({**SCORE, "criteria": [str(i) for i in range(11)]}, "score needs 2 to 10"),
        ({**NOUL, "criteria": {"yes": "only yes"}}, "noul criteria"),
        ({**NOUL, "instructions": "  "}, "empty instructions"),
        ({"type": "rank", "instructions": "x"}, "unknown type"),
    ],
)
def test_bad_questions_are_caught(question, problem):
    problems = check_row(row(q=question))
    assert any(problem in p for p in problems), problems


def test_bad_field_name_and_short_state():
    assert any("bad field name" in p for p in check_row({"lang": "en", "state": "long enough state text", "questions": {"1x": NOUL}}))
    assert "state too short" in check_row({"lang": "en", "state": "hi", "questions": {"q": NOUL}})


def test_question_count_limits():
    assert check_row(row(a=NOUL), min_q=2)
    many = {f"q{i}": NOUL for i in range(7)}
    assert check_row(row(**many))


def test_targets_must_match_compiled_options():
    good = {**row(refund=NOUL), "targets": {"refund": {"label": "yes", "dist": {"no": 0.1, "yes": 0.9}}}}
    assert check_targets(good) == []
    wrong_keys = {**good, "targets": {"refund": {"label": "yes", "dist": {"false": 0.1, "yes": 0.9}}}}
    assert check_targets(wrong_keys)
    wrong_label = {**good, "targets": {"refund": {"label": "no", "dist": {"no": 0.1, "yes": 0.9}}}}
    assert any("argmax" in p for p in check_targets(wrong_label))
    bad_sum = {**good, "targets": {"refund": {"label": "yes", "dist": {"no": 0.2, "yes": 0.9}}}}
    assert any("sum" in p for p in check_targets(bad_sum))


def test_abstain_rows_need_unknown_in_targets():
    r = {**row(team=CHOICE), "tags": ["abstain"], "targets": {"team": {"label": "unknown", "dist": {"billing": 0.005, "shipping": 0.005, "tech": 0.005, "unknown": 0.985}}}}
    assert check_targets(r) == []
    r["tags"] = []
    assert check_targets(r)


def test_leak_phrases_and_named_intended_option():
    leaked = row(q={**CHOICE, "instructions": "The correct answer is shipping. Which team?"})
    assert leaks(leaked)
    named = row(q={**CHOICE, "instructions": "Should this go to shipping?"})
    assert leaks(named, {"q": {"billing": 0.1, "shipping": 0.8, "tech": 0.1}})
    assert not leaks(named, {"q": {"billing": 0.8, "shipping": 0.1, "tech": 0.1}})
    all_named = row(q={**CHOICE, "instructions": "billing, shipping or tech?"})
    assert not leaks(all_named, {"q": {"billing": 0.1, "shipping": 0.8, "tech": 0.1}})
    assert leaks(row(q={**NOUL, "instructions": "Doğru cevap: evet. İade istiyor mu?"}))


def test_folded_turkish_and_german():
    ascii_tr = "Musteri siparisini iki gun once verdi ama kargo hala gelmedi, bu yuzden cok sinirli ve paranin iade edilmesini istiyor. " * 2
    real_tr = "Müşteri siparişini iki gün önce verdi ama kargo hâlâ gelmedi, bu yüzden çok sinirli ve paranın iade edilmesini istiyor. " * 2
    assert folded(ascii_tr, "tr") and not folded(real_tr, "tr")
    ascii_de = "Der Kunde moechte eine Rueckerstattung, weil das Paket nie angekommen ist und er sich ueber den Service aergert. " * 6
    real_de = "Der Kunde möchte eine Rückerstattung, weil das Paket nie angekommen ist und er sich über den Service ärgert. " * 6
    assert folded(ascii_de, "de") and not folded(real_de, "de")
    assert not folded(ascii_de, "en")


def test_language_check_flags_wrong_language():
    tr_row = {"lang": "tr", "state": "Müşteri siparişinin hâlâ gelmediğini söylüyor ve paranın iadesini istiyor.",
              "questions": {"iade": {"type": "noul", "instructions": "Müşteri iade istiyor mu?"}}}
    assert language_check(tr_row)["ok"]
    en_in_tr = {**tr_row, "state": "The customer says the parcel never arrived and asks for the money back right now."}
    assert not language_check(en_in_tr)["ok"]


def test_scrub_phones_emails_ibans():
    text, n = scrub_text("Call 0532 123 45 67 or mail ayse@firma.com.tr, IBAN TR33 0006 1005 1978 6457 8413 26", "tr")
    assert n == 3
    assert "0532 000 00 67" in text and "ayse@example." in text and "TR00 0006" in text
    us, _ = scrub_text("Reach me at (212) 867-5309.", "en")
    assert "(212) 555-0109" in us


def test_scrub_keeps_facts_and_distinctness():
    text = "Invoice 12345678901 dated 30.09.2026 for 1.234,50 EUR; old IBAN DE89 3704 0044 0532 0130 00, new IBAN DE12 3704 0044 0532 0130 99"
    out, _ = scrub_text(text, "de")
    assert "12345678901" in out and "30.09.2026" in out and "1.234,50" in out
    ibans = [w for w in out.split(", ") if "IBAN" in w]
    assert ibans[0] != ibans[1]  # a changed-IBAN fraud case stays a changed IBAN


def test_scrub_is_idempotent_and_recursive():
    state = {"contact": {"phone": "+49 30 12345678", "mail": "max@firma.de"}, "notes": ["Rückruf unter 0151 23456789"]}
    once, n1 = scrub(state, "de")
    twice, n2 = scrub(once, "de")
    assert n1 == 3 and n2 == 0 and once == twice


def test_identifier_folding_follows_the_language():
    from jevalt.data.generate import ascii_ident

    assert ascii_ident("gebühr_fällig", "de") == "gebuehr_faellig"
    assert ascii_ident("kanıt_gücü", "tr") == "kanit_gucu"
    assert ascii_ident("1st_check", "en") == "q_1st_check"


def test_salvage_drops_only_malformed_questions():
    from jevalt.data.generate import validate_rows

    good_q = {"type": "noul", "instructions": "Ist die Gebühr fällig?"}
    score_q = {"type": "score", "instructions": "Wie dringend ist der Fall?", "criteria": ["gering", "mittel", "hoch"]}
    bad_q = {"type": "choice", "instructions": "Was tun?", "criteria": ["zahlen", "warten"]}
    state = "Die Rechnung vom März ist seit drei Wochen offen und die Mahnung ging gestern raus."
    row = {"id": "r1", "lang": "de", "state": state, "tags": [], "writer": {"dist": {}},
           "questions": {"gebühr_fällig": good_q, "dringlichkeit": score_q, "naechster_schritt": bad_q}}
    kept, drops = validate_rows([row])
    assert len(kept) == 1 and list(kept[0]["questions"]) == ["gebuehr_faellig", "dringlichkeit"]
    mostly_bad = {**row, "id": "r2", "questions": {"a": bad_q, "b": bad_q, "c": good_q}}
    kept, drops = validate_rows([mostly_bad])
    assert not kept and drops[0]["reason"] == "schema"
