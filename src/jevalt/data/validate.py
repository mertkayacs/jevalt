"""Row checks: schema, compilation, leaked answers, language ID and PII scrubbing."""

from __future__ import annotations

import re
import urllib.request
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from jevalt.format import DECISION_TOKEN, MAX_OPTIONS, UNKNOWN

from .common import compiled, data_dir, sha, text_of

FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
LABEL_RE = re.compile(r"^[^\s:<>`]{1,64}$")
NOUL_KEYS = {"yes", "no", "true", "false", "1", "0"}
MAX_CHOICE = MAX_OPTIONS - 2  # leave room for the abstain option
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
# fastText labels Swiss and Bavarian dialect text separately; both are German here.
LANG_OK = {"en": {"en"}, "tr": {"tr"}, "de": {"de", "als", "bar"}}


# ---------------------------------------------------------------- schema


def check_question(name: str, q: Any) -> list[str]:
    if not FIELD_RE.match(str(name)):
        return [f"bad field name {name!r}"]
    if not isinstance(q, Mapping):
        return [f"{name}: question is not an object"]
    kind = q.get("type")
    problems = []
    instructions = q.get("instructions")
    if instructions in (None, "", {}, []) or (isinstance(instructions, str) and not instructions.strip()):
        problems.append(f"{name}: empty instructions")
    crit = q.get("criteria")
    if kind == "choice":
        if not isinstance(crit, Mapping) or not 2 <= len(crit) <= MAX_CHOICE:
            return problems + [f"{name}: choice needs 2 to {MAX_CHOICE} options"]
        labels = [str(k) for k in crit]
        if any(not LABEL_RE.match(label) for label in labels):
            problems.append(f"{name}: bad option label")
        if len({label.casefold() for label in labels}) != len(labels):
            problems.append(f"{name}: duplicate option labels")
        if UNKNOWN in {label.casefold() for label in labels}:
            problems.append(f"{name}: option label 'unknown' is reserved")
    elif kind == "score":
        levels = list(crit) if isinstance(crit, (list, Mapping)) else []
        if not 2 <= len(levels) <= 10:
            problems.append(f"{name}: score needs 2 to 10 levels")
        elif isinstance(crit, list) and any(v in (None, "", {}, []) for v in crit):
            problems.append(f"{name}: empty score level")
    elif kind == "noul":
        if crit not in (None, {}):
            keys = {str(k).lower() for k in crit} if isinstance(crit, Mapping) else set()
            if not keys or not keys <= NOUL_KEYS or not (keys & {"yes", "true", "1"} and keys & {"no", "false", "0"}):
                problems.append(f"{name}: noul criteria must describe yes and no")
    else:
        problems.append(f"{name}: unknown type {kind!r}")
    return problems


def check_row(row: Mapping[str, Any], *, min_q: int = 1, max_q: int = 6) -> list[str]:
    problems = []
    state = row.get("state")
    if isinstance(state, str):
        if len(state.strip()) < 15:
            problems.append("state too short")
    elif not isinstance(state, (dict, list)) or not state:
        problems.append("state must be a non-empty string, object or array")
    questions = row.get("questions")
    if not isinstance(questions, Mapping) or not min_q <= len(questions) <= max_q:
        return problems + [f"need {min_q} to {max_q} questions"]
    for name, q in questions.items():
        problems += check_question(name, q)
    if DECISION_TOKEN in text_of(state) or DECISION_TOKEN in text_of(questions):
        problems.append("reserved <decision> marker in text")
    if not problems and (hit := denylisted(row)):
        problems.append(f"contains a denylisted name ({hit[:1]}...)")
    if not problems:
        from .prompts import copied  # rows must be teacher text, never our prompt wording

        grab = copied(" ".join(natural_text(row)))
        if grab:
            problems.append(f"copies prompt text: '{grab}'")
    if not problems:
        try:
            compiled(row)
        except (ValueError, KeyError, TypeError) as exc:
            problems.append(f"does not compile: {exc}")
    return problems


def check_targets(row: Mapping[str, Any]) -> list[str]:
    """Targets must cover exactly the compiled option values, sum to 1 and name the argmax."""
    from .common import field_values

    problems = []
    values = field_values(row)
    targets = row.get("targets") or {}
    if set(targets) != set(values):
        return [f"targets {sorted(targets)} do not match fields {sorted(values)}"]
    for name, t in targets.items():
        dist = t.get("dist") or {}
        if set(dist) != set(values[name]):
            problems.append(f"{name}: dist keys do not match options")
            continue
        if abs(sum(dist.values()) - 1) > 1e-3 or min(dist.values()) < 0:
            problems.append(f"{name}: dist does not sum to 1")
        if t.get("label") != max(dist, key=lambda k: dist[k]):
            problems.append(f"{name}: label is not the argmax")
    return problems


# ---------------------------------------------------------------- denylist


def _fold(text: str) -> str:
    import unicodedata

    text = text.replace("ı", "i").replace("İ", "i").replace("ß", "ss")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text)


@lru_cache(maxsize=1)
def _denylist() -> tuple[str, ...]:
    """Names that must never appear in rows (real people close to the project). The
    list lives in the data directory, one name per line, never in the repository."""
    path = data_dir() / "denylist.txt"
    if not path.exists():
        return ()
    return tuple(f" {_fold(line).strip()} " for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def denylisted(row: Mapping[str, Any]) -> str | None:
    names = _denylist()
    if not names:
        return None
    text = f" {_fold(text_of(row.get('state')) + ' ' + text_of(row.get('questions')))} "
    return next((n.strip() for n in names if n in text), None)


# ---------------------------------------------------------------- leaked answers

LEAK_PHRASES = re.compile(
    r"\b(correct|right) (answer|option|choice)\b|\bthe answer is\b|\banswer:\s"
    r"|doğru (cevap|yanıt|seçenek)|\bcevap:\s|\byanıt:\s"
    r"|\brichtige (antwort|option)\b|\bdie antwort (ist|lautet)\b|\bantwort:\s",
    re.IGNORECASE,
)


def leaks(row: Mapping[str, Any], writer_dist: Mapping[str, Mapping[str, float]] | None = None) -> list[str]:
    """Instructions that give the answer away.

    Flags answer-announcing phrases, and choice instructions that name only the
    option the writer itself considered correct.
    """
    problems = []
    for name, q in row["questions"].items():
        text = text_of(q.get("instructions", ""))
        if LEAK_PHRASES.search(text):
            problems.append(f"{name}: instructions announce an answer")
        if q.get("type") == "choice" and writer_dist and name in writer_dist:
            labels = [str(k) for k in q["criteria"]]
            named = [label for label in labels if len(label) >= 4 and re.search(rf"(?<![\w]){re.escape(label)}(?![\w])", text)]
            dist = writer_dist[name]
            if len(labels) >= 3 and len(named) == 1 and dist and named[0] == max(dist, key=lambda k: dist.get(k, 0)):
                problems.append(f"{name}: instructions name only the intended option")
    return problems


# ---------------------------------------------------------------- language ID

IDENTIFIER = re.compile(r"^[\w.\-/@:#+]+$")


def natural_text(row: Mapping[str, Any], *, include_questions: bool = True) -> list[str]:
    """Natural-language strings of a row: string values of the state, instructions and
    option descriptions. Keys, labels and identifier-like values are skipped."""
    parts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            letters = sum(ch.isalpha() for ch in value)
            if letters >= 3 and not (IDENTIFIER.match(value) and ("_" in value or any(c.isdigit() for c in value))):
                parts.append(value)
        elif isinstance(value, Mapping):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)

    walk(row["state"])
    if include_questions:
        for q in row["questions"].values():
            if not isinstance(q, Mapping):
                continue
            walk(q.get("instructions"))
            crit = q.get("criteria")
            if isinstance(crit, Mapping):
                walk(list(crit.values()))
            elif isinstance(crit, list):
                walk(crit)
    return parts


def lid_model_path():
    path = data_dir() / "models" / "lid.176.ftz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LID_URL, path)
    return path


@lru_cache(maxsize=1)
def _lid():
    import fasttext

    return fasttext.load_model(str(lid_model_path()))


def detect(text: str) -> tuple[str, float]:
    labels, probs = _lid().predict(" ".join(text.split()), k=1)
    return labels[0].removeprefix("__label__"), float(probs[0])


def language_check(row: Mapping[str, Any], *, include_questions: bool = True, min_prob: float = 0.5, min_share: float = 0.6) -> dict:
    """Overall language of the natural text plus the share of characters in text
    segments (20+ characters) detected as the row language."""
    lang = row["lang"]
    parts = natural_text(row, include_questions=include_questions)
    text = " ".join(parts)
    if sum(ch.isalpha() for ch in text) < 20:
        return {"ok": True, "lang": lang, "prob": 1.0, "share": 1.0, "note": "too little text to check"}
    top, prob = detect(text)
    long_parts = [p for p in parts if len(p) >= 20] or [text]
    good = sum(len(p) for p in long_parts if detect(p)[0] in LANG_OK[lang])
    share = good / sum(len(p) for p in long_parts)
    fold = folded(text, lang)
    ok = top in LANG_OK[lang] and prob >= min_prob and share >= min_share and not fold
    return {"ok": ok, "lang": top, "prob": round(prob, 3), "share": round(share, 3), **({"folded": True} if fold else {})}


SPECIAL = {"tr": set("çğıöşüÇĞİÖŞÜ"), "de": set("äöüßÄÖÜ")}


def folded(text: str, lang: str) -> bool:
    """Turkish or German written without its own letters (ASCII-folded).

    fastText still calls such text Turkish or German, so it needs its own check:
    Turkish prose has several percent of ç ğ ı ö ş ü, German prose has umlauts.
    """
    if lang not in SPECIAL:
        return False
    letters = sum(ch.isalpha() for ch in text)
    special = sum(ch in SPECIAL[lang] for ch in text)
    if lang == "tr":
        return letters >= 150 and special / letters < 0.02
    return letters >= 400 and special == 0


# ---------------------------------------------------------------- PII

EMAIL = re.compile(r"(?<![\w.+-])([\w.+-]+)@((?:[\w-]+\.)+[A-Za-z]{2,})\b")
SAFE_DOMAINS = ("example.com", "example.org", "example.net")
SAFE_TLDS = (".example", ".test", ".invalid", ".localhost")
IBAN = re.compile(r"\b([A-Z]{2})(\d{2})((?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?)\b")
# Phone-shaped only: international (+..), national trunk prefix 0 (TR, DE, UK) or
# NANP with separators. Bare digit runs (order numbers, IDs) are left alone.
PHONE = re.compile(
    r"(?<![\w+.,/])(?:"
    r"\+\d{1,3}[\s.\-]?\(?\d{1,4}\)?(?:[\s.\-]?\d){5,10}"
    r"|\(?0\d{2,4}\)?[\s.\-/]\d(?:[\s.\-]?\d){4,8}"
    r"|05\d{9}|01[5-7]\d{7,9}"  # unseparated TR and DE mobile numbers
    r"|\(?[2-9]\d{2}\)?[\s.\-]\d{3}[\s.\-]\d{4}"
    r")(?!\w|[.,/\-]\d)"
)
CARD = re.compile(r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)")
TCKN = re.compile(r"(?i)(t\.?\s?c\.?|kimlik)[^0-9\n]{0,25}(\d{11})\b")


def _luhn(number: str) -> bool:
    digits = [int(d) for d in number][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return total % 10 == 0


def _redigit(text: str, digits: str) -> str:
    """Write new digits into the original formatting (spaces, dashes, brackets)."""
    it = iter(digits)
    return "".join(next(it) if ch.isdigit() else ch for ch in text)


def _fake_phone(match: str, lang: str) -> str:
    digits = "".join(ch for ch in match if ch.isdigit())
    if len(digits) < 9:
        return match
    tail = digits[-2:]
    if lang == "en":
        # NANP reserves 555-0100 to 555-0199 for fiction.
        if digits[-7:-2] == "55501":
            return match
        new = digits[:-7] + "55501" + tail
    else:
        if set(digits[-7:-2]) == {"0"}:
            return match
        new = digits[:-7] + "00000" + tail
    return _redigit(match, new)


def scrub_text(text: str, lang: str) -> tuple[str, int]:
    """Replace contact data by obviously fake values. Distinct inputs stay distinct in
    their last digits, so facts like "a different IBAN than on file" survive."""
    count = 0

    def email(m: re.Match) -> str:
        nonlocal count
        domain = m.group(2).lower()
        if domain in SAFE_DOMAINS or domain.endswith(SAFE_TLDS) or any(domain.endswith("." + d) for d in SAFE_DOMAINS):
            return m.group(0)
        count += 1
        return f"{m.group(1)}@{SAFE_DOMAINS[int(sha(domain)[:2], 16) % 3]}"

    def iban(m: re.Match) -> str:
        nonlocal count
        if m.group(2) == "00":
            return m.group(0)
        count += 1
        return m.group(1) + "00" + m.group(3)  # check digits 00 are never valid

    def phone(m: re.Match) -> str:
        nonlocal count
        new = _fake_phone(m.group(0), lang)
        count += new != m.group(0)
        return new

    def card(m: re.Match) -> str:
        nonlocal count
        digits = "".join(ch for ch in m.group(0) if ch.isdigit())
        if not 13 <= len(digits) <= 19 or not _luhn(digits):
            return m.group(0)
        count += 1
        return _redigit(m.group(0), digits[:4] + "0" * (len(digits) - 6) + digits[-2:])

    def tckn(m: re.Match) -> str:
        nonlocal count
        number = m.group(2)
        if number[1:9] == "0" * 8:
            return m.group(0)
        count += 1
        return m.group(0)[: m.start(2) - m.start(0)] + number[0] + "0" * 8 + number[-2:]

    # Mask each replaced value so a later pattern cannot rewrite digits inside it
    # (an IBAN tail looks like a phone number).
    masked: list[str] = []

    def mask(fn):
        def inner(m: re.Match) -> str:
            masked.append(fn(m))
            return f"\x00{len(masked) - 1}\x00"

        return inner

    text = EMAIL.sub(mask(email), text)
    text = IBAN.sub(mask(iban), text)
    text = TCKN.sub(mask(tckn), text)
    text = CARD.sub(mask(card), text)
    text = PHONE.sub(phone, text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: masked[int(m.group(1))], text)
    return text, count


def scrub(value: Any, lang: str) -> tuple[Any, int]:
    """Scrub every string inside a state or question structure."""
    if isinstance(value, str):
        return scrub_text(value, lang)
    if isinstance(value, Mapping):
        out, total = {}, 0
        for k, v in value.items():
            out[k], n = scrub(v, lang)
            total += n
        return out, total
    if isinstance(value, list):
        out_list, total = [], 0
        for v in value:
            new, n = scrub(v, lang)
            out_list.append(new)
            total += n
        return out_list, total
    return value, 0


def pii_hits(value: Any, lang: str) -> int:
    """How many replacements a scrub would make (0 means clean)."""
    return scrub(value, lang)[1]
