"""Programmatic fix sets with exact gold: dates (F5), numbers (F6), multi-hop
lookups (F9), inverted criteria (F10) and long policies (F12).

Code owns the facts and the decision rule. A teacher writes the surface text once
per rule and language: a state template with {placeholders}, the questions, the
option labels and descriptions. Code fills the placeholders with sampled values
(names from a teacher-written pool; dates, weekdays and amounts formatted from
CLDR locale data via Babel) and computes the gold label.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from babel.dates import format_date, get_day_names
from babel.numbers import format_currency, format_decimal

from .common import LANG_NAMES, text_of

# ---------------------------------------------------------------- locale formatting

DATE_PATTERNS = {
    "en_US": ["MM/dd/yyyy", "MMMM d, y", "MMM d, y", "yyyy-MM-dd", "EEEE, MMMM d, y"],
    "en_GB": ["dd/MM/yyyy", "d MMMM y", "d MMM y", "yyyy-MM-dd", "EEEE d MMMM y"],
    "tr_TR": ["dd.MM.yyyy", "d MMMM y", "d MMMM y EEEE", "dd/MM/yyyy", "yyyy-MM-dd"],
    "de_DE": ["dd.MM.yyyy", "d. MMMM y", "EEEE, d. MMMM y", "d.M.yyyy", "yyyy-MM-dd"],
}
DATE_PATTERNS["de_AT"] = DATE_PATTERNS["de_CH"] = DATE_PATTERNS["de_DE"]
CURRENCY = {"en_US": "USD", "en_GB": "GBP", "tr_TR": "TRY", "de_DE": "EUR", "de_AT": "EUR", "de_CH": "CHF"}
MONEY_SCALE = {"TRY": 35}  # rough price level, so Turkish amounts look plausible


class Fmt:
    """Formats values for one row in one locale. Numeric slash dates are only used
    when the day is above 12, so a date never reads two ways."""

    def __init__(self, lang: str, r: random.Random):
        self.lang, self.r = lang, r
        self.locale = {"en": r.choice(["en_US", "en_US", "en_US", "en_GB"]), "tr": "tr_TR", "de": r.choice(["de_DE"] * 8 + ["de_AT", "de_CH"])}[lang]
        self.currency = CURRENCY[self.locale]
        self.scale = MONEY_SCALE.get(self.currency, 1)
        self.mixed = r.random() < 0.3
        self.pattern = r.choice(DATE_PATTERNS[self.locale])
        self.tl_suffix = lang == "tr" and r.random() < 0.5

    def date(self, d: date) -> str:
        pattern = self.r.choice(DATE_PATTERNS[self.locale]) if self.mixed else self.pattern
        if "/" in pattern and d.day <= 12:
            pattern = DATE_PATTERNS[self.locale][1]
        return format_date(d, pattern, locale=self.locale)

    def money(self, x: Decimal) -> str:
        if self.tl_suffix:
            return format_decimal(x, format="#,##0.00", locale=self.locale) + " TL"
        return format_currency(x, self.currency, locale=self.locale)

    def num(self, x: Any) -> str:
        return format_decimal(x, locale=self.locale)

    def amount(self, lo: float, hi: float, *, scaled: bool = True) -> Decimal:
        """Random amount with cents. ``lo``/``hi`` are in dollar-like units unless
        ``scaled`` is False (teacher-given prices already in the row currency)."""
        k = self.scale if scaled else 1
        return Decimal(self.r.randint(int(lo * k * 100), int(hi * k * 100))) / 100

    def round_money(self, x: float) -> Decimal:
        step = 5 if self.scale == 1 else 50
        return Decimal(int(round(x * self.scale / step)) * step)

    def weekdays(self) -> list[tuple[str, str]]:
        """(label, name) for Monday..Sunday from CLDR; labels are ASCII-folded names."""
        names = get_day_names("wide", locale=self.locale)
        return [(ascii_label(names[i]), names[i]) for i in range(7)]


def ascii_label(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.replace("ı", "i").replace("ß", "ss")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", folded.lower()).strip("_")


# ---------------------------------------------------------------- date helpers


def rand_date(r: random.Random, lo: date = date(2024, 1, 1), hi: date = date(2027, 12, 31)) -> date:
    return lo + timedelta(days=r.randint(0, (hi - lo).days))


def add_business_days(start: date, n: int) -> date:
    """End of the n-th Monday-to-Friday day after ``start`` (start itself not counted)."""
    day = start
    while n > 0:
        day += timedelta(days=1)
        if day.weekday() < 5:
            n -= 1
    return day


def working_days(first: date, last: date) -> int:
    """Monday-to-Friday days from ``first`` to ``last``, both included."""
    return sum((first + timedelta(days=i)).weekday() < 5 for i in range((last - first).days + 1))


def age_on(birth: date, day: date) -> int:
    return day.year - birth.year - ((day.month, day.day) < (birth.month, birth.day))


def near(r: random.Random, base: int, *, span: int = 20) -> int:
    """An offset around a boundary: exactly on it, one off, or further away."""
    return r.choice([0, 0, 1, -1, r.randint(2, span), -r.randint(2, span)])


def side(r: random.Random, *, span: int = 20) -> int:
    """Offset from a boundary, at or below it (0 counts there) exactly half the time.
    Boundary and off-by-one values make up most draws, since that is where rules bite."""
    if r.random() < 0.5:
        return r.choice([0, 0, -1, -r.randint(2, span)])
    return r.choice([1, 1, r.randint(2, span)])


def money_side(r: random.Random, fmt: "Fmt", lo: float, hi: float) -> Decimal:
    """Money offset at or below zero half the time (zero included)."""
    if r.random() < 0.5:
        return r.choice([Decimal(0), -fmt.amount(lo, hi)])
    return fmt.amount(lo, hi)


# ---------------------------------------------------------------- rule specs


@dataclass
class Q:
    key: str
    kind: str  # noul | choice
    meaning: str
    outcomes: dict[str, str] = field(default_factory=dict)  # choice: internal key -> meaning
    options_from: str = ""  # choice options built by code: "weekday", "items", "counts", "depots"


@dataclass
class Rule:
    name: str
    fset: str
    setting: str
    slots: dict[str, str]
    rule: str
    questions: list[Q]
    sample: Callable[[random.Random, Fmt, Mapping], dict]
    gold: Callable[[dict], dict[str, str]]
    vocab: dict[str, tuple[str, int, str]] = field(default_factory=dict)  # key -> (kind, minimum count, meaning)
    extra: str = ""
    lists: dict[str, list[str]] = field(default_factory=dict)  # list slot -> placeholders of one line
    money: tuple[str, ...] = ()  # filled with an amount that already carries its currency
    units: tuple[str, ...] = ()  # filled with a number that already carries its unit
    q_slots: tuple[str, ...] = ()  # the only placeholders allowed in questions
    min_words: int = 0  # minimum length of the state template (long policies)


FORMATS = "email, chat message, JSON record, formal letter or notice, form printout, plain note"


def _pick_name(pool: Mapping, key: str, r: random.Random) -> str:
    return r.choice(pool.get(key) or ["-"])


# F5 dates ---------------------------------------------------------------


def _deadline(r, fmt, pool):
    deadline = rand_date(r)
    delta = side(r, span=40)
    submitted = deadline + timedelta(days=delta)
    return {"delta": delta, "slots": {"person": _pick_name(pool, "people", r), "org": _pick_name(pool, "orgs", r), "submitted": fmt.date(submitted), "deadline": fmt.date(deadline)}}


def _deadline_gold(f):
    d = f["delta"]
    return {"on_time": "yes" if d <= 0 else "no", "lateness": "on_time" if d <= 0 else "late_1_7" if d <= 7 else "late_8_plus"}


def _window(r, fmt, pool):
    window = r.choice([7, 10, 14, 15, 30])
    delivered = rand_date(r)
    delta = max(0, window + side(r, span=max(3, window - 1)))
    last = delivered + timedelta(days=window)
    return {"delta": delta, "window": window, "weekday": last.weekday(),
            "slots": {"person": _pick_name(pool, "people", r), "org": _pick_name(pool, "orgs", r), "delivered": fmt.date(delivered),
                      "requested": fmt.date(delivered + timedelta(days=delta)), "window_days": fmt.num(window)}}


def _business(r, fmt, pool):
    ordered = rand_date(r)
    while ordered.weekday() > 4:
        ordered += timedelta(days=1)
    n = r.randint(1, 15)
    ready = add_business_days(ordered, n)
    promised = ready - timedelta(days=side(r, span=6))
    return {"ready": ready, "promised": promised,
            "slots": {"org": _pick_name(pool, "orgs", r), "person": _pick_name(pool, "people", r), "ordered": fmt.date(ordered),
                      "n_days": fmt.num(n), "promised": fmt.date(promised)}}


def _relative(r, fmt, pool):
    today = rand_date(r, hi=date(2027, 12, 20))
    month_end = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    left = (month_end - today).days
    if r.random() < 0.5 and left >= 2:  # same month, half of the time when possible
        n = r.randint(2, left)
    else:
        n = r.randint(max(2, left + 1), left + 45)
    target = today + timedelta(days=n)
    return {"weekday": target.weekday(), "same_month": (target.year, target.month) == (today.year, today.month),
            "slots": {"person": _pick_name(pool, "people", r), "today": fmt.date(today), "n_days": fmt.num(n)}}


def _age(r, fmt, pool):
    event = rand_date(r)
    event = event.replace(day=min(event.day, 28))  # the birthday exists in every year
    years = r.choice([16, 18, 18, 21, 25, 65])
    birthday = date(event.year - years, event.month, event.day)
    birth = birthday + timedelta(days=side(r, span=300))
    return {"old_enough": age_on(birth, event) >= years,
            "slots": {"person": _pick_name(pool, "people", r), "birth": fmt.date(birth), "event_date": fmt.date(event), "min_age": fmt.num(years)}}


def _order(r, fmt, pool):
    a = rand_date(r)
    b = a + timedelta(days=r.choice([1, -1, r.randint(2, 90), -r.randint(2, 90)]))
    mixed = Fmt(fmt.lang, r)
    return {"a_first": a < b, "slots": {"org": _pick_name(pool, "orgs", r), "date_a": fmt.date(a), "date_b": mixed.date(b)}}


# F6 numbers -------------------------------------------------------------


def _items(r, fmt, pool, k):
    vocab = r.sample(pool["items"], min(k, len(pool["items"])))
    items, used = [], set()
    for v in vocab:
        amount = fmt.amount(float(v.get("price_min", 5)), float(v.get("price_max", 200)), scaled=False)
        while amount in used:
            amount += Decimal("0.10")
        used.add(amount)
        items.append({"label": v["label"], "name": v["name"], "amount": amount})
    return items


def _sum_limit(r, fmt, pool):
    items = _items(r, fmt, pool, r.randint(3, 6))
    total = sum(i["amount"] for i in items)
    gap = min(fmt.amount(0.5, 40), total / 4)
    if r.random() < 0.5:  # over the limit
        limit = total - gap
    else:  # at or under it; equal totals do not exceed
        limit = total if r.random() < 0.4 else total + gap
    largest = max(items, key=lambda i: i["amount"])
    return {"over": total > limit, "largest": largest["label"], "items": items,
            "slots": {"person": _pick_name(pool, "people", r), "org": _pick_name(pool, "orgs", r), "limit": fmt.money(limit)},
            "lines": {"items": [{"name": i["name"], "amount": fmt.money(i["amount"])} for i in items]},
            "raw": {"limit": float(limit), "items": [{"name": i["name"], "amount": float(i["amount"])} for i in items]}}


def _discount(r, fmt, pool):
    product = r.choice(pool["items"])
    price = Decimal(max(1, int(fmt.amount(float(product.get("price_min", 20)), float(product.get("price_max", 500)), scaled=False))))
    pct = r.choice([5, 10, 15, 20, 25, 30, 40, 50])
    final = (price * (100 - pct) / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    step = r.choice([Decimal("0.01"), fmt.amount(1, 30)])
    budget = final + (r.choice([Decimal(0), step]) if r.random() < 0.5 else -step)
    return {"within": final <= budget, "slots": {"product": product["name"], "price": fmt.money(price), "discount": fmt.num(pct), "budget": fmt.money(budget)},
            "raw": {"price": float(price), "discount": pct, "budget": float(budget)}}


def _count(r, fmt, pool):
    statuses = pool["statuses"]
    n = r.randint(4, 10)
    threshold = r.randint(10, 90)
    target = r.choice(statuses)
    records = []
    for i in range(n):
        status = r.choice(statuses)
        value = threshold + r.choice([0, 1, -1, r.randint(2, 40), -r.randint(2, 9)])
        records.append({"id": f"{pool.get('id_prefix', 'R-')}{r.randint(1000, 9999)}", "status": status["name"], "label": status["label"], "value": value})
    count = sum(rec["label"] == target["label"] and rec["value"] >= threshold for rec in records)
    return {"count": str(count), "n": n,
            "slots": {"status": target["name"], "threshold": fmt.num(threshold)},
            "lines": {"records": [{"id": rec["id"], "status": rec["status"], "value": fmt.num(rec["value"])} for rec in records]},
            "raw": {"threshold": threshold, "records": [{"id": rec["id"], "status": rec["status"], "value": rec["value"]} for rec in records]}}


UNITS = [("g", "kg", 1000), ("ml", "l", 1000), ("cm", "m", 100), ("mm", "cm", 10)]


def _units(r, fmt, pool):
    small, big, factor = r.choice(UNITS)
    limit = Decimal(r.choice([1, 2, 5, 10, 15, 20, 25, 30])) / r.choice([1, 2, 4])
    exact = limit * factor
    amount = exact + side(r, span=int(exact / 3) + 2)
    amount = max(Decimal(1), Decimal(amount))
    return {"within": amount <= exact, "slots": {"item": _pick_name(pool, "things", r), "amount": f"{fmt.num(amount)} {small}", "limit": f"{fmt.num(limit)} {big}"},
            "raw": {"amount": float(amount), "limit": float(limit)}}


def _average(r, fmt, pool):
    k = r.randint(3, 6)
    pass_mark = r.choice([50, 60, 65, 70, 75])
    target = pass_mark - side(r, span=15)
    scores = [max(0, min(100, target + r.randint(-20, 20))) for _ in range(k - 1)]
    last = target * k - sum(scores)
    if not 0 <= last <= 100:
        scores = [target] * (k - 1)
        last = target
    scores.append(last)
    r.shuffle(scores)
    return {"passes": sum(scores) / k >= pass_mark,
            "slots": {"person": _pick_name(pool, "people", r), "scores": ", ".join(fmt.num(s) for s in scores), "pass_mark": fmt.num(pass_mark)},
            "raw": {"scores": scores, "pass_mark": pass_mark}}


# F12 policies -----------------------------------------------------------


def _expense(r, fmt, pool):
    hotel_cap = fmt.round_money(r.randint(80, 180))
    expensive_cap = hotel_cap + fmt.round_money(r.randint(30, 90))
    meal_cap = fmt.round_money(r.randint(25, 60))
    receipt_min = fmt.round_money(r.choice([25, 50, 75, 100]))
    expensive = r.random() < 0.5
    city = r.choice(pool["expensive_cities"] if expensive else pool["other_cities"])
    nights = r.randint(1, 6)
    cap = (expensive_cap if expensive else hotel_cap) * nights
    hotel_total = cap + money_side(r, fmt, 1, 60)
    days = nights + 1
    meal_total = meal_cap * days + money_side(r, fmt, 1, 30)
    big = r.choice(pool["items"])
    if r.random() < 0.5:  # the receipt rule holds: small enough, or receipt attached
        small = r.random() < 0.5
        big_amount = receipt_min - (fmt.amount(1, 20) if r.random() < 0.7 else Decimal(0)) if small else receipt_min + fmt.amount(1, 80)
        attached = (not small) or r.random() < 0.5
    else:  # above the threshold without a receipt
        big_amount = receipt_min + fmt.amount(1, 80)
        attached = False
    return {"hotel_ok": hotel_total <= cap, "meals_ok": meal_total <= meal_cap * days, "receipt_ok": big_amount <= receipt_min or attached,
            "slots": {"person": _pick_name(pool, "people", r), "city": city, "nights": fmt.num(nights), "days": fmt.num(days),
                      "hotel_total": fmt.money(hotel_total), "hotel_cap": fmt.money(hotel_cap), "expensive_cap": fmt.money(expensive_cap),
                      "meal_cap": fmt.money(meal_cap), "meal_total": fmt.money(meal_total), "receipt_min": fmt.money(receipt_min),
                      "big_item": big["name"], "big_amount": fmt.money(big_amount),
                      "receipt_status": pool["receipt_attached"] if attached else pool["receipt_missing"]}}


def _returns(r, fmt, pool):
    std_days = r.choice([14, 15, 30])
    el_days = r.choice([d for d in (7, 10, 14) if d != std_days])
    category = r.choice(["electronics", "hygiene", "other"])
    product = r.choice(pool["products"][category])
    opened, on_sale = r.random() < 0.5, r.random() < 0.4
    window = el_days if category == "electronics" else std_days
    days = max(0, window + side(r, span=10))
    if category == "hygiene" and opened:
        outcome = "not_returnable"
    elif days > window:
        outcome = "not_returnable"
    else:
        outcome = "store_credit" if on_sale else "full_refund"
    return {"outcome": outcome, "within": days <= window,
            "slots": {"person": _pick_name(pool, "people", r), "org": _pick_name(pool, "orgs", r), "product": product, "days_since": fmt.num(days),
                      "std_days": fmt.num(std_days), "el_days": fmt.num(el_days),
                      "opened_status": pool["opened"] if opened else pool["unopened"],
                      "sale_status": pool["on_sale"] if on_sale else pool["full_price"]}}


def _insurance(r, fmt, pool):
    deductible = fmt.round_money(r.choice([100, 150, 250, 500]))
    el_sublimit = fmt.round_money(r.choice([1000, 1500, 2000, 2500]))
    max_payout = fmt.round_money(r.choice([4000, 5000, 7500, 10000]))
    if r.random() < 0.5:  # above the sublimit, so it is capped
        el_value = el_sublimit + fmt.amount(50, 1500)
    else:  # at or below it; equal is not above
        el_value = el_sublimit - r.choice([Decimal(0), fmt.amount(50, 900)])
    jw_value = fmt.amount(200, 2500)
    other = fmt.amount(0, 3000)
    declared = r.random() < 0.5
    payout = min(max_payout, max(Decimal(0), min(el_value, el_sublimit) + (jw_value if declared else 0) + other - deductible))
    threshold = fmt.round_money(float(payout / fmt.scale) + r.choice([40, 300]) * (1 if r.random() < 0.5 else -1))
    return {"capped": el_value > el_sublimit, "jewellery": declared, "over": payout > threshold,
            "slots": {"person": _pick_name(pool, "people", r), "org": _pick_name(pool, "orgs", r), "deductible": fmt.money(deductible),
                      "el_sublimit": fmt.money(el_sublimit), "max_payout": fmt.money(max_payout), "el_value": fmt.money(el_value),
                      "jw_value": fmt.money(jw_value), "other_value": fmt.money(other), "threshold": fmt.money(threshold),
                      "jw_status": pool["declared"] if declared else pool["not_declared"]}}


def _leave(r, fmt, pool):
    annual = r.choice([20, 24, 26, 30])
    carry_max = r.choice([5, 10])
    unused = r.randint(0, 15)
    taken = r.randint(0, annual)
    available = annual + min(unused, carry_max) - taken
    start = rand_date(r, date(2025, 1, 6), date(2027, 11, 30))
    while start.weekday() > 4:
        start += timedelta(days=1)
    want = max(1, available + side(r, span=8))
    end = start
    while working_days(start, end) < want:
        end += timedelta(days=1)
    wd = working_days(start, end)
    # Director approval and blackout overlap each hold half the time, boundaries included.
    director = max(1, wd - r.choice([1, 1, r.randint(2, 5)])) if r.random() < 0.5 else wd + r.choice([0, 0, r.randint(1, 5)])
    length = r.randint(3, 14)
    if r.random() < 0.5:  # overlaps: the blackout starts inside the request or ends inside it
        b_start = r.choice([start + timedelta(days=r.randint(0, (end - start).days)), start - timedelta(days=r.randint(0, length))])
    else:  # entirely before or after
        b_start = end + timedelta(days=r.randint(1, 20)) if r.random() < 0.5 else start - timedelta(days=length + r.randint(1, 20))
    b_end = b_start + timedelta(days=length)
    return {"enough": wd <= available, "director": wd > director, "blackout": not (end < b_start or start > b_end),
            "slots": {"person": _pick_name(pool, "people", r), "annual_days": fmt.num(annual), "carry_max": fmt.num(carry_max),
                      "director_days": fmt.num(director), "unused": fmt.num(unused), "taken": fmt.num(taken), "start": fmt.date(start),
                      "end": fmt.date(end), "blackout_start": fmt.date(b_start), "blackout_end": fmt.date(b_end)}}


# F9 multi-hop -----------------------------------------------------------


def _lookup(r, fmt, pool):
    depots = pool["depots"]
    cities = r.sample(pool["cities"], min(len(pool["cities"]), r.randint(4, 6)))
    shuffled = r.sample(depots, len(depots))
    city_depot = {c: shuffled[i % len(shuffled)] for i, c in enumerate(cities)}  # at least two depots in use
    customers = [{"id": f"C{r.randint(100, 999)}", "name": _pick_name(pool, "people", r), "city": r.choice(cities)} for _ in range(r.randint(4, 6))]
    orders = [{"id": f"{pool.get('id_prefix', 'O-')}{r.randint(10000, 99999)}", "customer": r.choice(customers)["id"]} for _ in range(r.randint(5, 8))]
    by_id = {c["id"]: c for c in customers}
    depot_of = {o["id"]: city_depot[by_id[o["customer"]]["city"]]["label"] for o in orders}
    first = r.choice(orders)
    want_same = r.random() < 0.5
    pool_same = [o for o in orders if o is not first and depot_of[o["id"]] == depot_of[first["id"]]]
    pool_diff = [o for o in orders if depot_of[o["id"]] != depot_of[first["id"]]]
    second = r.choice((pool_same if want_same else pool_diff) or pool_same or pool_diff)
    keys = pool["keys"]
    raw = {
        keys["customers"]: [{keys["customer_id"]: c["id"], keys["name"]: c["name"], keys["city"]: c["city"]} for c in customers],
        keys["orders"]: [{keys["order_id"]: o["id"], keys["customer_id"]: o["customer"]} for o in orders],
        keys["depots"]: [{keys["city"]: c, keys["depot"]: d["name"]} for c, d in city_depot.items()],
    }
    return {"depot": depot_of[first["id"]], "same": depot_of[first["id"]] == depot_of[second["id"]],
            "slots": {"order_id": first["id"], "order_id_2": second["id"]}, "raw": {"tables": raw}}


def _yes(value: bool) -> str:
    return "yes" if value else "no"


RULES: dict[str, Rule] = {
    r.name: r
    for r in [
        Rule(
            "deadline", "f5", "Someone submits an application, form, payment or document to an organisation that set a deadline.",
            {"person": "a person's full name", "org": "the organisation", "submitted": "the submission date", "deadline": "the deadline date"},
            "It is on time if the submission date is on or before the deadline date; the deadline day itself is still on time.",
            [Q("on_time", "noul", "was it submitted on time under this rule?"),
             Q("lateness", "choice", "how late was it?", {"on_time": "submitted on or before the deadline", "late_1_7": "1 to 7 days after the deadline", "late_8_plus": "8 or more days after the deadline"})],
            _deadline, _deadline_gold),
        Rule(
            "return_window", "f5", "A shop's return rule and a customer's return request.",
            {"person": "the customer's full name", "org": "the shop", "delivered": "delivery date", "requested": "date of the return request", "window_days": "number of days in the return window"},
            "A return request is accepted if it is made no later than {window_days} days after the delivery date: the delivery day is day 0 and a request exactly {window_days} days later is still accepted.",
            [Q("eligible", "noul", "is the request within the return window?"),
             Q("last_day", "choice", "on which weekday is the last day to request a return (delivery date plus {window_days} days)?", options_from="weekday")],
            _window, lambda f: {"eligible": _yes(f["delta"] <= f["window"]), "last_day": f["weekday"]}),
        Rule(
            "business_days", "f5", "An order or application is processed within a number of business days.",
            {"org": "the company or office", "person": "the customer", "ordered": "date the order was placed (a weekday)", "n_days": "number of business days", "promised": "date by which the customer was promised the result"},
            "Business days are Monday to Friday and there are no public holidays in this period. Work starts the next business day after the order date, so the result is ready at the end of the {n_days}-th business day after the order date.",
            [Q("ready_in_time", "noul", "is the result ready on or before the promised date?"),
             Q("ready_weekday", "choice", "on which weekday is the result ready?", options_from="weekday")],
            _business, lambda f: {"ready_in_time": _yes(f["ready"] <= f["promised"]), "ready_weekday": f["ready"].weekday()}),
        Rule(
            "relative_date", "f5", "A message written on a known date mentions an appointment a number of days later.",
            {"person": "a person's name", "today": "the date the message was written", "n_days": "how many days after that date the appointment is"},
            "The appointment is exactly {n_days} calendar days after the date the message was written.",
            [Q("weekday", "choice", "on which weekday is the appointment?", options_from="weekday"),
             Q("same_month", "noul", "is the appointment in the same calendar month and year as the message date?")],
            _relative, lambda f: {"weekday": f["weekday"], "same_month": _yes(f["same_month"])}),
        Rule(
            "age_limit", "f5", "An age requirement for an event, course, contract or benefit.",
            {"person": "the applicant's full name", "birth": "date of birth", "event_date": "the date that counts", "min_age": "minimum age in years"},
            "The person meets the requirement if their {min_age}-th birthday is on or before the date that counts.",
            [Q("old_enough", "noul", "does the person meet the minimum age on the date that counts?")],
            _age, lambda f: {"old_enough": _yes(f["old_enough"])}),
        Rule(
            "event_order", "f5", "Two events of one case with their dates, possibly written in different date formats (for example a contract signature and a payment).",
            {"org": "the organisation", "date_a": "date of the first-named event (event A)", "date_b": "date of the second-named event (event B)"},
            "Event A happened on {date_a} and event B on {date_b}; the two dates are never the same day.",
            [Q("a_before_b", "noul", "did event A happen before event B?")],
            _order, lambda f: {"a_before_b": _yes(f["a_first"])}),
        Rule(
            "sum_limit", "f6", "An expense claim, basket or order with several line items and a spending limit.",
            {"person": "the person", "org": "the company", "items": "the list of line items", "limit": "the spending limit"},
            "The total is the sum of all line items. The total exceeds the limit only if it is strictly greater than the limit; a total equal to the limit does not exceed it.",
            [Q("over_limit", "noul", "does the total exceed the limit?"),
             Q("largest_item", "choice", "which line item has the highest amount?", options_from="items")],
            _sum_limit, lambda f: {"over_limit": _yes(f["over"]), "largest_item": f["largest"]},
            vocab={"items": ("priced", 10, "typical line items for this setting with realistic prices in {currency}")},
            lists={"items": ["name", "amount"]}, money=("limit", "amount")),
        Rule(
            "discount", "f6", "A product price with a percentage discount compared with a budget.",
            {"product": "the product", "price": "the regular price", "discount": "the discount in percent (a number without the % sign)", "budget": "the buyer's budget"},
            "Final price = regular price x (100 - discount) / 100, rounded to the cent. The purchase fits the budget if the final price is less than or equal to the budget.",
            [Q("within_budget", "noul", "does the discounted price fit the budget?")],
            _discount, lambda f: {"within_budget": _yes(f["within"])},
            vocab={"items": ("priced", 10, "products for this setting with realistic prices in {currency}")}, money=("price", "budget")),
        Rule(
            "count_records", "f6", "A list of records (shipments, tickets, test runs, deliveries) each with an id, a status and a number.",
            {"records": "the list of records", "status": "the status to count", "threshold": "the minimum value"},
            "Count the records whose status is {status} and whose value is at least {threshold} (a value equal to {threshold} counts).",
            [Q("count", "choice", "how many records match?", options_from="counts")],
            _count, lambda f: {"count": f["count"]},
            vocab={"statuses": ("labeled", 3, "record statuses"), "id_prefix": ("text", 1, "a short record id prefix such as SHP-")},
            lists={"records": ["id", "status", "value"]}, q_slots=("status", "threshold")),
        Rule(
            "unit_limit", "f6", "A quantity (weight, volume or length) checked against a limit written in a larger unit.",
            {"item": "the thing being checked", "amount": "the measured quantity with its unit", "limit": "the limit with its unit"},
            "Convert both to the same unit (1 kg = 1000 g, 1 l = 1000 ml, 1 m = 100 cm, 1 cm = 10 mm). It is within the limit if the quantity is less than or equal to the limit.",
            [Q("within_limit", "noul", "is the quantity within the limit?")],
            _units, lambda f: {"within_limit": _yes(f["within"])},
            vocab={"things": ("strings", 8, "things that are weighed or measured in this setting")}, units=("amount", "limit")),
        Rule(
            "average_pass", "f6", "Someone's scores and a passing average.",
            {"person": "the person", "scores": "the list of scores", "pass_mark": "the minimum average needed"},
            "The average is the arithmetic mean of all scores. The person passes if the average is greater than or equal to the pass mark.",
            [Q("passes", "noul", "does the person pass?")],
            _average, lambda f: {"passes": _yes(f["passes"])}),
        Rule(
            "expense_policy", "f12", "A company travel expense policy and one employee's travel claim.",
            {"person": "the employee", "city": "the destination city", "nights": "hotel nights", "days": "trip days", "hotel_total": "total hotel cost",
             "hotel_cap": "standard hotel cap per night", "expensive_cap": "higher hotel cap per night in the expensive cities", "meal_cap": "meal allowance per day",
             "meal_total": "total meal cost", "receipt_min": "amount above which a single expense needs a receipt", "big_item": "one other expense",
             "big_amount": "the amount of that expense", "receipt_status": "whether its receipt is attached"},
            "Hotel: at most {hotel_cap} per night, except in the expensive cities named in the policy, where the cap is {expensive_cap} per night; the hotel is within policy if the hotel total is at most nights x the applicable cap. Meals: at most {meal_cap} per day; within policy if the meal total is at most days x {meal_cap}. Receipts: any single expense above {receipt_min} needs a receipt; the claim states that hotel and meal receipts are attached, so only {big_item} can break the receipt rule.",
            [Q("hotel_ok", "noul", "is the hotel cost within the policy?"), Q("meals_ok", "noul", "are the meal costs within the policy?"),
             Q("receipts_ok", "noul", "does the claim meet the receipt rule?")],
            _expense, lambda f: {"hotel_ok": _yes(f["hotel_ok"]), "meals_ok": _yes(f["meals_ok"]), "receipts_ok": _yes(f["receipt_ok"])},
            vocab={"expensive_cities": ("strings", 3, "the cities your policy names as expensive, exactly as written in the policy"),
                   "other_cities": ("strings", 4, "other cities, not named as expensive"), "items": ("labeled", 6, "other expense types"),
                   "receipt_attached": ("text", 1, "the short phrase the claim uses when that expense's receipt is attached"),
                   "receipt_missing": ("text", 1, "the short phrase the claim uses when that receipt is missing")},
            money=("hotel_total", "hotel_cap", "expensive_cap", "meal_cap", "meal_total", "receipt_min", "big_amount"), min_words=200,
            extra="Write the policy as a realistic internal document of 250 to 450 words with numbered clauses, including 3 to 5 extra clauses on other matters (booking channels, approvals, submission deadlines) that do not affect any question."),
        Rule(
            "return_policy", "f12", "An online shop's return policy and one customer's return request.",
            {"person": "the customer", "org": "the shop", "product": "the product", "days_since": "days since delivery", "std_days": "standard return window in days",
             "el_days": "return window for electronics in days", "opened_status": "whether the product was opened", "sale_status": "whether it was bought on sale"},
            "1) Opened hygiene products can never be returned. 2) The return window is {el_days} days for electronics and {std_days} days for everything else, counted from delivery; a request on the last day is accepted. 3) Items bought on sale are refunded as store credit instead of money. 4) Everything else gets a full refund.",
            [Q("outcome", "choice", "what does the customer get?", {"full_refund": "a full refund of the money", "store_credit": "store credit only", "not_returnable": "no return possible"}),
             Q("within_window", "noul", "is the request within the return window for this product?")],
            _returns, lambda f: {"outcome": f["outcome"], "within_window": _yes(f["within"])},
            vocab={"products": ("groups", 3, 'product names grouped as {"electronics": [...], "hygiene": [...], "other": [...]}'),
                   "opened": ("text", 1, "the short phrase the request uses for an opened product"), "unopened": ("text", 1, "the phrase for an unopened product"),
                   "on_sale": ("text", 1, "the phrase for a product bought on sale"), "full_price": ("text", 1, "the phrase for a product bought at full price")},
            min_words=200,
            extra="Write the policy as a realistic page of 250 to 450 words with numbered clauses, including 3 to 5 extra clauses (shipping costs, packaging, how to start a return) that do not affect any question."),
        Rule(
            "insurance_claim", "f12", "A household insurance policy excerpt and one theft claim.",
            {"person": "the policy holder", "org": "the insurer", "deductible": "deductible per claim", "el_sublimit": "maximum paid for electronics per claim",
             "max_payout": "maximum payout per claim", "el_value": "value of stolen electronics", "jw_value": "value of stolen jewellery",
             "other_value": "value of other stolen items", "jw_status": "whether the jewellery was declared in the policy", "threshold": "an amount to compare the payout with"},
            "Payout = min({max_payout}, max(0, min(electronics value, {el_sublimit}) + jewellery value only if the jewellery was declared + other value - {deductible})). Undeclared jewellery is excluded.",
            [Q("electronics_capped", "noul", "is the electronics value above the electronics sublimit?"),
             Q("jewellery_covered", "noul", "is the stolen jewellery covered?"),
             Q("payout_over", "noul", "is the payout above {threshold}?")],
            _insurance, lambda f: {"electronics_capped": _yes(f["capped"]), "jewellery_covered": _yes(f["jewellery"]), "payout_over": _yes(f["over"])},
            vocab={"declared": ("text", 1, "the short phrase the claim uses when the jewellery was declared in the policy"),
                   "not_declared": ("text", 1, "the phrase when it was not declared")},
            money=("deductible", "el_sublimit", "max_payout", "el_value", "jw_value", "other_value", "threshold"), q_slots=("threshold",), min_words=200,
            extra="Write the policy excerpt as 250 to 450 words with numbered clauses, including 3 to 5 extra clauses (reporting deadlines, police report, fire damage) that do not affect any question."),
        Rule(
            "leave_policy", "f12", "A company leave policy and one employee's leave request.",
            {"person": "the employee", "annual_days": "annual leave in working days", "carry_max": "maximum days carried over from last year",
             "director_days": "longest request (in working days) that does not need director approval", "unused": "unused days from last year",
             "taken": "days already taken this year", "start": "first day of leave", "end": "last day of leave", "blackout_start": "start of the blackout period",
             "blackout_end": "end of the blackout period"},
            "Available days = {annual_days} + the smaller of {unused} and {carry_max} - {taken}. The request counts Monday to Friday days from {start} to {end}, both included, and there are no public holidays in it. It needs director approval if it is longer than {director_days} working days. It conflicts with the blackout period if any day from {start} to {end} falls between {blackout_start} and {blackout_end}, both included.",
            [Q("enough_days", "noul", "does the employee have enough available days?"), Q("needs_director", "noul", "does the request need director approval?"),
             Q("blackout_conflict", "noul", "does the request conflict with the blackout period?")],
            _leave, lambda f: {"enough_days": _yes(f["enough"]), "needs_director": _yes(f["director"]), "blackout_conflict": _yes(f["blackout"])},
            extra="Write the policy as 250 to 450 words with numbered clauses, including 3 to 5 extra clauses (sick leave reporting, parental leave, how to file a request) that do not affect any question.",
            min_words=200),
        Rule(
            "depot_lookup", "f9", "A small database export with customers, orders and a table assigning each city to a depot.",
            {"order_id": "an order id", "order_id_2": "a second order id"},
            "An order ships from the depot assigned to the city of the customer who placed it (order -> customer id -> city -> depot).",
            [Q("depot", "choice", "which depot ships order {order_id}?", options_from="depots"),
             Q("same_depot", "noul", "do orders {order_id} and {order_id_2} ship from the same depot?")],
            _lookup, lambda f: {"depot": f["depot"], "same_depot": _yes(f["same"])},
            vocab={"keys": ("keys", 8, "the JSON key to use for customers, orders, depots, customer_id, name, city, order_id and depot"),
                   "cities": ("strings", 6, "cities"), "depots": ("labeled", 3, "depots"), "id_prefix": ("text", 1, "an order id prefix")},
            q_slots=("order_id", "order_id_2"),
            extra='The state is a JSON object with an "intro" text field and a field whose value is exactly "{tables}"; code inserts the three tables there.'),
    ]
}
SETS = {"f5": [n for n, r in RULES.items() if r.fset == "f5"], "f6": [n for n, r in RULES.items() if r.fset == "f6"],
        "f9": ["depot_lookup"], "f12": [n for n, r in RULES.items() if r.fset == "f12"]}
SETS["f10"] = [n for n in SETS["f5"] + SETS["f6"] + SETS["f12"] if any(q.kind == "noul" for q in RULES[n].questions)]


# ---------------------------------------------------------------- template request and checks

TEMPLATE_PROMPT = """Write {k} surface-text variants in {language} for a data generator. Code fills the placeholders with random values and computes the correct answers, so every variant must express the rule below with exactly the same meaning.

Setting: {setting}
Placeholders (use each at least once, in the state or in the questions, written exactly as shown, braces included):
{slots}
The rule that decides the answers (the state must contain it, or the questions must state it):
{rule}
Questions (keep each key; choose your own ASCII snake_case field name):
{questions}
{vocab}{extra}
Requirements:
- Write natively in {language}, realistic for the setting; each variant uses a different format ({formats}).
- Say the rule in your own words and word it differently in each variant, the way such a document would (terms, a policy excerpt, a note from the office). Never copy sentences from this brief.
- The state gives the facts but never the answers.
- Questions ask in general terms: they never repeat facts or numbers from the case and never walk through the calculation.{q_slots} No other braces anywhere.
- The state never talks about the questions (no "please answer the questions below").
- Outcome letters only link your criteria entries to the outcomes: never write them in any text.
- Add no facts or conditions that could change an answer (no extensions, holidays, exceptions or rounding rules beyond the rule).
- A JSON-record state is a JSON object with placeholders inside string values; a value that is exactly one placeholder, like "{{deadline}}", receives the raw value.
Reply with JSON only: {{"variants": [{{"format": "...", "state": <string or JSON object>, "questions": {{"<key>": {{"field": "...", "instructions": "...", "criteria": ...}}}}{list_keys}}}]{vocab_key}}}"""

VOCAB_SHAPES = {
    "priced": '[{{"label": "<ASCII snake_case>", "name": "<{language} name>", "price_min": <number>, "price_max": <number>}}, ...]',
    "labeled": '[{{"label": "<ASCII snake_case>", "name": "<{language} name>"}}, ...]',
    "strings": '["...", ...]',
    "text": '"..."',
    "groups": '{{"electronics": ["...", ...], "hygiene": ["...", ...], "other": ["...", ...]}}',
    "keys": '{{"customers": "<key>", "orders": "<key>", "depots": "<key>", "customer_id": "<key>", "name": "<key>", "city": "<key>", "order_id": "<key>", "depot": "<key>"}}',
}
CURRENCY_WORDS = r"[$€£₺]|\b(?:TL|TRY|EUR|USD|GBP|CHF|Euro|Euros|Dollar|Dollars|dollars|lira|Lira)\b"
UNIT_WORDS = r"\b(?:g|kg|ml|l|cm|m|mm|gram|grams|Gramm|kilo|Kilo|litre|liter|Liter|metre|meter|Meter)\b"


def _letter(index: int) -> str:
    return chr(65 + index)


def template_messages(rule: Rule, lang: str, k: int = 3) -> list[dict]:
    language = LANG_NAMES[lang]
    slots = "\n".join(f"- {{{name}}}: {meaning}" for name, meaning in rule.slots.items())
    qs = []
    for q in rule.questions:
        if q.kind == "noul":
            qs.append(f'- key `{q.key}`, yes/no: {q.meaning} Give "instructions" and "criteria": {{"yes": "...", "no": "..."}} describing each answer.')
        elif q.options_from:
            qs.append(f'- key `{q.key}`, choice: {q.meaning} Give "instructions" only; code adds the options ({q.options_from}).')
        else:
            outs = "; ".join(f"{_letter(i)} = {meaning}" for i, meaning in enumerate(q.outcomes.values()))
            qs.append(f'- key `{q.key}`, choice: {q.meaning} Outcomes: {outs}. Give "instructions" and "criteria": a list of {{"key": "<outcome letter>", "label": "<your own ASCII snake_case label>", "description": "<{language} description>"}}, one per outcome.')
    list_keys = ""
    for name, parts in rule.lists.items():
        ph = ", ".join("{" + p + "}" for p in parts)
        key_map = ", ".join(f'"{p}": "<JSON key>"' for p in parts)
        list_keys += f', "{name}_line": "<one line for one entry, using {ph}>", "{name}_keys": {{{key_map}}}'
        slots += f"\n  (The {{{name}}} placeholder becomes one line per entry using your \"{name}_line\" template, or a JSON array in a JSON state.)"
    if rule.money:
        slots += "\n  (" + ", ".join("{" + m + "}" for m in rule.money) + " receive an amount that already includes its currency symbol: never write a currency sign or code next to them.)"
    if rule.units:
        slots += "\n  (" + ", ".join("{" + u + "}" for u in rule.units) + " receive a number that already includes its unit: never write a unit next to them.)"
    currency = {"en": "US dollars or pounds", "tr": "Turkish lira", "de": "euros"}[lang]
    vocab = ""
    if rule.vocab:
        vocab = 'Also give "vocab" once, shared by all variants, with exactly these keys:\n' + "\n".join(
            f'- "{key}": {VOCAB_SHAPES[kind].format(language=language)} with at least {n} entries: {meaning.replace("{currency}", currency)}'
            if kind not in {"text", "groups", "keys"} else f'- "{key}": {VOCAB_SHAPES[kind].format(language=language)}: {meaning}'
            for key, (kind, n, meaning) in rule.vocab.items()
        ) + "\n"
    content = TEMPLATE_PROMPT.format(
        k=k, language=language, setting=rule.setting, slots=slots, rule=rule.rule, questions="\n".join(qs), vocab=vocab,
        extra=(rule.extra + "\n") if rule.extra else "", formats=FORMATS, list_keys=list_keys,
        vocab_key=', "vocab": {...}' if rule.vocab else "",
        q_slots=(" Questions may use only these placeholders: " + ", ".join("{" + x + "}" for x in rule.q_slots) + ".") if rule.q_slots else " Questions use no placeholders.",
    )
    return [{"role": "user", "content": content}]


PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def check_variant(rule: Rule, v: Any) -> Mapping:
    """Raise ValueError when a teacher's template cannot be used as-is."""
    from .prompts import copied
    from .validate import FIELD_RE, LABEL_RE

    if not isinstance(v, Mapping) or not isinstance(v.get("state"), (str, dict)) or not isinstance(v.get("questions"), Mapping):
        raise ValueError("variant needs a state (string or object) and questions")
    texts = list(_strings([v["state"], v["questions"], [v.get(f"{n}_line", "") for n in rule.lists]]))
    used = {m for t in texts for m in PLACEHOLDER.findall(t)}
    missing = set(rule.slots) - used
    if missing:
        raise ValueError(f"placeholders missing from state and questions: {sorted(missing)}")
    stray = used - set(rule.slots) - {"tables"} - {p for parts in rule.lists.values() for p in parts}
    if stray:
        raise ValueError(f"unknown placeholders {sorted(stray)}")
    if rule.name == "depot_lookup" and "tables" not in {m for t in _strings(v["state"]) for m in PLACEHOLDER.findall(t)}:
        raise ValueError('the state needs a field whose value is exactly "{tables}"')
    for t in texts:
        for slot in rule.money:
            if re.search(rf"(?:{CURRENCY_WORDS})\s*\{{{slot}\}}|\{{{slot}\}}\s*(?:{CURRENCY_WORDS})", t):
                raise ValueError(f"{{{slot}}} already includes its currency; remove the currency next to it")
        for slot in rule.units:
            if re.search(rf"\{{{slot}\}}\s*{UNIT_WORDS}", t):
                raise ValueError(f"{{{slot}}} already includes its unit; remove the unit next to it")
        grab = copied(PLACEHOLDER.sub(" ", t))
        if grab:
            raise ValueError(f"copies the brief word for word ('{grab}'); write it in your own words")
    in_questions = {m for t in _strings(v["questions"]) for m in PLACEHOLDER.findall(t)}
    if in_questions - set(rule.q_slots):
        raise ValueError(f"questions may only use {sorted(rule.q_slots) or 'no placeholders'}; they must not repeat case facts")
    if rule.min_words and len(" ".join(_strings(v["state"])).split()) < rule.min_words:
        raise ValueError(f"the policy is too short: write at least {rule.min_words} words with the extra clauses")
    if any(re.search(r"(?<![\w])[A-E]\s*[=:]\s", t) for t in _strings(v["questions"])):
        raise ValueError("outcome letters (A = ...) must not appear in any text")
    fields = set()
    for q in rule.questions:
        got = v["questions"].get(q.key)
        if not isinstance(got, Mapping) or not FIELD_RE.match(str(got.get("field", ""))) or not got.get("instructions"):
            raise ValueError(f"question {q.key} needs field and instructions")
        if got["field"] in fields:
            raise ValueError("duplicate field names")
        fields.add(got["field"])
        crit = got.get("criteria")
        if q.kind == "noul" and not (isinstance(crit, Mapping) and crit.get("yes") and crit.get("no")):
            raise ValueError(f"question {q.key} needs criteria yes and no")
        if q.kind == "choice" and not q.options_from:
            letters = {_letter(i) for i in range(len(q.outcomes))}
            if not isinstance(crit, list) or {str(c.get("key", "")).strip().upper() for c in crit if isinstance(c, Mapping)} != letters:
                raise ValueError(f"question {q.key} needs one criteria entry per outcome {sorted(letters)}")
            labels = [str(c.get("label", "")) for c in crit]
            if any(not LABEL_RE.match(label) or len(label) < 3 for label in labels) or len({x.casefold() for x in labels}) != len(labels):
                raise ValueError(f"question {q.key}: labels must be distinct ASCII snake_case words of your own")
            if any(not c.get("description") for c in crit):
                raise ValueError(f"question {q.key}: every outcome needs a description")
    for name, parts in rule.lists.items():
        line = v.get(f"{name}_line")
        if not isinstance(line, str) or any("{" + p + "}" not in line for p in parts):
            raise ValueError(f"{name}_line must use {parts}")
    return v


def check_vocab(rule: Rule, vocab: Any) -> Mapping:
    """Word lists shared by a rule's variants must have exactly the declared shapes."""
    if not rule.vocab:
        return {}
    if not isinstance(vocab, Mapping):
        raise ValueError('"vocab" object missing')
    snake = re.compile(r"^[a-z0-9_]{1,40}$")
    for key, (kind, n, _) in rule.vocab.items():
        got = vocab.get(key)
        if kind == "text":
            ok = isinstance(got, str) and bool(got.strip())
        elif kind == "strings":
            ok = isinstance(got, list) and len(got) >= n and all(isinstance(x, str) and x.strip() for x in got)
        elif kind in {"labeled", "priced"}:
            ok = isinstance(got, list) and len(got) >= n and all(
                isinstance(x, Mapping) and snake.match(str(x.get("label", ""))) and str(x.get("name", "")).strip() for x in got)
            ok = ok and len({x["label"] for x in got}) == len(got)
            if ok and kind == "priced":
                try:
                    ok = all(0 < float(x["price_min"]) <= float(x["price_max"]) for x in got)
                except (KeyError, TypeError, ValueError):
                    ok = False
        elif kind == "groups":
            ok = isinstance(got, Mapping) and all(isinstance(got.get(g), list) and len(got[g]) >= 2 for g in ("electronics", "hygiene", "other"))
        else:  # keys
            wanted = ("customers", "orders", "depots", "customer_id", "name", "city", "order_id", "depot")
            ok = isinstance(got, Mapping) and all(snake.match(str(got.get(w, ""))) for w in wanted) and len({got[w] for w in wanted}) == len(wanted)
        if not ok:
            raise ValueError(f'vocab.{key} must be {VOCAB_SHAPES[kind].format(language="...")} with at least {n} entries')
    return vocab


# ---------------------------------------------------------------- filling


def fill(value: Any, text: Mapping[str, str], raw: Mapping[str, Any]) -> Any:
    """Replace {slot} in every string; a JSON value that is exactly "{slot}" takes the raw value."""
    if isinstance(value, str):
        whole = PLACEHOLDER.fullmatch(value.strip())
        if whole and whole.group(1) in raw:
            return raw[whole.group(1)]
        return PLACEHOLDER.sub(lambda m: str(text.get(m.group(1), m.group(0))), value)
    if isinstance(value, Mapping):
        return {k: fill(v, text, raw) for k, v in value.items()}
    if isinstance(value, list):
        return [fill(v, text, raw) for v in value]
    return value


def instance(rule: Rule, variant: Mapping, vocab: Mapping, names: Mapping, lang: str, r: random.Random, *, invert: str | None = None) -> dict:
    """One filled row: state, questions and exact gold labels (question field -> label)."""
    fmt = Fmt(lang, r)
    pool = {**names, **vocab}
    facts = rule.sample(r, fmt, pool)
    gold = rule.gold(facts)
    text = dict(facts["slots"])
    raw = dict(facts.get("raw", {}))
    json_state = isinstance(variant["state"], Mapping)
    for name, parts in rule.lists.items():
        entries = facts["lines"][name]
        line = variant[f"{name}_line"]
        text[name] = "\n".join(PLACEHOLDER.sub(lambda m, e=e: str(e.get(m.group(1), m.group(0))), line) for e in entries)
        keys = variant.get(f"{name}_keys") if isinstance(variant.get(f"{name}_keys"), Mapping) else {}
        raw_entries = facts["raw"][name] if name in facts.get("raw", {}) else entries
        raw[name] = [{str(keys.get(p) or p): e[p] for p in parts} for e in raw_entries]
    if "tables" in raw:
        text["tables"] = text_of(raw["tables"])
    state = fill(variant["state"], text, raw if json_state else {})
    questions, answers = {}, {}
    weekdays = fmt.weekdays()
    for q in rule.questions:
        spec = variant["questions"][q.key]
        entry: dict[str, Any] = {"type": q.kind, "instructions": fill(spec["instructions"], text, {})}
        label = gold[q.key]
        if q.kind == "noul":
            crit = {"yes": fill(spec["criteria"]["yes"], text, {}), "no": fill(spec["criteria"]["no"], text, {})}
            if invert == q.key:  # F10: yes now describes the opposite outcome; follow the written mapping
                crit = {"yes": crit["no"], "no": crit["yes"]}
                label = "no" if label == "yes" else "yes"
            entry["criteria"] = crit
        elif q.options_from == "weekday":
            days = weekdays[:5] if q.key == "ready_weekday" else weekdays
            entry["criteria"] = {lab: name for lab, name in days}
            label = weekdays[label][0]
        elif q.options_from == "items":
            entry["criteria"] = {i["label"]: i["name"] for i in facts["items"]}
        elif q.options_from == "counts":
            entry["criteria"] = {str(n): None for n in range(facts["n"] + 1)}
        elif q.options_from == "depots":
            entry["criteria"] = {d["label"]: d["name"] for d in pool["depots"]}
        else:  # teacher-labeled outcomes arrive keyed by letter in the rule's outcome order
            letter = _letter(list(q.outcomes).index(label))
            crit_list = spec["criteria"]
            entry["criteria"] = {c["label"]: fill(c["description"], text, {}) for c in crit_list}
            label = next(c["label"] for c in crit_list if str(c["key"]).strip().upper() == letter)
        questions[spec["field"]] = entry
        answers[spec["field"]] = label
    return {"state": state, "questions": questions, "gold": answers, "locale": fmt.locale}
