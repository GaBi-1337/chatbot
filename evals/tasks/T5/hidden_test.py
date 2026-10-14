"""Hidden behaviour tests for T5 (kept outside the fixture). WORKDIR env var = the agent's work dir."""
import importlib.util
import os
import random
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

import pytest

W = Path(os.environ["WORKDIR"])
H = Path(__file__).resolve().parent / "hidden"
sys.path.insert(0, str(W))
sys.dont_write_bytecode = True

from shop import catalog, discounts, money, shipping, tax  # noqa: E402
from shop.cart import Cart  # noqa: E402
from shop.invoice import build_invoice, render_invoice  # noqa: E402
from shop import report  # noqa: E402

PRICES = {"A100": 1999, "B200": 435, "C300": 4510, "D400": 29, "E500": 1250, "F600": 115}
HEAVY = {"C300"}
RATES = {"CA": Fraction(725, 100), "NY": Fraction(8875, 1000), "TX": Fraction(625, 100), "OR": Fraction(0)}


def hu(x):  # exact half-up of a non-negative Fraction
    return int((Decimal(x.numerator) / Decimal(x.denominator)).quantize(Decimal(1), rounding=ROUND_HALF_UP)) \
        if x.denominator != 1 else int(x)


def hu_exact(x):
    n, d = x.numerator, x.denominator
    return (2 * n + d) // (2 * d)


def run(*args):
    return subprocess.run([sys.executable, "-m", "shop", *args], cwd=W, capture_output=True, text=True,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def test_old_util_gone_and_money_present():
    assert not (W / "shop" / "util.py").exists()
    assert importlib.util.find_spec("shop.util") is None
    assert (W / "shop" / "money.py").exists()


def test_catalog_cents():
    for sku, c in PRICES.items():
        assert catalog.get_price(sku) == c and type(catalog.get_price(sku)) is int


def test_money_parse_format():
    assert money.parse_dollars("12.50") == 1250
    assert money.parse_dollars("$1,234.5") == 123450
    assert money.parse_dollars(" 7 ") == 700
    assert money.parse_dollars("4.35") == 435
    assert money.parse_dollars("0.29") == 29
    assert money.parse_dollars("-3.20") == -320
    for bad in ("abc", "1.005", "", "$"):
        with pytest.raises(ValueError):
            money.parse_dollars(bad)
    assert money.format_cents(1234550) == "$12,345.50"
    assert money.format_cents(-320) == "-$3.20"
    assert money.format_cents(5) == "$0.05"
    assert money.format_cents(0) == "$0.00"
    assert money.format_cents(100000000) == "$1,000,000.00"


def test_cart_cents():
    c = Cart()
    c.add("B200", 3)
    c.add("D400", 7)
    assert c.line_total("B200") == 1305
    assert c.subtotal() == 1508 and type(c.subtotal()) is int


def test_percent_and_tax_half_up():
    assert discounts.percent_of(40, 6.25) == 3      # 2.5 -> 3 (not banker's 2)
    assert discounts.percent_of(10, 25) == 3        # 2.5 -> 3
    assert discounts.percent_of(50, 50) == 25
    assert discounts.percent_of(1005, 10) == 101    # 100.5 -> 101
    assert tax.tax_for(40, "TX") == 3
    assert tax.tax_for(1000, "NY") == 89            # 88.75
    assert tax.tax_for(10, "NY") == 1               # 0.8875
    assert tax.tax_for(4, "NY") == 0                # 0.355
    assert tax.tax_for(10000, "CA") == 725
    with pytest.raises(ValueError):
        tax.tax_for(100, "XX")


def test_discounts_cents():
    assert discounts.bulk_discount(250, 10) == 13   # 12.5 -> 13
    assert discounts.bulk_discount(10000, 9) == 0
    assert discounts.coupon_discount(5, "HALF") == 3
    assert discounts.coupon_discount(1000, "FIVEOFF") == 500
    assert discounts.coupon_discount(300, "FIVEOFF") == 300
    with pytest.raises(ValueError):
        discounts.coupon_discount(100, "NOPE")


def test_shipping_cents():
    assert shipping.shipping_cost(4999) == 499
    assert shipping.shipping_cost(5000) == 0
    assert shipping.shipping_cost(5000, 2) == 250
    assert shipping.shipping_cost(100, 1) == 624


def oracle(items, region, coupon, credit):
    lines = []
    heavy = 0
    for sku, q in sorted(items.items()):
        total = PRICES[sku] * q
        disc = hu_exact(Fraction(total * 5, 100)) if q >= 10 else 0
        lines.append((sku, q, PRICES[sku], total, disc))
        if sku in HEAVY:
            heavy += q
    sub = sum(l[3] for l in lines)
    bulk = sum(l[4] for l in lines)
    after = sub - bulk
    cd = 0
    if coupon == "SAVE10":
        cd = min(hu_exact(Fraction(after * 10, 100)), after)
    elif coupon == "HALF":
        cd = min(hu_exact(Fraction(after * 50, 100)), after)
    elif coupon == "FIVEOFF":
        cd = min(500, after)
    taxable = after - cd
    ship = (0 if taxable >= 5000 else 499) + 125 * heavy
    tx = hu_exact(Fraction(taxable) * RATES[region] / 100)
    gross = taxable + ship + tx
    cr = min(credit, gross)
    return dict(subtotal=sub, bulk_discount=bulk, coupon_discount=cd, taxable=taxable, shipping=ship, tax=tx,
                credit_applied=cr, total=gross - cr), lines


def random_order(rng):
    items = {}
    for _ in range(rng.randint(1, 4)):
        items[rng.choice(list(PRICES))] = rng.choice([1, 2, 3, 5, 9, 10, 11, 25, 40])
    return (items, rng.choice(list(RATES)), rng.choice([None, "SAVE10", "HALF", "FIVEOFF"]),
            rng.choice([0, 0, 250, 750, 100000]))


def make_cart(items):
    c = Cart()
    for sku, q in items.items():
        c.add(sku, q)
    return c


def test_invoices_match_oracle():
    rng = random.Random(20261013)
    for _ in range(400):
        items, region, coupon, credit = random_order(rng)
        inv = build_invoice(make_cart(items), region, coupon, credit)
        want, lines = oracle(items, region, coupon, credit)
        for k, v in want.items():
            got = getattr(inv, k)
            assert type(got) is int and got == v, (items, region, coupon, credit, k, got, v)
        assert [(l.sku, l.qty, l.unit_price, l.line_total, l.discount) for l in inv.lines] == lines


def test_report_matches_oracle():
    rng = random.Random(7)
    for _ in range(60):
        invs, wants = [], []
        for _ in range(rng.randint(1, 5)):
            items, region, coupon, credit = random_order(rng)
            invs.append(build_invoice(make_cart(items), region, coupon, credit))
            wants.append(oracle(items, region, coupon, credit))
        total = sum(w[0]["total"] for w in wants)
        assert report.total_revenue(invs) == total and type(report.total_revenue(invs)) is int
        assert report.average_order_value(invs) == hu_exact(Fraction(total, len(invs)))
        rev = {}
        for _, lines in wants:
            for sku, q, u, t, d in lines:
                rev[sku] = rev.get(sku, 0) + t - d
        assert report.revenue_by_sku(invs) == dict(sorted(rev.items()))
    assert report.average_order_value([]) == 0


def test_average_tie_rounds_up():
    a, b = Cart(), Cart()
    a.add("D400", 1)   # 29 + 499 shipping = 528 (OR, no tax)
    b.add("D400", 2)   # 58 + 499 = 557
    invs = [build_invoice(a, "OR"), build_invoice(b, "OR")]
    assert report.total_revenue(invs) == 1085
    assert report.average_order_value(invs) == 543   # 542.5 -> 543


def test_render_and_cli_match_golden():
    inv = build_invoice(make_cart({"A100": 3, "B200": 12, "C300": 1, "D400": 7}), "CA")
    assert render_invoice(inv) == (H / "golden_invoice1.txt").read_text()
    r = run("invoice", str(H / "order1.json"), "--region", "CA")
    assert r.returncode == 0 and r.stdout == (H / "golden_invoice1.txt").read_text()
    r = run("invoice", str(H / "order1.json"), "--region", "NY", "--coupon", "SAVE10", "--credit", "$7.50")
    assert r.returncode == 0 and r.stdout == (H / "golden_invoice2.txt").read_text()
    r = run("report", str(H / "orders.json"))
    assert r.returncode == 0 and r.stdout == (H / "golden_report.txt").read_text()


def test_cli_errors_exit_2():
    assert run("invoice", str(H / "order1.json"), "--region", "XX").returncode == 2
    assert run("invoice", str(H / "order1.json"), "--region", "CA", "--credit", "abc").returncode == 2
    assert run("invoice", str(H / "missing.json"), "--region", "CA").returncode == 2
