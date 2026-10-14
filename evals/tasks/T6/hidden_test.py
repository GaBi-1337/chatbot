"""Hidden behaviour tests for T6 (kept outside the fixture). WORKDIR env var = the agent's work dir."""
import os
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

W = Path(os.environ["WORKDIR"])
sys.path.insert(0, str(W))
sys.dont_write_bytecode = True

from ledger import fx, summary  # noqa: E402
from ledger.ingest import parse_entries, parse_timestamp  # noqa: E402
from ledger.periods import fiscal_quarter, fiscal_year  # noqa: E402

DEFAULT = {"USD": 1.0, "EUR": 1.10, "GBP": 1.30, "JPY": 0.0070}


def test_timestamp_is_local_date():
    cases = {
        "2025-04-01T00:30:00+05:30": date(2025, 4, 1),
        "2026-03-31T23:30:00-05:00": date(2026, 3, 31),
        "2025-12-31T23:59:59Z": date(2025, 12, 31),
        "2025-01-01T00:00:00+14:00": date(2025, 1, 1),
        "2025-06-30T23:59:59-12:00": date(2025, 6, 30),
        "2025-03-31T23:59:59": date(2025, 3, 31),
        "2025-06-30": date(2025, 6, 30),
    }
    for text, want in cases.items():
        assert parse_timestamp(text) == want, text


def test_rate_overrides_do_not_leak():
    before = {c: v["rate"] for c, v in fx.load_rates().items()}
    assert before == DEFAULT
    r = fx.load_rates({"EUR": 1.2, "CHF": 1.05})
    assert r["EUR"]["rate"] == 1.2 and r["CHF"]["rate"] == 1.05
    r2 = fx.load_rates({"GBP": 2.0})
    assert r2["EUR"]["rate"] == 1.10 and "CHF" not in r2
    now = fx.load_rates()
    assert {c: v["rate"] for c, v in now.items()} == DEFAULT
    assert fx.convert(100, "EUR") == 110.0
    assert fx.convert(100, "EUR", r) == 120.0
    assert fx.convert(100, "GBP", r) == 130.0
    assert fx.convert(100, "GBP", r2) == 200.0


def test_caller_mutation_does_not_leak():
    r = fx.load_rates()
    r["EUR"]["rate"] = 9.0
    r["EUR"]["source"] = "hacked"
    r["JPY"] = {"rate": 1.0, "source": "x"}
    now = fx.load_rates()
    assert now["EUR"] == {"rate": 1.10, "source": "ecb-monthly"}
    assert now["JPY"]["rate"] == 0.0070
    assert fx.convert(100, "EUR") == 110.0


def test_override_keeps_other_fields():
    r = fx.load_rates({"EUR": 1.5})
    assert r["EUR"]["source"] == "ecb-monthly"


def oracle(entries_text, rates):
    out = []
    for line in entries_text.splitlines():
        if not line or line.startswith(("#", "id,")):
            continue
        _id, acct, amt, cur, ts = [p.strip() for p in line.split(",")]
        y, m, d = map(int, ts[:10].split("-"))
        fy = y if m >= 4 else y - 1
        q = ((m - 4) % 12) // 3 + 1
        out.append((acct, round(float(amt) * rates[cur.upper()], 2), fy, q))
    return out


def make_text(seed):
    rng = random.Random(seed)
    lines = ["id,account,amount,currency,timestamp"]
    for i in range(1, 80):
        base = datetime(2024, 1, 1) + timedelta(days=rng.randint(0, 900), minutes=rng.choice([0, 1, 30, 719, 1439]))
        off = rng.choice([None, "Z", "+05:30", "-05:00", "+13:00", "-11:00", "+00:00"])
        ts = base.isoformat() + (off or "")
        lines.append(f"{i},{rng.choice(['sales', 'rent', 'travel'])},{rng.randint(-5000, 9000) / 100:.2f},"
                     f"{rng.choice(['USD', 'EUR', 'GBP', 'JPY'])},{ts}")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_summaries_match_oracle_in_any_order(seed):
    text = make_text(seed)
    entries = parse_entries(text)
    # interleave override use with default use, in an order that exposes leaks
    fx.load_rates({"EUR": 3.0, "GBP": 0.5})
    for fy in (2023, 2024, 2025, 2026):
        want = {}
        wantq = {}
        for acct, usd, f, q in oracle(text, DEFAULT):
            if f == fy:
                want[acct] = round(want.get(acct, 0.0) + usd, 2)
            if f == fy:
                wantq.setdefault(q, {})
                wantq[q][acct] = round(wantq[q].get(acct, 0.0) + usd, 2)
        assert summary.fy_totals(entries, fy) == dict(sorted(want.items())), fy
        for q in (1, 2, 3, 4):
            assert summary.quarter_totals(entries, fy, q) == dict(sorted(wantq.get(q, {}).items())), (fy, q)
        assert summary.grand_total(entries, fy) == round(sum(want.values()), 2)
    rates = fx.load_rates({"EUR": 2.0})
    mod = dict(DEFAULT, EUR=2.0)
    want = {}
    for acct, usd, f, q in oracle(text, mod):
        if f == 2025:
            want[acct] = round(want.get(acct, 0.0) + usd, 2)
    assert summary.fy_totals(entries, 2025, rates) == dict(sorted(want.items()))


def test_periods_unchanged():
    assert fiscal_year(date(2025, 3, 31)) == 2024 and fiscal_year(date(2025, 4, 1)) == 2025
    assert fiscal_quarter(date(2025, 4, 1)) == 1 and fiscal_quarter(date(2026, 3, 31)) == 4
