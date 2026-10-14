from pathlib import Path

from ledger.fx import load_rates
from ledger.ingest import load_entries
from ledger.summary import fy_totals, grand_total, quarter_totals

DATA = Path(__file__).parent / "data" / "entries.csv"
ENTRIES = load_entries(DATA)


def test_quarter_totals():
    # Q3 FY2025 = Oct-Dec 2025 (entries 7 and 8, both EUR at the default rate)
    assert quarter_totals(ENTRIES, 2025, 3) == {"sales": 88.0, "travel": -132.0}


def test_fy_totals_by_local_booking_date():
    assert fy_totals(ENTRIES, 2025) == {"rent": -90.5, "sales": 983.0, "travel": -207.0}


def test_grand_total():
    assert grand_total(ENTRIES, 2025) == 685.5


def test_quarter_totals_with_rate_override():
    rates = load_rates({"EUR": 2.0})
    assert quarter_totals(ENTRIES, 2025, 3, rates) == {"sales": 160.0, "travel": -240.0}


def test_no_entries_in_year():
    assert fy_totals(ENTRIES, 1999) == {}
