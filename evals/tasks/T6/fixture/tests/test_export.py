from pathlib import Path

from ledger.export import totals_csv
from ledger.ingest import load_entries

DATA = Path(__file__).parent / "data" / "entries.csv"


def test_totals_csv_shape():
    text = totals_csv(load_entries(DATA), 2024)
    assert text.splitlines()[0] == "account,usd_total"
    assert text.splitlines()[-1].startswith("TOTAL,")


def test_totals_csv_empty_year():
    assert totals_csv(load_entries(DATA), 1999) == "account,usd_total\nTOTAL,0.00\n"
