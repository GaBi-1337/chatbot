from datetime import date

import pytest

from ledger.ingest import parse_entries, parse_line, parse_timestamp


def test_parse_line():
    e = parse_line("7, sales , 12.50, eur, 2025-10-01T00:00:00")
    assert (e.id, e.account, e.amount, e.currency, e.booked) == (7, "sales", 12.5, "EUR", date(2025, 10, 1))


def test_naive_timestamp():
    assert parse_timestamp("2025-03-31T23:59:59") == date(2025, 3, 31)


def test_parse_entries_skips_comments_and_header():
    text = "id,account,amount,currency,timestamp\n# note\n\n1,a,1,USD,2025-01-01T00:00:00\n"
    assert [e.id for e in parse_entries(text)] == [1]


def test_bad_line_reports_line_number():
    with pytest.raises(ValueError, match="line 2"):
        parse_entries("1,a,1,USD,2025-01-01T00:00:00\n2,b,oops,USD,2025-01-01T00:00:00\n")
