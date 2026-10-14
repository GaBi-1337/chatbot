import pytest

from shop.money import format_cents, parse_dollars


def test_parse():
    assert parse_dollars("12.50") == 1250
    assert parse_dollars("$1,234.5") == 123450
    assert parse_dollars("7") == 700


def test_parse_rejects_junk():
    for bad in ("abc", "1.005", ""):
        with pytest.raises(ValueError):
            parse_dollars(bad)


def test_format():
    assert format_cents(1234550) == "$12,345.50"
    assert format_cents(-320) == "-$3.20"
    assert format_cents(5) == "$0.05"
