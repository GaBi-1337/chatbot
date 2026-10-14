import pytest

from ledger.fx import convert, load_rates


def test_default_rates():
    assert convert(100, "EUR") == 110.0
    assert convert(100, "USD") == 100.0


def test_convert_rounds_to_cents():
    assert convert(3000, "JPY") == 21.0
    assert convert(0.333, "GBP") == 0.43


def test_override_rate():
    rates = load_rates({"EUR": 1.2})
    assert convert(100, "EUR", rates) == 120.0
    assert rates["GBP"]["rate"] == 1.30


def test_override_unknown_currency():
    rates = load_rates({"CHF": 1.05})
    assert convert(10, "CHF", rates) == 10.5


def test_unknown_currency():
    with pytest.raises(KeyError):
        convert(1, "XXX")
