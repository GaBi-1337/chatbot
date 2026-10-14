from datetime import date

from ledger.periods import fiscal_quarter, fiscal_year


def test_fiscal_year_boundaries():
    assert fiscal_year(date(2025, 3, 31)) == 2024
    assert fiscal_year(date(2025, 4, 2)) == 2025
    assert fiscal_year(date(2026, 1, 1)) == 2025


def test_fiscal_quarters():
    assert [fiscal_quarter(date(2025, m, 10)) for m in range(4, 13)] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert [fiscal_quarter(date(2026, m, 10)) for m in (1, 2, 3)] == [4, 4, 4]
