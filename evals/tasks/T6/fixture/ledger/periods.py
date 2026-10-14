"""Fiscal calendar: the fiscal year starts on 1 April and is named after the calendar year it starts in
(FY2025 = 2025-04-01 .. 2026-03-31)."""
from functools import lru_cache


@lru_cache(maxsize=None)
def fiscal_year(d):
    return d.year if d.month >= 4 else d.year - 1


@lru_cache(maxsize=None)
def fiscal_quarter(d):
    """1..4; Q1 = Apr-Jun."""
    return ((d.month - 4) % 12) // 3 + 1
