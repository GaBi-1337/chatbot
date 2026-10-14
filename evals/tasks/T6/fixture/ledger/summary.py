from .fx import convert
from .periods import fiscal_quarter, fiscal_year


def _totals(entries, keep, rates):
    totals = {}
    for e in entries:
        if not keep(e):
            continue
        totals[e.account] = round(totals.get(e.account, 0.0) + convert(e.amount, e.currency, rates), 2)
    return dict(sorted(totals.items()))


def fy_totals(entries, fy, rates=None):
    """USD total per account for a fiscal year."""
    return _totals(entries, lambda e: fiscal_year(e.booked) == fy, rates)


def quarter_totals(entries, fy, quarter, rates=None):
    """USD total per account for one fiscal quarter."""
    return _totals(entries, lambda e: fiscal_year(e.booked) == fy and fiscal_quarter(e.booked) == quarter, rates)


def grand_total(entries, fy, rates=None):
    return round(sum(fy_totals(entries, fy, rates).values()), 2)
