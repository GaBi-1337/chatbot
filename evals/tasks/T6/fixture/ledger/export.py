import csv
import io

from .summary import fy_totals


def totals_csv(entries, fy, rates=None):
    """CSV text: account,usd_total, sorted by account, with a trailing TOTAL row."""
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["account", "usd_total"])
    totals = fy_totals(entries, fy, rates)
    for account, amount in totals.items():
        w.writerow([account, f"{amount:.2f}"])
    w.writerow(["TOTAL", f"{round(sum(totals.values()), 2):.2f}"])
    return out.getvalue()
