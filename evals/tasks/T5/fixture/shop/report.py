from .util import fmt


def revenue_by_sku(invoices):
    """Merchandise revenue per sku in dollars (line totals minus bulk discounts)."""
    rev = {}
    for inv in invoices:
        for l in inv.lines:
            rev[l.sku] = round(rev.get(l.sku, 0.0) + l.line_total - l.discount, 2)
    return dict(sorted(rev.items()))


def total_revenue(invoices):
    return round(sum(inv.total for inv in invoices), 2)


def average_order_value(invoices):
    if not invoices:
        return 0.0
    return round(total_revenue(invoices) / len(invoices), 2)


def render_report(invoices):
    out = [f"Orders: {len(invoices)}", f"Revenue: {fmt(total_revenue(invoices))}",
           f"Average order: {fmt(average_order_value(invoices))}"]
    for sku, amount in revenue_by_sku(invoices).items():
        out.append(f"  {sku}: {fmt(amount)}")
    return "\n".join(out) + "\n"
