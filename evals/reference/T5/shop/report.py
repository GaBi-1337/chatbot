from .money import format_cents as fmt


def revenue_by_sku(invoices):
    """Merchandise revenue per sku in cents (line totals minus bulk discounts)."""
    rev = {}
    for inv in invoices:
        for l in inv.lines:
            rev[l.sku] = rev.get(l.sku, 0) + l.line_total - l.discount
    return dict(sorted(rev.items()))


def total_revenue(invoices):
    return sum(inv.total for inv in invoices)


def average_order_value(invoices):
    if not invoices:
        return 0
    return (2 * total_revenue(invoices) + len(invoices)) // (2 * len(invoices))


def render_report(invoices):
    out = [f"Orders: {len(invoices)}", f"Revenue: {fmt(total_revenue(invoices))}",
           f"Average order: {fmt(average_order_value(invoices))}"]
    for sku, amount in revenue_by_sku(invoices).items():
        out.append(f"  {sku}: {fmt(amount)}")
    return "\n".join(out) + "\n"
