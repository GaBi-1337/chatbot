BULK_MIN_QTY = 10
BULK_PERCENT = 5
# code -> (kind, value): percent value is a percentage, flat value is dollars
COUPONS = {
    "SAVE10": ("percent", 10),
    "HALF": ("percent", 50),
    "FIVEOFF": ("flat", 5.00),
}


def percent_of(amount, pct):
    """pct percent of a dollar amount, rounded to cents."""
    return round(amount * pct / 100, 2)


def bulk_discount(line_total, qty):
    if qty >= BULK_MIN_QTY:
        return percent_of(line_total, BULK_PERCENT)
    return 0.0


def coupon_discount(subtotal, code):
    try:
        kind, value = COUPONS[code]
    except KeyError:
        raise ValueError(f"unknown coupon: {code}") from None
    amount = percent_of(subtotal, value) if kind == "percent" else value
    return min(amount, subtotal)
