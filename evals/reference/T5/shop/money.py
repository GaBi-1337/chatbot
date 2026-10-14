"""Money helpers. All amounts in the package are integer cents."""
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation


def parse_dollars(text):
    """'12.50' -> 1250, '$1,234.5' -> 123450. At most two decimal places."""
    s = text.replace("$", "").replace(",", "").strip()
    try:
        d = Decimal(s)
    except InvalidOperation:
        raise ValueError(f"not a dollar amount: {text!r}") from None
    if not d.is_finite():
        raise ValueError(f"not a dollar amount: {text!r}")
    cents = d * 100
    if cents != cents.to_integral_value():
        raise ValueError(f"more than two decimal places: {text!r}")
    return int(cents)


def format_cents(cents):
    """1234550 -> '$12,345.50'; -320 -> '-$3.20'."""
    sign = "-" if cents < 0 else ""
    c = abs(cents)
    return f"{sign}${c // 100:,}.{c % 100:02d}"


def percent_of_cents(cents, pct):
    """pct percent of an amount in cents, rounded half up to a whole cent, without floats."""
    return int((Decimal(cents) * Decimal(str(pct)) / 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))
