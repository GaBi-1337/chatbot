"""Small helpers shared by the rest of the package."""


def fmt(amount):
    """Format dollars as $1,234.50 (negative as -$3.20)."""
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def parse_amount(text):
    """'$1,234.5' -> 1234.5"""
    return float(text.replace("$", "").replace(",", "").strip())
