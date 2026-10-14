# region -> sales tax percentage
RATES = {"CA": 7.25, "NY": 8.875, "TX": 6.25, "OR": 0.0}


def tax_for(amount, region):
    """Sales tax on a dollar amount, rounded to cents."""
    try:
        rate = RATES[region]
    except KeyError:
        raise ValueError(f"unknown region: {region}") from None
    return round(amount * rate / 100, 2)
