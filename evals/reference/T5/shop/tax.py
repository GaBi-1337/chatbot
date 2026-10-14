from .money import percent_of_cents

# region -> sales tax percentage
RATES = {"CA": 7.25, "NY": 8.875, "TX": 6.25, "OR": 0.0}


def tax_for(amount, region):
    """Sales tax on an amount in cents, rounded half up to a whole cent."""
    try:
        rate = RATES[region]
    except KeyError:
        raise ValueError(f"unknown region: {region}") from None
    return percent_of_cents(amount, rate)
