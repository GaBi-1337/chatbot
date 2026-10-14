"""Currency conversion to USD."""
import copy

_DEFAULT = {
    "USD": {"rate": 1.0, "source": "fixed"},
    "EUR": {"rate": 1.10, "source": "ecb-monthly"},
    "GBP": {"rate": 1.30, "source": "boe-monthly"},
    "JPY": {"rate": 0.0070, "source": "boj-monthly"},
}


def load_rates(overrides=None):
    """Rate table currency -> {"rate": USD per unit, "source": str}.

    `overrides` maps currency -> rate and replaces the default rate for this table only.
    """
    rates = copy.deepcopy(_DEFAULT)
    for cur, rate in (overrides or {}).items():
        rates[cur] = rates.get(cur, {"source": "override"})
        rates[cur]["rate"] = rate
    return rates


def convert(amount, currency, rates=None):
    """Convert `amount` of `currency` to USD, rounded to cents."""
    rates = rates if rates is not None else load_rates()
    return round(amount * rates[currency]["rate"], 2)
