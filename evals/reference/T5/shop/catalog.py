"""Product catalog, loaded from products.json (prices there are dollars). Prices here are integer cents."""
import json
from pathlib import Path

from .money import parse_dollars

_DATA = json.loads(Path(__file__).with_name("products.json").read_text())


class UnknownSku(KeyError):
    pass


def _product(sku):
    try:
        return _DATA[sku]
    except KeyError:
        raise UnknownSku(sku) from None


def get_price(sku):
    """Unit price in cents."""
    return parse_dollars(repr(_product(sku)["price"]))


def get_name(sku):
    return _product(sku)["name"]


def is_heavy(sku):
    return bool(_product(sku)["heavy"])


def skus():
    return sorted(_DATA)
