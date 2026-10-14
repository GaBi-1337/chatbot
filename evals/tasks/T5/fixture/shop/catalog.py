"""Product catalog, loaded from products.json. Prices are dollars."""
import json
from pathlib import Path

_DATA = json.loads(Path(__file__).with_name("products.json").read_text())


class UnknownSku(KeyError):
    pass


def _product(sku):
    try:
        return _DATA[sku]
    except KeyError:
        raise UnknownSku(sku) from None


def get_price(sku):
    """Unit price in dollars."""
    return _product(sku)["price"]


def get_name(sku):
    return _product(sku)["name"]


def is_heavy(sku):
    return bool(_product(sku)["heavy"])


def skus():
    return sorted(_DATA)
