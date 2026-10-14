import pytest

from shop.shipping import shipping_cost
from shop.tax import tax_for


def test_tax():
    assert tax_for(10000, "CA") == 725
    assert tax_for(10000, "OR") == 0


def test_tax_unknown_region():
    with pytest.raises(ValueError):
        tax_for(100, "XX")


def test_shipping_flat_and_free():
    assert shipping_cost(1000) == 499
    assert shipping_cost(5000) == 0


def test_shipping_heavy():
    assert shipping_cost(6000, heavy_units=2) == 250
