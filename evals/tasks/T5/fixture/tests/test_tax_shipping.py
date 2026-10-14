import pytest

from shop.shipping import shipping_cost
from shop.tax import tax_for


def test_tax():
    assert tax_for(100.0, "CA") == 7.25
    assert tax_for(100.0, "OR") == 0.0


def test_tax_unknown_region():
    with pytest.raises(ValueError):
        tax_for(1.0, "XX")


def test_shipping_flat_and_free():
    assert shipping_cost(10.0) == 4.99
    assert shipping_cost(50.0) == 0.0


def test_shipping_heavy():
    assert shipping_cost(60.0, heavy_units=2) == 2.5
