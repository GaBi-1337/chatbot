import pytest

from shop.catalog import UnknownSku, get_name, get_price, is_heavy, skus


def test_price_and_name():
    assert get_price("A100") == 19.99
    assert get_name("A100") == "Widget"


def test_heavy():
    assert is_heavy("C300") and not is_heavy("A100")


def test_unknown():
    with pytest.raises(UnknownSku):
        get_price("ZZZ")


def test_skus_sorted():
    assert skus() == sorted(skus()) and "F600" in skus()
