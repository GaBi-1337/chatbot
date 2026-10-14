import pytest

from shop.cart import Cart


def test_subtotal():
    c = Cart()
    c.add("A100", 2)
    c.add("B200")
    assert c.subtotal() == 4433


def test_add_merges():
    c = Cart()
    c.add("A100")
    c.add("A100", 2)
    assert c.items() == [("A100", 3)]
    assert c.line_total("A100") == 5997


def test_bad_qty():
    with pytest.raises(ValueError):
        Cart().add("A100", 0)
