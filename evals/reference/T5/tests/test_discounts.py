import pytest

from shop.discounts import bulk_discount, coupon_discount, percent_of


def test_percent_of():
    assert percent_of(4000, 10) == 400


def test_bulk():
    assert bulk_discount(10000, 10) == 500
    assert bulk_discount(10000, 9) == 0


def test_coupons():
    assert coupon_discount(8000, "SAVE10") == 800
    assert coupon_discount(8000, "FIVEOFF") == 500
    assert coupon_discount(300, "FIVEOFF") == 300


def test_unknown_coupon():
    with pytest.raises(ValueError):
        coupon_discount(1000, "NOPE")
