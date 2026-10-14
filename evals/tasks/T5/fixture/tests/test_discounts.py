import pytest

from shop.discounts import bulk_discount, coupon_discount, percent_of


def test_percent_of():
    assert percent_of(40.0, 10) == 4.0


def test_bulk():
    assert bulk_discount(100.0, 10) == 5.0
    assert bulk_discount(100.0, 9) == 0.0


def test_coupons():
    assert coupon_discount(80.0, "SAVE10") == 8.0
    assert coupon_discount(80.0, "FIVEOFF") == 5.0
    assert coupon_discount(3.0, "FIVEOFF") == 3.0


def test_unknown_coupon():
    with pytest.raises(ValueError):
        coupon_discount(10.0, "NOPE")
