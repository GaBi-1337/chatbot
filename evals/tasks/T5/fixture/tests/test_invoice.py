from shop.cart import Cart
from shop.invoice import build_invoice, render_invoice


def _cart():
    c = Cart()
    c.add("A100", 2)
    c.add("B200", 12)
    return c


def test_build_invoice_totals():
    inv = build_invoice(_cart(), "CA")
    assert inv.subtotal == 92.18
    assert inv.bulk_discount == 2.61
    assert inv.shipping == 0.0
    assert inv.tax == 6.49
    assert inv.total == 96.06


def test_credit_is_capped():
    c = Cart()
    c.add("D400")
    inv = build_invoice(c, "OR", credit=100.0)
    assert inv.credit_applied == 5.28
    assert inv.total == 0.0


def test_render():
    text = render_invoice(build_invoice(_cart(), "CA", coupon="SAVE10"))
    assert "Widget (A100) x2 @ $19.99 = $39.98" in text
    assert text.endswith("\n") and "Coupon: -$8.96" in text
