from dataclasses import dataclass, field

from .catalog import get_name, get_price, is_heavy
from .discounts import bulk_discount, coupon_discount
from .shipping import shipping_cost
from .tax import tax_for
from .money import format_cents as fmt


@dataclass
class Line:
    sku: str
    name: str
    qty: int
    unit_price: int
    line_total: int
    discount: int


@dataclass
class Invoice:
    region: str
    lines: list = field(default_factory=list)
    subtotal: int = 0
    bulk_discount: int = 0
    coupon_discount: int = 0
    taxable: int = 0
    shipping: int = 0
    tax: int = 0
    credit_applied: int = 0
    total: int = 0


def build_invoice(cart, region, coupon=None, credit=0):
    inv = Invoice(region=region)
    heavy_units = 0
    for sku, qty in cart.items():
        unit = get_price(sku)
        total = unit * qty
        inv.lines.append(Line(sku, get_name(sku), qty, unit, total, bulk_discount(total, qty)))
        if is_heavy(sku):
            heavy_units += qty
    inv.subtotal = sum(l.line_total for l in inv.lines)
    inv.bulk_discount = sum(l.discount for l in inv.lines)
    after_bulk = inv.subtotal - inv.bulk_discount
    inv.coupon_discount = coupon_discount(after_bulk, coupon) if coupon else 0
    inv.taxable = after_bulk - inv.coupon_discount
    inv.shipping = shipping_cost(inv.taxable, heavy_units)
    inv.tax = tax_for(inv.taxable, region)
    gross = inv.taxable + inv.shipping + inv.tax
    inv.credit_applied = min(credit, gross)
    inv.total = gross - inv.credit_applied
    return inv


def render_invoice(inv):
    out = []
    for l in inv.lines:
        out.append(f"{l.name} ({l.sku}) x{l.qty} @ {fmt(l.unit_price)} = {fmt(l.line_total)}")
    out.append(f"Subtotal: {fmt(inv.subtotal)}")
    if inv.bulk_discount:
        out.append(f"Bulk discount: -{fmt(inv.bulk_discount)}")
    if inv.coupon_discount:
        out.append(f"Coupon: -{fmt(inv.coupon_discount)}")
    out.append(f"Shipping: {fmt(inv.shipping)}")
    out.append(f"Tax ({inv.region}): {fmt(inv.tax)}")
    if inv.credit_applied:
        out.append(f"Store credit: -{fmt(inv.credit_applied)}")
    out.append(f"Total: {fmt(inv.total)}")
    return "\n".join(out) + "\n"
