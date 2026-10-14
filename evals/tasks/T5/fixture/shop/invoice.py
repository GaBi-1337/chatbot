from dataclasses import dataclass, field

from .catalog import get_name, get_price, is_heavy
from .discounts import bulk_discount, coupon_discount
from .shipping import shipping_cost
from .tax import tax_for
from .util import fmt


@dataclass
class Line:
    sku: str
    name: str
    qty: int
    unit_price: float
    line_total: float
    discount: float


@dataclass
class Invoice:
    region: str
    lines: list = field(default_factory=list)
    subtotal: float = 0.0
    bulk_discount: float = 0.0
    coupon_discount: float = 0.0
    taxable: float = 0.0
    shipping: float = 0.0
    tax: float = 0.0
    credit_applied: float = 0.0
    total: float = 0.0


def build_invoice(cart, region, coupon=None, credit=0.0):
    inv = Invoice(region=region)
    heavy_units = 0
    for sku, qty in cart.items():
        unit = get_price(sku)
        total = round(unit * qty, 2)
        inv.lines.append(Line(sku, get_name(sku), qty, unit, total, bulk_discount(total, qty)))
        if is_heavy(sku):
            heavy_units += qty
    inv.subtotal = round(sum(l.line_total for l in inv.lines), 2)
    inv.bulk_discount = round(sum(l.discount for l in inv.lines), 2)
    after_bulk = round(inv.subtotal - inv.bulk_discount, 2)
    inv.coupon_discount = coupon_discount(after_bulk, coupon) if coupon else 0.0
    inv.taxable = round(after_bulk - inv.coupon_discount, 2)
    inv.shipping = shipping_cost(inv.taxable, heavy_units)
    inv.tax = tax_for(inv.taxable, region)
    gross = round(inv.taxable + inv.shipping + inv.tax, 2)
    inv.credit_applied = min(credit, gross)
    inv.total = round(gross - inv.credit_applied, 2)
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
