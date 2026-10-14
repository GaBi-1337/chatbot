from .catalog import get_price


class Cart:
    def __init__(self):
        self._items = {}

    def add(self, sku, qty=1):
        if not isinstance(qty, int) or qty <= 0:
            raise ValueError(f"bad quantity: {qty!r}")
        get_price(sku)  # validates the sku
        self._items[sku] = self._items.get(sku, 0) + qty

    def items(self):
        """Sorted list of (sku, qty)."""
        return sorted(self._items.items())

    def line_total(self, sku):
        return get_price(sku) * self._items[sku]

    def subtotal(self):
        """Sum of all line totals, before any discount (cents)."""
        return sum(self.line_total(sku) for sku in self._items)
