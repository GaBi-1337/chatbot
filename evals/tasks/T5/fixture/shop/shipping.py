FREE_THRESHOLD = 50.00      # dollars; orders at or above this ship free
FLAT_RATE = 4.99            # dollars
HEAVY_SURCHARGE = 1.25      # dollars per heavy unit, always charged


def shipping_cost(subtotal, heavy_units=0):
    base = 0.0 if subtotal >= FREE_THRESHOLD else FLAT_RATE
    return round(base + heavy_units * HEAVY_SURCHARGE, 2)
