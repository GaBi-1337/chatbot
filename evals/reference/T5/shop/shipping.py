FREE_THRESHOLD = 5000       # cents; orders at or above this ship free
FLAT_RATE = 499             # cents
HEAVY_SURCHARGE = 125       # cents per heavy unit, always charged


def shipping_cost(subtotal, heavy_units=0):
    base = 0 if subtotal >= FREE_THRESHOLD else FLAT_RATE
    return base + heavy_units * HEAVY_SURCHARGE
