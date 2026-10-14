import argparse
import json
import sys

from .cart import Cart
from .invoice import build_invoice, render_invoice
from .report import render_report
from .money import parse_dollars


def _cart(order):
    cart = Cart()
    for item in order["items"]:
        cart.add(item["sku"], item.get("qty", 1))
    return cart


def main(argv=None):
    p = argparse.ArgumentParser(prog="shop")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("invoice", help="price one order")
    pi.add_argument("order", help="JSON file: {\"items\": [{\"sku\": ..., \"qty\": ...}]}")
    pi.add_argument("--region", required=True)
    pi.add_argument("--coupon")
    pi.add_argument("--credit", default="0", help="store credit in dollars, e.g. 5.50")
    pr = sub.add_parser("report", help="revenue report over many orders")
    pr.add_argument("orders", help="JSON file: list of {\"items\": [...], \"region\": ..., \"coupon\": ...}")
    args = p.parse_args(argv)
    try:
        if args.cmd == "invoice":
            order = json.load(open(args.order))
            inv = build_invoice(_cart(order), args.region, args.coupon, parse_dollars(args.credit))
            sys.stdout.write(render_invoice(inv))
        else:
            orders = json.load(open(args.orders))
            invs = [build_invoice(_cart(o), o["region"], o.get("coupon"), parse_dollars(o.get("credit", "0")))
                    for o in orders]
            sys.stdout.write(render_report(invs))
    except (ValueError, KeyError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0
