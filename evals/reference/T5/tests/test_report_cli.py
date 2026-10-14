import subprocess
import sys
from pathlib import Path

from shop.cart import Cart
from shop.invoice import build_invoice
from shop.report import average_order_value, revenue_by_sku, total_revenue

ROOT = Path(__file__).resolve().parent.parent


def _invs():
    a = Cart(); a.add("A100", 2)
    b = Cart(); b.add("E500", 3)
    return [build_invoice(a, "OR"), build_invoice(b, "OR")]


def test_report_numbers():
    invs = _invs()
    assert revenue_by_sku(invs) == {"A100": 3998, "E500": 3750}
    assert total_revenue(invs) == 8746
    assert average_order_value(invs) == 4373


def test_empty_report():
    assert average_order_value([]) == 0


def test_cli_invoice():
    r = subprocess.run([sys.executable, "-m", "shop", "invoice", "examples/order1.json", "--region", "CA"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0
    assert "Total: $169.30" in r.stdout


def test_cli_bad_region():
    r = subprocess.run([sys.executable, "-m", "shop", "invoice", "examples/order1.json", "--region", "XX"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2
