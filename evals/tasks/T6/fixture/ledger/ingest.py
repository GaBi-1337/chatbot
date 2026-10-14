"""Reading postings from CSV text: id,account,amount,currency,timestamp"""
from datetime import datetime, timezone
from pathlib import Path

from .models import Entry


def parse_timestamp(text):
    """Booking date of a posting: the calendar date shown in the ISO timestamp.

    Postings carry the local time of the posting office, so a posting stamped
    2025-04-01T00:30:00+05:30 is booked on 2025-04-01.
    """
    dt = datetime.fromisoformat(text.strip())
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.date()


def parse_line(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 5:
        raise ValueError(f"expected 5 fields, got {len(parts)}")
    eid, account, amount, currency, ts = parts
    return Entry(int(eid), account, float(amount), currency.upper(), parse_timestamp(ts))


def parse_entries(text):
    entries = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("id,"):
            continue
        try:
            entries.append(parse_line(line))
        except ValueError as e:
            raise ValueError(f"line {n}: {e}") from None
    return entries


def load_entries(path):
    return parse_entries(Path(path).read_text())
