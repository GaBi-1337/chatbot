from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Entry:
    id: int
    account: str
    amount: float          # in `currency`
    currency: str
    booked: date           # booking date (see ingest.parse_timestamp)
