"""Interval helpers. An interval is a (start, end) tuple with start <= end."""


def merge_intervals(intervals):
    """Merge overlapping or touching intervals.

    (1, 3) and (3, 5) touch and merge into (1, 5). Input order is arbitrary and
    the input is not modified. Returns a sorted list of tuples.
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [tuple(m) for m in merged]


def total_covered(intervals):
    """Total length covered by the union of the intervals."""
    return sum(e - s for s, e in merge_intervals(intervals))
