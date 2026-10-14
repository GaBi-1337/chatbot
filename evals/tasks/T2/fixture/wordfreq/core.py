import re
from collections import Counter

_WORD = re.compile(r"[a-z']+")


def tokenize(text):
    """Lowercase the text and return its words (letters and apostrophes only)."""
    return _WORD.findall(text.lower())


def count_words(text):
    """Return a Counter of word frequencies."""
    return Counter(tokenize(text))


def top_words(counter, n=None):
    """Return (word, count) pairs sorted by count descending, then word ascending."""
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked if n is None else ranked[:n]
