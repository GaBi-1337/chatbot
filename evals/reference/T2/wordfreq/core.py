import re
from collections import Counter

_WORD = re.compile(r"[a-z']+")


def tokenize(text):
    """Lowercase the text and return its words (letters and apostrophes only)."""
    return _WORD.findall(text.lower())


def count_words(text, exclude=None):
    """Return a Counter of word frequencies, skipping words in `exclude`."""
    exclude = exclude or ()
    return Counter(w for w in tokenize(text) if w not in exclude)


def load_exclusions(path):
    """Read a set of lowercase words from a file (one per line, '#' comments, blanks ignored)."""
    words = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip().lower()
            if line and not line.startswith("#"):
                words.add(line)
    return words


def top_words(counter, n=None):
    """Return (word, count) pairs sorted by count descending, then word ascending."""
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked if n is None else ranked[:n]
