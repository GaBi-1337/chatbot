from wordfreq.core import count_words, top_words, tokenize


def test_tokenize():
    assert tokenize("Hello, World! Don't") == ["hello", "world", "don't"]


def test_count_and_rank():
    c = count_words("b a b c a b")
    assert top_words(c) == [("b", 3), ("a", 2), ("c", 1)]
    assert top_words(c, 1) == [("b", 3)]
