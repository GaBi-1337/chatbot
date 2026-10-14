from schedule import merge_intervals, total_covered


def test_empty():
    assert merge_intervals([]) == []


def test_disjoint():
    assert merge_intervals([(5, 6), (1, 2)]) == [(1, 2), (5, 6)]


def test_overlap():
    assert merge_intervals([(1, 4), (3, 6)]) == [(1, 6)]


def test_touching():
    assert merge_intervals([(1, 3), (3, 5)]) == [(1, 5)]


def test_nested():
    assert merge_intervals([(1, 10), (2, 3), (4, 5)]) == [(1, 10)]


def test_input_not_modified():
    data = [(3, 4), (1, 2)]
    merge_intervals(data)
    assert data == [(3, 4), (1, 2)]


def test_total_covered():
    assert total_covered([(0, 5), (3, 8), (20, 22)]) == 10


def test_total_covered_nested():
    assert total_covered([(0, 100), (10, 20), (30, 40)]) == 100
