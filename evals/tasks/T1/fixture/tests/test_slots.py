from schedule import free_slots


def test_no_busy():
    assert free_slots([], 9, 17) == [(9, 17)]


def test_simple_gap():
    assert free_slots([(10, 11), (13, 14)], 9, 17) == [(9, 10), (11, 13), (14, 17)]


def test_busy_outside_day_ignored():
    assert free_slots([(1, 2), (20, 22)], 9, 17) == [(9, 17)]


def test_long_meeting_hides_short_ones():
    assert free_slots([(9, 15), (10, 11), (12, 13)], 9, 17) == [(15, 17)]


def test_fully_booked():
    assert free_slots([(8, 18)], 9, 17) == []
