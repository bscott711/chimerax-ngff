"""schedule: which time points to keep ready, and the direction of travel."""

import pytest
from cxngff import schedule


def test_offset_goes_the_short_way_round():
    assert schedule.offset(3, 1, 10) == 2
    assert schedule.offset(1, 3, 10) == -2
    assert schedule.offset(9, 1, 10) == -2  # wraps backwards
    assert schedule.offset(1, 9, 10) == 2  # wraps forwards


def test_ahead_is_needed_before_behind():
    nt, cur = 100, 50
    assert schedule.score(cur, cur, 1, nt) == 0
    assert schedule.score(51, cur, 1, nt) < schedule.score(49, cur, 1, nt)
    # Travelling backwards the roles swap.
    assert schedule.score(49, cur, -1, nt) < schedule.score(51, cur, -1, nt)


def test_wanted_starts_with_current_then_ahead_two_to_one():
    w = schedule.wanted(50, 1, 100, 9)
    assert w[0] == 50
    assert len(w) == 9 and len(set(w)) == 9
    ahead = sorted(t for t in w if t > 50)
    behind = sorted(t for t in w if t < 50)
    assert ahead == [51, 52, 53, 54, 55, 56] and behind == [48, 49]


def test_wanted_wraps_round_the_series():
    w = schedule.wanted(98, 1, 100, 6)
    assert w[:3] == [98, 99, 0]
    assert set(w) == {98, 99, 0, 1, 97, 2} or set(w) == {98, 99, 0, 1, 2, 97}


def test_wanted_is_capped_by_series_and_capacity():
    assert sorted(schedule.wanted(2, 1, 5, 100)) == [0, 1, 2, 3, 4]
    assert schedule.wanted(2, 1, 5, 0) == []
    assert schedule.wanted(0, 1, 0, 3) == []
    assert schedule.wanted(3, 1, 5, 1) == [3]


def test_direction_follows_small_steps_and_ignores_jumps():
    d = schedule.Direction(100)
    assert d.update(10) == 1
    assert d.update(9) == -1
    assert d.update(8) == -1
    assert d.update(70) == -1  # a drag of the slider: keep the direction
    assert d.update(71) == 1


def test_playing_needs_a_streak():
    d = schedule.Direction(100)
    for t in (1, 2, 3):
        d.update(t)
    assert not d.playing  # two moves
    d.update(4)
    assert d.playing
    d.update(50)  # jump
    assert not d.playing
    d.update(51)
    d.update(50)  # reverses: streak restarts
    assert not d.playing


@pytest.mark.parametrize("nt", [1, 2, 3])
def test_tiny_series(nt):
    d = schedule.Direction(nt)
    for t in range(nt):
        d.update(t)
    assert sorted(schedule.wanted(0, d.direction, nt, 10)) == list(range(nt))
