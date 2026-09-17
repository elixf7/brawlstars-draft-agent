"""When retraining happens. Arithmetic, so the assertions are exact."""
from datetime import date, timedelta

import pytest

from bsdraft.schedule import (
    in_season_opening,
    season_start,
    should_run_selfplay,
    third_thursday,
    weeks_into_season,
)


@pytest.mark.parametrize("y,m,expected", [
    (2026, 8, date(2026, 8, 20)),
    (2026, 9, date(2026, 9, 17)),
    (2026, 10, date(2026, 10, 15)),
    (2027, 1, date(2027, 1, 21)),
])
def test_third_thursday(y, m, expected):
    assert third_thursday(y, m) == expected
    assert expected.weekday() == 3


def test_a_day_before_the_reset_belongs_to_the_previous_season():
    assert season_start(date(2026, 9, 16)) == date(2026, 8, 20)
    assert season_start(date(2026, 9, 17)) == date(2026, 9, 17)


def test_season_start_crosses_the_new_year():
    assert season_start(date(2027, 1, 5)) == date(2026, 12, 17)


@pytest.mark.parametrize("day,week", [
    (date(2026, 9, 17), 1),   # reset day
    (date(2026, 9, 23), 1),
    (date(2026, 9, 24), 2),
    (date(2026, 10, 1), 3),
    (date(2026, 10, 8), 4),
])
def test_weeks_into_season(day, week):
    assert weeks_into_season(day) == week


def test_selfplay_skips_the_first_week():
    """Season data is thin then, and below ~100k games the model scores worse
    than counting — distilling from it would bake in the weakness."""
    assert not should_run_selfplay(date(2026, 9, 18))
    assert should_run_selfplay(date(2026, 9, 24))     # week 2
    assert not should_run_selfplay(date(2026, 10, 1))  # week 3
    assert should_run_selfplay(date(2026, 10, 8))     # week 4


def test_selfplay_weeks_are_configurable():
    assert should_run_selfplay(date(2026, 10, 1), weeks=(3,))


class TestSeasonOpening:
    """The daily training entry fires year-round; this is what turns it away.

    Season 54 began 2026-09-17, season 55 on 2026-10-15.
    """

    def test_the_whole_first_week_is_opening(self):
        for day in range(7):
            when = date(2026, 9, 17) + timedelta(days=day)
            assert in_season_opening(when), when

    def test_it_stops_before_week_two(self):
        assert not in_season_opening(date(2026, 9, 24))

    def test_the_opening_window_never_overlaps_a_self_play_week(self):
        """A daily run landing on a self-play week would distil one every day."""
        day = date(2026, 9, 17)
        while day < date(2026, 10, 15):
            assert not (in_season_opening(day) and should_run_selfplay(day)), day
            day += timedelta(days=1)

    def test_the_day_before_the_next_reset_is_not_opening(self):
        assert not in_season_opening(date(2026, 10, 14))

    def test_the_next_reset_opens_again_with_no_edit(self):
        assert in_season_opening(date(2026, 10, 15))
