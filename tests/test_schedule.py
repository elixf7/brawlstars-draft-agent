"""When retraining happens. Arithmetic, so the assertions are exact."""
from datetime import date, timedelta

import pytest

from bsdraft.schedule import (
    is_opening_run_day,
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


class TestOpeningRun:
    """The one extra run a new season gets, and what stops it becoming seven.

    Season 54 began Thursday 2026-09-17, season 55 on Thursday 2026-10-15.
    """

    def test_it_is_the_monday_after_the_reset(self):
        assert is_opening_run_day(date(2026, 9, 21))
        assert date(2026, 9, 21).weekday() == 0

    def test_no_other_day_of_the_first_week_runs(self):
        for day in range(8):
            when = date(2026, 9, 17) + timedelta(days=day)
            assert is_opening_run_day(when) == (when == date(2026, 9, 21)), when

    def test_it_never_lands_on_a_self_play_week(self):
        """A run landing on a self-play week would distil a policy on top of it."""
        day = date(2026, 9, 17)
        while day < date(2026, 10, 15):
            assert not (is_opening_run_day(day) and should_run_selfplay(day)), day
            day += timedelta(days=1)

    def test_the_next_reset_runs_again_with_no_edit(self):
        assert is_opening_run_day(date(2026, 10, 19))
        assert date(2026, 10, 19).weekday() == 0

    def test_it_fires_once_a_season_across_a_year(self):
        """Exactly one run per season, always a Monday, for twelve resets."""
        days = [
            date(2026, 9, 17) + timedelta(days=n)
            for n in range((date(2027, 9, 17) - date(2026, 9, 17)).days)
        ]
        hits = [d for d in days if is_opening_run_day(d)]
        assert len(hits) == 12
        assert all(d.weekday() == 0 for d in hits), hits
        assert all((b - a).days >= 28 for a, b in zip(hits, hits[1:]))
