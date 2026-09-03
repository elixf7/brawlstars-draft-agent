"""When retraining happens. Arithmetic, so the assertions are exact."""
from datetime import date

import pytest

from bsdraft.schedule import (
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
