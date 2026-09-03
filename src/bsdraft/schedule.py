"""When to retrain.

Ranked seasons start on the third Thursday of each month, so "week 2 of the
season" is arithmetic rather than something to look up. The companion ETL
pipeline computes the same boundaries; this is the consumer side of it.

The cadence is asymmetric on purpose. The win-probability model trains in
seconds and keeps improving as data accumulates — measured, there is no plateau
at a million games — so it is retrained weekly. Self-play is slower and plateaus
after one iteration, so it runs twice a season: once when there is enough new
data to be worth distilling, and again once the meta has settled.
"""
from __future__ import annotations

import calendar
from datetime import date

_THURSDAY = 3

#: Weeks into the season on which self-play is worth running.
SELFPLAY_WEEKS = (2, 4)


def third_thursday(year: int, month: int) -> date:
    days = [
        d for d in calendar.Calendar().itermonthdates(year, month)
        if d.month == month and d.weekday() == _THURSDAY
    ]
    return days[2]


def season_start(on: date) -> date:
    """The start of the season containing `on`."""
    this_month = third_thursday(on.year, on.month)
    if on >= this_month:
        return this_month
    prev_year, prev_month = (on.year - 1, 12) if on.month == 1 else (on.year, on.month - 1)
    return third_thursday(prev_year, prev_month)


def weeks_into_season(on: date | None = None) -> int:
    """1 during the season's first seven days, 2 during the next, and so on."""
    on = on or date.today()
    return (on - season_start(on)).days // 7 + 1


def should_run_selfplay(on: date | None = None, weeks: tuple[int, ...] = SELFPLAY_WEEKS) -> bool:
    """Self-play is worth running only in certain weeks of the season.

    Not week 1: the season's data is still thin then, and below roughly 100,000
    games the model scores worse than counting character-and-map win rates.
    Distilling a policy from an evaluator in that state bakes in the weakness.
    """
    return weeks_into_season(on) in weeks


__all__ = ["SELFPLAY_WEEKS", "season_start", "should_run_selfplay",
           "third_thursday", "weeks_into_season"]
