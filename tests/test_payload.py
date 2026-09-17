"""The payload's per-character grid is positional, which is the whole reason
it is affordable — and the whole reason it can break quietly. The page reads a
cell by arithmetic on the map and band indices, so if the export ever writes
them in a different order the numbers stay plausible and stop being true.
These tests pin the layout the page assumes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bsdraft.payload.export import (
    ELO_STEPS_PER_POINT,
    N_SKILL_BANDS,
    SKILL_BAND_LABELS,
    SKILL_BIN,
    SKILL_LO,
    TEAM1_BRAWLER_COLS,
    TEAM2_BRAWLER_COLS,
    character_stats,
    rating_distribution,
    season_maps,
    season_stats,
)

MAPS = ["Alpha Base", "Beta Ridge", "Gamma Gulch"]
MODES = {"Alpha Base": "brawlBall", "Beta Ridge": "brawlBall", "Gamma Gulch": "heist"}


def a_season(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Games spread over every map and the whole skill range."""
    rng = np.random.default_rng(seed)
    names = ["RICO", "COLT", "BULL", "PIPER", "SHELLY", "MORTIS"]
    mp = rng.choice(MAPS, n)
    rows = {
        "map": mp,
        "mode": [MODES[m] for m in mp],
        "skill_ns": rng.normal(0, 1, n),
        "team1_wins": rng.integers(0, 2, n),
        "battle_time": ["20260901T120000.000Z"] * n,
    }
    for i, c in enumerate(TEAM1_BRAWLER_COLS + TEAM2_BRAWLER_COLS):
        rows[c] = [names[(j + i) % len(names)] for j in range(n)]
    return pd.DataFrame(rows)


@pytest.fixture
def season():
    return a_season()


def test_grid_is_maps_by_bands_by_two(season):
    stats = character_stats(season, min_games=1)
    width = len(season_maps(season)) * N_SKILL_BANDS * 2
    assert all(len(c["grid"]) == width for c in stats)


def test_the_map_order_the_grid_uses_is_the_one_the_page_is_given(season):
    """The page indexes the grid with D.season.maps. If these two ever
    disagree, every cell is read off the wrong map."""
    assert season_stats(season, "s53", "ds")["maps"] == season_maps(season)


def test_band_labels_and_band_count_agree(season):
    assert len(season_stats(season, "s53", "ds")["skill_bands"]) == N_SKILL_BANDS
    assert list(SKILL_BAND_LABELS) == season_stats(season, "s53", "ds")["skill_bands"]


def test_grid_totals_match_the_headline_figures(season):
    """Summing every cell must reproduce the season-wide numbers, or the
    filtered views and the unfiltered one tell different stories."""
    for c in character_stats(season, min_games=1):
        g = c["grid"]
        assert sum(g[0::2]) == c["games"]
        assert sum(g[1::2]) == pytest.approx(c["win_rate"] * c["games"], abs=0.5)


def test_a_cell_holds_the_games_from_that_map_and_that_band(season):
    """Read one cell by the page's own arithmetic and check it against the
    games it claims to describe."""
    maps = season_maps(season)
    ranks = season["skill_ns"].rank(pct=True, method="average")
    band = np.minimum((ranks * N_SKILL_BANDS).astype(int), N_SKILL_BANDS - 1)
    stats = {c["name"]: c for c in character_stats(season, min_games=1)}

    name, mi, bi = "RICO", 1, 3
    at = (mi * N_SKILL_BANDS + bi) * 2
    got_n, got_w = stats[name]["grid"][at], stats[name]["grid"][at + 1]

    sel = (season["map"] == maps[mi]) & (band == bi)
    want_n = want_w = 0
    for cols, won in ((TEAM1_BRAWLER_COLS, season["team1_wins"]),
                      (TEAM2_BRAWLER_COLS, 1 - season["team1_wins"])):
        for col in cols:
            hit = sel & (season[col] == name)
            want_n += int(hit.sum())
            want_w += int(won[hit].sum())
    assert (got_n, got_w) == (want_n, want_w)


def test_bands_split_the_season_into_equal_slices(season):
    """The page turns a slider position into a band through the percentile it
    represents, which only lines up if the bands are equal slices."""
    stats = character_stats(season, min_games=1)
    per_band = [0] * N_SKILL_BANDS
    for c in stats:
        for mi in range(len(season_maps(season))):
            for b in range(N_SKILL_BANDS):
                per_band[b] += c["grid"][(mi * N_SKILL_BANDS + b) * 2]
    total = sum(per_band)
    assert all(n / total == pytest.approx(1 / N_SKILL_BANDS, abs=0.03)
               for n in per_band)


def test_a_character_confined_to_one_map_fills_only_that_column():
    """Cells are addressed by map index. A character who only ever played one
    map must be absent from every other map's cells — if the index arithmetic
    is off, their games show up under the wrong map instead."""
    df = a_season(600)
    # GALE exists on exactly one map, in exactly one slot.
    df[TEAM1_BRAWLER_COLS[0]] = "RICO"
    df.loc[df["map"] == MAPS[2], TEAM1_BRAWLER_COLS[0]] = "GALE"
    for c in TEAM1_BRAWLER_COLS[1:] + TEAM2_BRAWLER_COLS:
        df[c] = "COLT"

    maps = season_maps(df)
    stats = {c["name"]: c for c in character_stats(df, min_games=1)}
    g = stats["GALE"]["grid"]
    home = maps.index(MAPS[2])
    played = {mi: sum(g[(mi * N_SKILL_BANDS + b) * 2] for b in range(N_SKILL_BANDS))
              for mi in range(len(maps))}

    assert played[home] == int((df["map"] == MAPS[2]).sum())
    assert all(n == 0 for mi, n in played.items() if mi != home)


# ----------------------------------------------------- rating distribution
def a_season_with_ratings(n: int = 900, seed: int = 3) -> pd.DataFrame:
    """As above, with an `id` per set and avg_elo on the grid it really uses:
    the mean of six whole numbers, so always a sixth of a point."""
    df = a_season(n, seed)
    rng = np.random.default_rng(seed)
    df["id"] = range(len(df))
    df["avg_elo"] = rng.integers(72, 132, len(df)) / ELO_STEPS_PER_POINT
    return df


def test_one_bar_per_value_the_average_can_actually_take():
    """A lobby rating is a mean of six integers, so it only lands on sixths.
    Binning on that grid is the finest honest resolution and leaves no bin
    straddling two levels."""
    df = a_season_with_ratings()
    h = rating_distribution(df)["elo"]
    assert h["width"] == pytest.approx(1 / ELO_STEPS_PER_POINT)
    per_bin = [sum(col) for col in zip(*h["counts"], strict=True)]
    assert sum(per_bin) == len(df)
    # Every distinct rating in the data occupies its own bin.
    assert sum(1 for n in per_bin if n) == df["avg_elo"].nunique()


def test_every_match_is_counted_once():
    df = a_season_with_ratings()
    r = rating_distribution(df)
    assert sum(sum(x) for x in r["elo"]["counts"]) == len(df)
    assert sum(sum(x) for x in r["skill"]["counts"]) == len(df)


def test_a_long_set_is_not_counted_once_per_game():
    """The frame handed in has one row per *game*. Counting it directly would
    weight a three-game set three times."""
    df = a_season_with_ratings(300)
    expanded = pd.concat([df, df.iloc[:100], df.iloc[:100]], ignore_index=True)
    assert (sum(sum(x) for x in rating_distribution(expanded)["elo"]["counts"])
            == len(df))


def test_each_day_gets_its_own_bucket_row():
    df = a_season_with_ratings()
    df["battle_time"] = [f"2026090{1 + i % 5}T120000.000Z" for i in range(len(df))]
    r = rating_distribution(df)
    assert r["days"] == sorted(r["days"])
    assert len(r["elo"]["counts"]) == len(r["days"]) == 5
    for day, row in zip(r["days"], r["elo"]["counts"], strict=True):
        assert sum(row) == int((df["battle_time"].str[:8] == day).sum())


def test_a_rating_lands_in_the_bin_the_page_reads_it_from():
    """The page turns a bin index back into a value with lo + i*width."""
    df = a_season_with_ratings(200)
    df["avg_elo"] = 101 / ELO_STEPS_PER_POINT
    h = rating_distribution(df)["elo"]
    per_bin = [sum(col) for col in zip(*h["counts"], strict=True)]
    hit = [i for i, n in enumerate(per_bin) if n]
    assert len(hit) == 1
    assert h["lo"] + hit[0] * h["width"] == pytest.approx(101 / ELO_STEPS_PER_POINT)


def test_skill_bins_are_placed_on_the_declared_grid():
    r = rating_distribution(a_season_with_ratings())
    assert r["skill"]["lo"] == SKILL_LO
    assert r["skill"]["width"] == SKILL_BIN


def test_a_season_without_ratings_still_renders():
    df = a_season_with_ratings().drop(columns=["avg_elo"])
    r = rating_distribution(df)
    assert "elo" not in r
    assert "skill" in r
