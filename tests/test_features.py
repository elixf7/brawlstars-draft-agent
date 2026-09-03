"""Feature construction, checked against a sample of real games.

Synthetic rows agree with whatever assumptions wrote them. These use 4,000
actual games — every map, every mode, the real brawler vocabulary."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bsdraft.features.engineering import (
    build_feature_matrix,
    build_schema,
    chronological_split,
)

FIXTURE = Path(__file__).parent / "fixtures" / "games_sample.parquet"


@pytest.fixture(scope="module")
def games():
    assert FIXTURE.exists(), "fixture missing; see tests/fixtures/README.md"
    return pd.read_parquet(FIXTURE)


@pytest.fixture(scope="module")
def schema(games):
    return build_schema(games)


def test_the_fixture_is_real_data(games):
    assert len(games) >= 1_000
    assert games["map"].nunique() > 10
    assert set(games["mode"].unique()) <= {
        "bounty", "brawlBall", "gemGrab", "heist", "hotZone", "knockout"}
    assert games["team1_wins"].isin([0, 1]).all()


# ------------------------------------------------------------------- schema
def test_schema_covers_every_brawler_in_the_data(games, schema):
    seen = pd.unique(games[[f"t{t}_b{b}_name" for t in (1, 2) for b in range(3)]]
                     .to_numpy().ravel())
    assert set(schema.vocab) >= {b for b in seen if isinstance(b, str)}


def test_schema_blocks_do_not_overlap(schema):
    """Overlapping offsets would silently mix a brawler with a map."""
    offsets = [schema.t1_offset, schema.t2_offset, schema.map_offset]
    assert offsets == sorted(offsets)
    assert schema.t2_offset - schema.t1_offset == len(schema.vocab)


# ---------------------------------------------------------- feature matrix
def test_each_game_is_emitted_twice_for_symmetry(games, schema):
    """Once as played and once with teams swapped, so the model cannot learn
    that team 1 wins slightly more often than team 2."""
    X, y = build_feature_matrix(games.head(200), schema)
    assert X.shape[0] == 400
    assert len(y) == 400


def test_the_flipped_copy_carries_the_opposite_label(games, schema):
    X, y = build_feature_matrix(games.head(200), schema)
    assert np.array_equal(y[1::2], 1 - y[0::2])


def test_the_as_played_labels_match_the_source(games, schema):
    subset = games.head(200)
    _, y = build_feature_matrix(subset, schema)
    assert np.array_equal(y[0::2], subset["team1_wins"].to_numpy())


def test_flipping_swaps_the_team_blocks(games, schema):
    """A game and its mirror must be the same features with the team halves
    exchanged — otherwise augmentation teaches the model something false."""
    X, _ = build_feature_matrix(games.head(1), schema)
    played, mirrored = X[0].toarray().ravel(), X[1].toarray().ravel()
    n = len(schema.vocab)
    t1, t2 = schema.t1_offset, schema.t2_offset
    assert np.array_equal(played[t1:t1 + n], mirrored[t2:t2 + n])
    assert np.array_equal(played[t2:t2 + n], mirrored[t1:t1 + n])


def test_every_row_has_the_expected_number_of_active_features(games, schema):
    """Three brawlers a side, one map, one skill value, optionally one mode."""
    X, _ = build_feature_matrix(games.head(500), schema)
    per_row = np.diff(X.indptr)
    expected = 9 if schema.include_mode else 8
    assert set(per_row.tolist()) == {expected}


def test_context_is_shared_by_a_game_and_its_mirror(games, schema):
    """Map and mode describe the game, not a team; flipping must not move them."""
    X, _ = build_feature_matrix(games.head(1), schema)
    played, mirrored = X[0].toarray().ravel(), X[1].toarray().ravel()
    assert np.array_equal(played[schema.map_offset:], mirrored[schema.map_offset:])


# -------------------------------------------------------------------- split
def test_the_split_is_chronological(games):
    """A random split leaks the meta forward: the model would see the future."""
    train, val = chronological_split(games, val_fraction=0.2)
    assert train["battle_time"].max() <= val["battle_time"].min()


def test_the_split_keeps_every_row_exactly_once(games):
    train, val = chronological_split(games, val_fraction=0.25)
    assert len(train) + len(val) == len(games)
    assert set(train["id"]).isdisjoint(val["id"])


def test_the_split_honours_the_requested_fraction(games):
    _, val = chronological_split(games, val_fraction=0.3)
    assert len(val) / len(games) == pytest.approx(0.3, abs=0.01)
