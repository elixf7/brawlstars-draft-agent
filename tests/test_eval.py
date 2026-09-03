"""A metric without a baseline is not a result.

These check that each baseline knows what it claims to know, and that the
comparison scores everything on the same rows."""
import numpy as np
import pandas as pd
import pytest

from bsdraft.eval import (
    RANDOM_LOGLOSS,
    compare_predictors,
    default_baselines,
    expected_calibration_error,
    score,
)
from bsdraft.eval.baselines import (
    BrawlerWinRateBaseline,
    ConstantBaseline,
    MapBrawlerBaseline,
    PairwiseMatchupBaseline,
)


def games(n=600, seed=0):
    """Synthetic games where STRONG genuinely wins more often."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        t1_strong = rng.random() < 0.5
        t1 = ["STRONG" if t1_strong else "WEAK", "A", "B"]
        t2 = ["WEAK" if t1_strong else "STRONG", "C", "D"]
        p = 0.75 if t1_strong else 0.25
        rows.append({
            "map": rng.choice(["M1", "M2"]), "mode": "brawlBall",
            "t1_b0_name": t1[0], "t1_b1_name": t1[1], "t1_b2_name": t1[2],
            "t2_b0_name": t2[0], "t2_b1_name": t2[1], "t2_b2_name": t2[2],
            "team1_wins": int(rng.random() < p),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def split():
    df = games()
    return df.iloc[:400].reset_index(drop=True), df.iloc[400:].reset_index(drop=True)


# ------------------------------------------------------------------- metrics
def test_random_baseline_is_log_two():
    assert RANDOM_LOGLOSS == pytest.approx(0.6931, abs=1e-4)


def test_a_coin_flip_scores_the_random_baseline():
    labels = np.array([0, 1] * 50)
    s = score(np.full(100, 0.5), labels)
    assert s["logloss"] == pytest.approx(RANDOM_LOGLOSS, abs=1e-6)
    assert s["brier"] == pytest.approx(0.25, abs=1e-9)


def test_perfect_predictions_score_near_zero():
    labels = np.array([0, 1, 0, 1])
    s = score(np.array([0.001, 0.999, 0.001, 0.999]), labels)
    assert s["logloss"] < 0.01 and s["auc"] == 1.0


def test_calibration_error_is_zero_when_predictions_match_reality():
    probs = np.concatenate([np.full(500, 0.2), np.full(500, 0.8)])
    labels = np.concatenate([
        np.array([1] * 100 + [0] * 400), np.array([1] * 400 + [0] * 100)
    ])
    assert expected_calibration_error(probs, labels) < 0.01


def test_calibration_error_catches_overconfidence():
    """Search multiplies probabilities, so an overconfident evaluator compounds."""
    probs = np.full(1000, 0.95)
    labels = np.array([1] * 500 + [0] * 500)      # actually a coin flip
    assert expected_calibration_error(probs, labels) == pytest.approx(0.45, abs=0.01)


# ----------------------------------------------------------------- baselines
def test_constant_baseline_predicts_the_training_base_rate(split):
    train, val = split
    b = ConstantBaseline().fit(train)
    preds = b.predict(val)
    assert np.allclose(preds, train["team1_wins"].mean())
    assert len(preds) == len(val)


def test_constant_baseline_cannot_rank(split):
    """AUC 0.5 by construction — it is the floor, and should look like it."""
    train, val = split
    s = score(ConstantBaseline().fit(train).predict(val), val["team1_wins"].to_numpy())
    assert s["auc"] == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize("cls", [BrawlerWinRateBaseline, MapBrawlerBaseline,
                                 PairwiseMatchupBaseline])
def test_each_baseline_beats_the_floor_on_learnable_data(cls, split):
    """If a baseline cannot beat a constant on data with real signal, it is
    not measuring what it claims to."""
    train, val = split
    labels = val["team1_wins"].to_numpy()
    floor = score(ConstantBaseline().fit(train).predict(val), labels)["logloss"]
    got = score(cls().fit(train).predict(val), labels)["logloss"]
    assert got < floor


def test_baselines_produce_valid_probabilities(split):
    train, val = split
    for b in default_baselines():
        p = b.fit(train).predict(val)
        assert p.shape == (len(val),)
        assert np.all((p > 0) & (p < 1)), f"{b.name} produced values outside (0,1)"


def test_unseen_brawlers_fall_back_rather_than_crash(split):
    """New brawlers ship most seasons; a model that crashes on one is useless."""
    train, val = split
    val = val.copy()
    val.loc[val.index[0], "t1_b0_name"] = "BRAND_NEW_BRAWLER"
    for b in default_baselines():
        p = b.fit(train).predict(val)
        assert np.isfinite(p).all(), f"{b.name} produced non-finite output"


def test_shrinkage_pulls_sparse_rates_toward_the_prior():
    """A brawler seen four times has not earned an 80% win rate."""
    train = games(400)
    rare = train.iloc[:4].copy()
    rare["t1_b0_name"] = "RARE"
    rare["team1_wins"] = 1
    train = pd.concat([train, rare], ignore_index=True)

    strong = BrawlerWinRateBaseline(alpha=200.0).fit(train)
    weak = BrawlerWinRateBaseline(alpha=1.0).fit(train)
    base = train["team1_wins"].mean()
    assert abs(strong.rates_["RARE"] - base) < abs(weak.rates_["RARE"] - base)


# ---------------------------------------------------------------- comparison
def test_comparison_scores_everything_on_the_same_rows(split):
    train, val = split
    c = compare_predictors(train, val)
    assert c.n_train == len(train) and c.n_val == len(val)
    assert len(c.rows) == len(default_baselines())
    assert c.best["logloss"] == min(r["logloss"] for r in c.rows)


def test_a_model_can_be_added_to_the_comparison(split):
    train, val = split
    probs = np.clip(val["team1_wins"].to_numpy() * 0.9 + 0.05, 0.01, 0.99)
    c = compare_predictors(train, val, model_probs=probs, model_name="fm")
    assert c.best["name"] == "fm"        # a near-oracle should win


def test_misaligned_model_predictions_are_rejected(split):
    """Silently scoring the wrong rows would produce a meaningless number."""
    train, val = split
    with pytest.raises(ValueError, match="predictions for"):
        compare_predictors(train, val, model_probs=np.full(len(val) - 1, 0.5))


def test_report_shows_the_lift_over_random(split):
    train, val = split
    text = compare_predictors(train, val).render()
    assert "vs random" in text and "logloss" in text
    assert "best:" in text
