"""The antisymmetric model's defining property, and the judge that breaks the
self-play loop's self-reference."""
import numpy as np
import pandas as pd
import pytest
import torch

from bsdraft.fm.ffm import AntisymmetricFFM, FFMInference
from bsdraft.fm.train_ffm import build_vocabularies, encode, predict_df, train_ffm

FIXTURE = "tests/fixtures/games_sample.parquet"


@pytest.fixture(scope="module")
def games():
    return pd.read_parquet(FIXTURE)


def a_model(seed=0, k=8, n=20):
    torch.manual_seed(seed)
    m = AntisymmetricFFM(n, 5, 3, k=k)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.3)     # non-trivial weights
    return m


def batch():
    return (torch.tensor([[0, 1, 2], [3, 4, 5]]), torch.tensor([[6, 7, 8], [9, 10, 11]]),
            torch.tensor([0, 1]), torch.tensor([0, 2]), torch.tensor([0.5, -1.0]))


# -------------------------------------------------------- the whole point
def test_swapping_the_teams_inverts_the_probability_exactly():
    """The original FM was off by up to 0.307 here. Search flips the evaluator
    to model the opponent and self-play labels the second team `1 - p`, so both
    depend on this identity holding."""
    m = a_model()
    t1, t2, mp, md, sk = batch()
    p = torch.sigmoid(m(t1, t2, mp, md, sk))
    q = torch.sigmoid(m(t2, t1, mp, md, sk))
    assert torch.allclose(p + q, torch.ones_like(p), atol=1e-6)


def test_a_mirror_match_is_exactly_even():
    """Identical teams must give 0.5 — no residual team-1 advantage."""
    m = a_model()
    team = torch.tensor([[0, 1, 2]])
    p = torch.sigmoid(m(team, team, torch.tensor([0]), torch.tensor([0]),
                        torch.tensor([0.3])))
    assert p.item() == pytest.approx(0.5, abs=1e-6)


def test_antisymmetry_survives_the_numpy_inference_path():
    """Tree search uses the NumPy path; a discrepancy there would reintroduce
    the bug where it matters most."""
    m = a_model()
    inf = FFMInference.from_model(m, [f"b{i}" for i in range(20)],
                                  [f"m{i}" for i in range(5)], ["x", "y", "z"])
    t1, t2, mp, md, sk = (a.numpy() for a in batch())
    p = inf.predict(t1, t2, mp, md, sk.astype(np.float32))
    q = inf.predict(t2, t1, mp, md, sk.astype(np.float32))
    assert np.allclose(p + q, 1.0, atol=1e-6)


def test_numpy_and_torch_agree():
    m = a_model()
    t1, t2, mp, md, sk = batch()
    torch_p = torch.sigmoid(m(t1, t2, mp, md, sk)).detach().numpy()
    inf = FFMInference.from_model(m, [f"b{i}" for i in range(20)],
                                  [f"m{i}" for i in range(5)], ["x", "y", "z"])
    numpy_p = inf.predict(t1.numpy(), t2.numpy(), mp.numpy(), md.numpy(),
                          sk.numpy().astype(np.float32))
    assert np.abs(torch_p - numpy_p).max() < 1e-5


def test_context_does_not_break_the_symmetry():
    """Map, mode and skill describe the game, not a team — they must cancel."""
    m = a_model()
    t1, t2 = torch.tensor([[0, 1, 2]]), torch.tensor([[6, 7, 8]])
    for mp, md, sk in [(0, 0, -2.0), (3, 1, 0.0), (4, 2, 3.0)]:
        args = (torch.tensor([mp]), torch.tensor([md]), torch.tensor([sk]))
        p = torch.sigmoid(m(t1, t2, *args))
        q = torch.sigmoid(m(t2, t1, *args))
        assert (p + q).item() == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------ encoding
def test_unknown_brawlers_do_not_crash(games):
    """New brawlers ship most seasons; the vocabulary grew 95 -> 106 already."""
    vocab, maps, modes = build_vocabularies(games)
    odd = games.head(5).copy()
    odd.loc[odd.index[0], "t1_b0_name"] = "BRAWLER_FROM_THE_FUTURE"
    odd.loc[odd.index[0], "map"] = "Map That Does Not Exist"
    enc = encode(odd, vocab, maps, modes)
    assert enc["t1"].shape == (5, 3)
    assert np.isfinite(enc["skill"]).all()


def test_vocabularies_cover_the_data(games):
    vocab, maps, modes = build_vocabularies(games)
    assert len(vocab) > 50
    assert set(maps) == set(games["map"].dropna().unique())
    assert set(modes) <= {"bounty", "brawlBall", "gemGrab", "heist", "hotZone", "knockout"}


# ------------------------------------------------------------------ training
def test_training_learns_something_and_stays_antisymmetric(games):
    train, val = games.iloc[:3000], games.iloc[3000:]
    inf = train_ffm(train, val, k=8, max_epochs=3, verbose=False)
    assert inf.val_logloss < 0.6931          # better than knowing nothing

    played = predict_df(inf, val)
    flipped = val.rename(columns={
        **{f"t1_b{i}_name": f"t2_b{i}_name" for i in range(3)},
        **{f"t2_b{i}_name": f"t1_b{i}_name" for i in range(3)},
    })
    assert np.allclose(played + predict_df(inf, flipped), 1.0, atol=1e-5)


# --------------------------------------------------------------------- judge
def test_the_judge_is_trained_on_data_the_search_model_never_saw(games):
    """Promotion judged by the evaluator being optimised against is circular."""
    from bsdraft.eval.judge import train_judge_pair

    pair = train_judge_pair(games, k=8, max_epochs=2, verbose=False)
    assert pair.search is not pair.judge
    assert set(pair.summary) == {"search_logloss", "search_auc",
                                 "judge_logloss", "judge_auc"}


def test_promotion_reports_a_margin_against_noise(games):
    from bsdraft.eval.judge import promotion_verdict, train_judge_pair

    pair = train_judge_pair(games, k=8, max_epochs=2, verbose=False)
    verdict = promotion_verdict(pair.judge, games.iloc[:200], games.iloc[200:400])
    assert set(verdict) >= {"new_mean", "old_mean", "margin", "std_error",
                            "promote", "significant"}
    # A bare "0.53" cannot be told from noise without this.
    assert verdict["std_error"] > 0
    assert verdict["promote"] == (verdict["new_mean"] >= verdict["threshold"])
