"""Baselines the model has to beat.

A log-loss of 0.685 means nothing on its own. It means something against 0.693,
which is what you get from knowing nothing, and against what you get from
knowing progressively more: which brawlers win, which win on this map, and which
beat which. Each baseline below adds one kind of knowledge, so where the model
lands says what it actually learned.

All of them fit on train and predict on val, like the model does.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from bsdraft.data.sources import TEAM1_BRAWLER_COLS, TEAM2_BRAWLER_COLS

#: Pseudo-count pulling a sparsely-observed rate back toward the base rate. A
#: brawler seen four times has not earned an 80% win rate.
DEFAULT_SHRINKAGE = 50.0


def _shrink(wins: pd.Series, n: pd.Series, prior: float, alpha: float) -> pd.Series:
    return (wins + alpha * prior) / (n + alpha)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Baseline(ABC):
    """A predictor of P(team 1 wins), fit on train and scored on val."""

    name: str = "baseline"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> Baseline: ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


class ConstantBaseline(Baseline):
    """Always predict the training base rate.

    The floor. Anything that cannot beat this has learned nothing at all.
    """

    name = "constant (base rate)"

    def fit(self, train: pd.DataFrame) -> ConstantBaseline:
        self.rate_ = float(train["team1_wins"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.rate_, dtype=np.float64)


class BrawlerWinRateBaseline(Baseline):
    """Each brawler's own win rate, summed over the team.

    Knows which brawlers are strong, and nothing about maps or matchups. The gap
    between this and the model is what composition and context are worth.
    """

    name = "brawler win rate"

    def __init__(self, alpha: float = DEFAULT_SHRINKAGE) -> None:
        self.alpha = alpha

    def fit(self, train: pd.DataFrame) -> BrawlerWinRateBaseline:
        self.prior_ = float(train["team1_wins"].mean())
        appearances = []
        for cols, won in ((TEAM1_BRAWLER_COLS, train["team1_wins"]),
                          (TEAM2_BRAWLER_COLS, 1 - train["team1_wins"])):
            for c in cols:
                appearances.append(pd.DataFrame({"brawler": train[c], "won": won}))
        stacked = pd.concat(appearances, ignore_index=True).dropna(subset=["brawler"])
        grouped = stacked.groupby("brawler")["won"].agg(["sum", "count"])
        self.rates_ = _shrink(grouped["sum"], grouped["count"], self.prior_, self.alpha)
        return self

    def _team_logit(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        vals = [
            _logit(df[c].map(self.rates_).fillna(self.prior_).to_numpy(dtype=float))
            for c in cols
        ]
        return np.mean(vals, axis=0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        diff = self._team_logit(df, TEAM1_BRAWLER_COLS) - self._team_logit(df, TEAM2_BRAWLER_COLS)
        # Team 1's advantage is the difference in aggregate strength, re-centred
        # on the base rate so the average prediction stays honest.
        return _sigmoid(diff + _logit(np.array([self.prior_]))[0])


class MapBrawlerBaseline(BrawlerWinRateBaseline):
    """Win rate per (map, brawler), falling back to the brawler's overall rate.

    Knows that brawlers are map-dependent — the single largest source of
    structure in this game short of interactions between picks.
    """

    name = "brawler x map win rate"

    def fit(self, train: pd.DataFrame) -> MapBrawlerBaseline:
        super().fit(train)
        rows = []
        for cols, won in ((TEAM1_BRAWLER_COLS, train["team1_wins"]),
                          (TEAM2_BRAWLER_COLS, 1 - train["team1_wins"])):
            for c in cols:
                rows.append(pd.DataFrame(
                    {"map": train["map"], "brawler": train[c], "won": won}))
        stacked = pd.concat(rows, ignore_index=True).dropna(subset=["brawler"])
        grouped = stacked.groupby(["map", "brawler"])["won"].agg(["sum", "count"])
        # Shrink toward the brawler's overall rate, not the global one: a
        # brawler with few games on a map should look like itself elsewhere.
        brawler_prior = grouped.index.get_level_values("brawler").map(self.rates_)
        brawler_prior = pd.Series(brawler_prior, index=grouped.index).fillna(self.prior_)
        self.map_rates_ = (
            (grouped["sum"] + self.alpha * brawler_prior) / (grouped["count"] + self.alpha)
        )
        return self

    def _team_logit(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        vals = []
        for c in cols:
            keyed = pd.MultiIndex.from_arrays([df["map"], df[c]])
            rate = pd.Series(self.map_rates_.reindex(keyed).to_numpy(), index=df.index)
            fallback = df[c].map(self.rates_).fillna(self.prior_)
            vals.append(_logit(rate.fillna(fallback).to_numpy(dtype=float)))
        return np.mean(vals, axis=0)


class PairwiseMatchupBaseline(Baseline):
    """Empirical win rate for each (team-1 brawler, team-2 brawler) pair.

    Knows which brawlers beat which — the interaction the tree search leans on.
    Averaged over the nine cross-team pairs in a game.
    """

    name = "pairwise matchup"

    def __init__(self, alpha: float = DEFAULT_SHRINKAGE) -> None:
        self.alpha = alpha

    def fit(self, train: pd.DataFrame) -> PairwiseMatchupBaseline:
        self.prior_ = float(train["team1_wins"].mean())
        pairs = []
        for a in TEAM1_BRAWLER_COLS:
            for b in TEAM2_BRAWLER_COLS:
                pairs.append(pd.DataFrame({
                    "a": train[a], "b": train[b], "won": train["team1_wins"]}))
        stacked = pd.concat(pairs, ignore_index=True).dropna(subset=["a", "b"])
        grouped = stacked.groupby(["a", "b"])["won"].agg(["sum", "count"])
        self.rates_ = _shrink(grouped["sum"], grouped["count"], self.prior_, self.alpha)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        logits = []
        for a in TEAM1_BRAWLER_COLS:
            for b in TEAM2_BRAWLER_COLS:
                keyed = pd.MultiIndex.from_arrays([df[a], df[b]])
                rate = pd.Series(self.rates_.reindex(keyed).to_numpy(), index=df.index)
                logits.append(_logit(rate.fillna(self.prior_).to_numpy(dtype=float)))
        return _sigmoid(np.mean(logits, axis=0))


def default_baselines(alpha: float = DEFAULT_SHRINKAGE) -> list[Baseline]:
    """Ordered by how much they know, least to most."""
    return [
        ConstantBaseline(),
        BrawlerWinRateBaseline(alpha),
        MapBrawlerBaseline(alpha),
        PairwiseMatchupBaseline(alpha),
    ]
