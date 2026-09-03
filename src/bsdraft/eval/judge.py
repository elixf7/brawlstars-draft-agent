"""An evaluator the policy is not being optimised against.

Self-play optimises a policy against the FM and then promotes it on the FM's own
opinion of the drafts it produced. That is a closed loop: a policy that learns
to draft compositions the evaluator overrates gets promoted for it, and the
promotion number looks healthy the whole way.

The fix is to judge with something else. Two evaluators are trained on disjoint
time slices of the season — the first half drives search, the second half
decides promotion. They see different matches, so a composition that only the
search evaluator likes does not survive.

It is not a perfect independence: both learn from the same game in the same
season. But it breaks the self-reference, which is the part that was wrong.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bsdraft.fm.ffm import FFMInference
from bsdraft.fm.train_ffm import predict_df, train_ffm


@dataclass
class JudgePair:
    """Two evaluators over disjoint time slices."""

    search: FFMInference   # earlier half — drives self-play
    judge: FFMInference    # later half — decides promotion

    @property
    def summary(self) -> dict[str, float]:
        return {
            "search_logloss": self.search.val_logloss,
            "search_auc": self.search.val_auc,
            "judge_logloss": self.judge.val_logloss,
            "judge_auc": self.judge.val_auc,
        }


def train_judge_pair(
    df: pd.DataFrame, *, val_fraction: float = 0.15, verbose: bool = False, **kw
) -> JudgePair:
    """Split the season in half by time and train one evaluator on each.

    Each half keeps its own validation tail, so both numbers are honest.
    """
    df = df.sort_values("battle_time").reset_index(drop=True)
    mid = len(df) // 2
    halves = [df.iloc[:mid], df.iloc[mid:]]

    trained = []
    for half in halves:
        cut = int(len(half) * (1 - val_fraction))
        trained.append(train_ffm(
            half.iloc[:cut], half.iloc[cut:], verbose=verbose, **kw
        ))
    return JudgePair(search=trained[0], judge=trained[1])


def score_compositions(judge: FFMInference, drafts: pd.DataFrame) -> np.ndarray:
    """The judge's win probability for each finished draft."""
    return predict_df(judge, drafts)


def promotion_verdict(
    judge: FFMInference,
    new_drafts: pd.DataFrame,
    old_drafts: pd.DataFrame,
    *,
    threshold: float = 0.525,
) -> dict[str, float | bool]:
    """Should the new policy replace the old one?

    Judged on compositions each produced, by an evaluator neither optimised
    against. A margin is reported alongside so a promotion just over the line
    can be told from a decisive one.
    """
    new_scores = score_compositions(judge, new_drafts)
    old_scores = score_compositions(judge, old_drafts)
    new_mean, old_mean = float(new_scores.mean()), float(old_scores.mean())

    # Standard error of the difference, so "0.53" can be read against noise.
    se = float(np.sqrt(new_scores.var(ddof=1) / len(new_scores)
                       + old_scores.var(ddof=1) / len(old_scores)))
    return {
        "new_mean": new_mean,
        "old_mean": old_mean,
        "margin": new_mean - old_mean,
        "std_error": se,
        "threshold": threshold,
        "promote": new_mean >= threshold,
        "significant": abs(new_mean - old_mean) > 2 * se if se > 0 else False,
    }
