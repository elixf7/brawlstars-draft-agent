"""An antisymmetric, field-aware win-probability model.

Two problems with the plain FM this replaces.

**It was not antisymmetric.** `t1_SHELLY` and `t2_SHELLY` were separate features
with separate embeddings, so P(A beats B) and P(B beats A) only summed to 1
insofar as the flip augmentation taught them to. Measured on real games they
disagreed by 0.047 on average and by as much as 0.307. That matters twice over:
tree search flips the evaluator's output to model the opponent, and self-play
labels the non-primary team with `1 - p`. Both assume an identity the model did
not have.

Here the score is a difference of team-wise terms, so `p(A,B) + p(B,A) = 1`
holds exactly, by construction rather than by training. Augmentation becomes
unnecessary, which halves the rows.

**Capacity was not the bottleneck.** Widening the plain FM from k=32 to k=128
changed log-loss by 0.0002. A single embedding per brawler had to serve every
kind of interaction at once. Here each brawler gets separate vectors for playing
*beside* someone, *against* someone, and *on* a map — the field-aware
factorization, which is the standard answer when a plain FM saturates.

Counters need care: an inner product is symmetric, so `<v_a, v_b>` says the same
thing about "a beats b" as about "b beats a". Attack and defend vectors are kept
apart so the term can be genuinely directional and still antisymmetrise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

TEAM_SIZE = 3


class AntisymmetricFFM(nn.Module):
    """P(team 1 wins), guaranteed to invert exactly when the teams swap.

    logit = [linear + synergy + context](t1) - [same](t2) + counter(t1, t2)

    There is no bias: a constant would survive the difference and break the
    guarantee. The small team-1 win rate in the data is a labelling artefact,
    not a property of either team.
    """

    def __init__(self, n_brawlers: int, n_maps: int, n_modes: int, k: int = 32) -> None:
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.zeros(n_brawlers))
        # One field per kind of interaction a brawler takes part in.
        self.e_syn = nn.Parameter(torch.randn(n_brawlers, k) * 0.01)   # beside
        self.e_att = nn.Parameter(torch.randn(n_brawlers, k) * 0.01)   # against, as attacker
        self.e_def = nn.Parameter(torch.randn(n_brawlers, k) * 0.01)   # against, as defender
        self.e_ctx = nn.Parameter(torch.randn(n_brawlers, k) * 0.01)   # with the context
        self.m_map = nn.Parameter(torch.randn(n_maps, k) * 0.01)
        self.m_mode = nn.Parameter(torch.randn(n_modes, k) * 0.01)
        self.v_skill = nn.Parameter(torch.zeros(k))

    def _synergy(self, team: torch.Tensor) -> torch.Tensor:
        """Sum of pairwise products within a team, the usual FM trick."""
        e = self.e_syn[team]                                  # (B, 3, k)
        pooled = e.sum(dim=1)                                 # (B, k)
        return 0.5 * (pooled.pow(2).sum(1) - e.pow(2).sum(dim=(1, 2)))

    def _context(self, team: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return (self.e_ctx[team].sum(dim=1) * ctx).sum(dim=1)

    def forward(
        self, t1: torch.Tensor, t2: torch.Tensor,
        map_idx: torch.Tensor, mode_idx: torch.Tensor, skill: torch.Tensor,
    ) -> torch.Tensor:
        ctx = (self.m_map[map_idx] + self.m_mode[mode_idx]
               + self.v_skill * skill.unsqueeze(1))           # (B, k)

        own = (self.w[t1].sum(1) + self._synergy(t1) + self._context(t1, ctx))
        opp = (self.w[t2].sum(1) + self._synergy(t2) + self._context(t2, ctx))

        # Directional: team 1 attacking team 2, minus team 2 attacking team 1.
        counter = ((self.e_att[t1].sum(1) * self.e_def[t2].sum(1)).sum(1)
                   - (self.e_att[t2].sum(1) * self.e_def[t1].sum(1)).sum(1))

        return own - opp + counter


@dataclass
class FFMInference:
    """NumPy inference, for tree search where the model is called constantly."""

    w: np.ndarray
    e_syn: np.ndarray
    e_att: np.ndarray
    e_def: np.ndarray
    e_ctx: np.ndarray
    m_map: np.ndarray
    m_mode: np.ndarray
    v_skill: np.ndarray
    vocab: list[str]
    maps: list[str]
    modes: list[str]
    val_logloss: float = float("nan")
    val_auc: float = float("nan")
    val_brier: float = float("nan")
    n_train: int = 0
    n_val: int = 0
    train_history: list = None

    @classmethod
    def from_model(cls, model: AntisymmetricFFM, vocab, maps, modes, **meta) -> FFMInference:
        g = {n: p.detach().cpu().numpy().astype(np.float32)
             for n, p in model.named_parameters()}
        return cls(w=g["w"], e_syn=g["e_syn"], e_att=g["e_att"], e_def=g["e_def"],
                   e_ctx=g["e_ctx"], m_map=g["m_map"], m_mode=g["m_mode"],
                   v_skill=g["v_skill"], vocab=list(vocab), maps=list(maps),
                   modes=list(modes), **meta)

    def logits(self, t1, t2, map_idx, mode_idx, skill) -> np.ndarray:
        ctx = self.m_map[map_idx] + self.m_mode[mode_idx] + self.v_skill * skill[:, None]

        def side(team):
            e = self.e_syn[team]
            pooled = e.sum(axis=1)
            syn = 0.5 * ((pooled ** 2).sum(1) - (e ** 2).sum(axis=(1, 2)))
            return self.w[team].sum(1) + syn + (self.e_ctx[team].sum(axis=1) * ctx).sum(1)

        counter = ((self.e_att[t1].sum(axis=1) * self.e_def[t2].sum(axis=1)).sum(1)
                   - (self.e_att[t2].sum(axis=1) * self.e_def[t1].sum(axis=1)).sum(1))
        return side(t1) - side(t2) + counter

    def predict(self, t1, t2, map_idx, mode_idx, skill) -> np.ndarray:
        z = self.logits(t1, t2, map_idx, mode_idx, skill)
        return 1.0 / (1.0 + np.exp(-z))
