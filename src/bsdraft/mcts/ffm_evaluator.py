"""Wraps the antisymmetric model for tree search.

Same `evaluate(state) -> P(my team wins)` contract as FMEvaluator, so search and
self-play do not care which model they are given.

Two things come free from the model being antisymmetric. Team order within a
side does not matter, because each side is summed — so the frozensets in
DraftState map straight through. And evaluating the mirrored state costs
nothing: it is exactly `1 - p`, which is what search assumed all along and only
now is actually true.
"""
from __future__ import annotations

import numpy as np

from bsdraft.fm.ffm import FFMInference
from bsdraft.mcts.state import DraftState


class _SchemaView:
    """The three vocabularies search asks an evaluator for.

    Search, rollouts and self-play read `evaluator._fm.schema.vocab` in six
    places. The antisymmetric model holds the same lists under different names,
    so this exposes them in the shape those call sites expect rather than
    changing all six and the classic evaluator alongside them.
    """

    __slots__ = ("vocab", "maps", "modes")

    def __init__(self, model: FFMInference) -> None:
        self.vocab = model.vocab
        self.maps = model.maps
        self.modes = model.modes


class _ModelView:
    __slots__ = ("schema",)

    def __init__(self, model: FFMInference) -> None:
        self.schema = _SchemaView(model)


class FFMEvaluator:
    """Cached terminal-state evaluation for tree search."""

    def __init__(self, model: FFMInference) -> None:
        self.model = model
        self._fm = _ModelView(model)
        self._vocab = {b: i for i, b in enumerate(model.vocab)}
        self._maps = {m: i for i, m in enumerate(model.maps)}
        self._modes = {m: i for i, m in enumerate(model.modes)}
        self._cache: dict[tuple, float] = {}
        self.hits = self.misses = 0

    def _team(self, team: frozenset[str]) -> list[int]:
        return [self._vocab.get(b, 0) for b in sorted(team)]

    def evaluate(self, state: DraftState) -> float:
        if not state.is_terminal:
            raise ValueError(
                f"evaluate() requires a terminal state; got "
                f"my_team={len(state.my_team)}, opp_team={len(state.opp_team)}"
            )
        key = (state.my_team, state.opp_team, state.map_name, state.mode,
               round(state.skill_ns, 4))
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1

        p = float(self.model.predict(
            np.array([self._team(state.my_team)], dtype=np.int64),
            np.array([self._team(state.opp_team)], dtype=np.int64),
            np.array([self._maps.get(state.map_name, 0)], dtype=np.int64),
            np.array([self._modes.get(state.mode, 0)], dtype=np.int64),
            np.array([state.skill_ns], dtype=np.float32),
        )[0])
        self._cache[key] = p
        return p

    def evaluate_batch(self, states: list[DraftState]) -> np.ndarray:
        """Score many terminal states at once — far faster than one at a time."""
        if not states:
            return np.zeros(0, dtype=np.float32)
        return self.model.predict(
            np.array([self._team(s.my_team) for s in states], dtype=np.int64),
            np.array([self._team(s.opp_team) for s in states], dtype=np.int64),
            np.array([self._maps.get(s.map_name, 0) for s in states], dtype=np.int64),
            np.array([self._modes.get(s.mode, 0) for s in states], dtype=np.int64),
            np.array([s.skill_ns for s in states], dtype=np.float32),
        )

    def cache_stats(self) -> dict[str, object]:
        total = self.hits + self.misses
        return {"size": len(self._cache), "hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0}

    def clear_cache(self) -> None:
        self._cache.clear()
        self.hits = self.misses = 0
