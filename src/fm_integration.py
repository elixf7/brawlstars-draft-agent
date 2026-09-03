"""
fm_integration.py — Step 2.3: FM Integration with MCTS

Provides FMEvaluator: the single object MCTS uses to score terminal states.

    evaluator = FMEvaluator.load()
    win_prob = evaluator.evaluate(terminal_state)   # P(my_team wins)

Team assignment convention (must be consistent everywhere):
    my_team  → t1 feature slots  (t1_ prefix indices in FeatureSchema)
    opp_team → t2 feature slots  (t2_ prefix indices in FeatureSchema)

The FM predicts P(team1 wins).  "My team" is always team1.
Violating this silently returns (1 - correct_probability) — hard to debug.

Caching:
    Terminal compositions are frequently revisited via different rollout paths.
    Cache key: (my_team, opp_team, map_name, mode, skill_ns)
    is_first_pick and bans are excluded — they don't affect FM predictions.
    With a large brawler pool (~95), stochastic rollouts almost never reach the
    same terminal state twice within a single MCTS run — expect near-zero hit
    rate there.  The cache pays off across repeated recommend() calls in a
    session (same map/mode, different partial states sharing terminal states)
    and when bans significantly reduce the pick pool.

Note on tree reuse across picks:
    After the user confirms a pick, the MCTS root shifts to a child subtree.
    Reusing that subtree's simulation results is theoretically possible but
    requires careful root management and adds complexity.  Skipped for now —
    the cache already amortises the cost of repeated terminal evaluation.
    Revisit only if Step 2.8.1 shows throughput is insufficient.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from draft_state import DraftState       # noqa: E402
from fm_evaluate import FMEncoder        # noqa: E402
from fm_model import FMInference         # noqa: E402


class FMEvaluator:
    """
    Stateful evaluator that wraps FMInference + FMEncoder + evaluation cache.

    Construct once per MCTS session; call evaluate() at every terminal node.

    Parameters
    ----------
    fm : trained FMInference (weights + schema loaded from data/fm_model.pkl)
    """

    __slots__ = ("_fm", "_encoder", "_cache", "_hits", "_misses")

    def __init__(self, fm: FMInference) -> None:
        self._fm = fm
        self._encoder = FMEncoder(fm)
        self._cache: dict[tuple, float] = {}
        self._hits: int = 0
        self._misses: int = 0

    # ── Core evaluation ───────────────────────────────────────────────────────

    def evaluate(self, state: DraftState) -> float:
        """
        Return P(my_team wins) ∈ (0, 1) for a terminal draft state.

        my_team  → t1 feature slots (FM predicts P(team1 wins))
        opp_team → t2 feature slots

        Parameters
        ----------
        state : must be a terminal DraftState (both teams have 3 brawlers)

        Raises
        ------
        ValueError if state is not terminal
        """
        if not state.is_terminal:
            raise ValueError(
                f"evaluate() requires a terminal state; "
                f"got my_team={len(state.my_team)}, opp_team={len(state.opp_team)}"
            )

        # Cache key captures the FM-relevant context only.
        # is_first_pick and bans don't affect FM output — omitting them avoids
        # false cache misses when the same composition is reached from different
        # call-site contexts.
        cache_key = (
            state.my_team,
            state.opp_team,
            state.map_name,
            state.mode,
            state.skill_ns,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._hits += 1
            return cached

        self._misses += 1

        # Sort for deterministic tuple ordering.
        # Within-team ordering is irrelevant to the FM (it sums embeddings),
        # but FMEncoder requires a 3-tuple so we supply a consistent one.
        my_brawlers = tuple(sorted(state.my_team))
        opp_brawlers = tuple(sorted(state.opp_team))

        idx, val = self._encoder.encode(
            my_brawlers,   # type: ignore[arg-type]  # always exactly 3
            opp_brawlers,  # type: ignore[arg-type]
            state.map_name,
            state.mode,
            state.skill_ns,
        )
        # evaluate_sparse mutates idx/val on the next encode() call,
        # but we only need the scalar result here — no copy required.
        prob = self._fm.evaluate_sparse(idx, val)

        self._cache[cache_key] = prob
        return prob

    # ── Cache introspection ───────────────────────────────────────────────────

    @property
    def cache_stats(self) -> dict[str, object]:
        """
        Return cache performance metrics.

        Keys: hits, misses, hit_rate (float 0–1), size (number of cached states).
        """
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Reset the evaluation cache and hit/miss counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    # ── Construction helpers ──────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "FMEvaluator":
        """Load the trained FM from data/fm_model.pkl and return a ready evaluator."""
        fm = FMInference.load()
        return cls(fm)


# ── Sanity checks (run: python src/fm_integration.py) ─────────────────────────

if __name__ == "__main__":
    print("=== Step 2.3 Sanity Checks ===\n")

    evaluator = FMEvaluator.load()
    schema = evaluator._fm.schema

    # Pick a map and mode that exist in the schema
    _MAP  = schema.maps[0]
    _MODE = schema.modes[0]

    # ── Check 1: evaluate() returns a probability in (0, 1) ───────────────────
    vocab = schema.vocab
    state = DraftState(
        my_team=frozenset(vocab[:3]),
        opp_team=frozenset(vocab[3:6]),
        mode=_MODE,
        map_name=_MAP,
        skill_ns=0.5,
        is_first_pick=True,
    )
    assert state.is_terminal

    prob = evaluator.evaluate(state)
    assert 0.0 < prob < 1.0, f"Expected prob in (0,1), got {prob}"
    print(f"Check 1: evaluate() → {prob:.4f}  (expected ∈ (0, 1))")
    print(f"  my_team : {sorted(state.my_team)}")
    print(f"  opp_team: {sorted(state.opp_team)}")
    print("  ✓\n")

    # ── Check 2: swapping teams returns approximately (1 - original) ──────────
    # exact symmetry is guaranteed by construction of the FM + symmetric augmentation
    swapped = DraftState(
        my_team=state.opp_team,
        opp_team=state.my_team,
        mode=_MODE,
        map_name=_MAP,
        skill_ns=0.5,
        is_first_pick=True,
    )
    prob_swapped = evaluator.evaluate(swapped)
    symmetry_err = abs(prob + prob_swapped - 1.0)
    assert symmetry_err < 0.02, (
        f"Symmetry check failed: P(A) + P(B) = {prob + prob_swapped:.4f} (expected ≈ 1.0)"
    )
    print(f"Check 2: symmetry  P(A→B) + P(B→A) = {prob:.4f} + {prob_swapped:.4f} = {prob + prob_swapped:.4f}")
    print(f"  symmetry error = {symmetry_err:.6f}  (threshold 0.02)  ✓\n")

    # ── Check 3: second call is served from cache ──────────────────────────────
    stats_before = evaluator.cache_stats
    _ = evaluator.evaluate(state)           # second call — must be a cache hit
    stats_after = evaluator.cache_stats
    assert stats_after["hits"] == stats_before["hits"] + 1, "Expected a cache hit on second call"
    print(f"Check 3: cache hit on second call")
    print(f"  hits={stats_after['hits']}, misses={stats_after['misses']}, "
          f"hit_rate={stats_after['hit_rate']:.2f}  ✓\n")

    # ── Check 4: non-terminal state raises ValueError ─────────────────────────
    partial = DraftState(
        my_team=frozenset(vocab[:2]),
        opp_team=frozenset(vocab[3:5]),
        mode=_MODE,
        map_name=_MAP,
        skill_ns=0.5,
        is_first_pick=True,
    )
    assert not partial.is_terminal
    try:
        evaluator.evaluate(partial)
        assert False, "Expected ValueError for non-terminal state"
    except ValueError as e:
        print(f"Check 4: evaluate() on partial state raises ValueError: {e}")
        print("  ✓\n")

    print("=== All checks passed ===")
