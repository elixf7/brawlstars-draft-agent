"""
rollout.py — Step 2.4 / 3.1: Rollout Policy & Opponent Modeling

Completes an unfinished draft from a leaf node to a terminal state via
stochastic sampling, then returns the FM win probability.

Sampling modes
--------------
Each picker (my team or opponent) can use one of two rollout modes:

  "weighted" — sample proportionally to a linear blend of pick_rate and
               avg_counter_rate.  Controlled by the corresponding
               counter_weight parameter (0 = popularity-only, 1 = pure
               counter).  This is the original behavior.

  "top_k"   — rank all available brawlers by avg_counter_rate against the
               opposing team, then sample uniformly from the top-k.
               k=1 is deterministic greedy (always picks the hardest counter).
               k=3 is the default: adversarial but not perfectly precise.
               Falls back to "weighted" (pure pick_rate) when there are no
               opposing picks yet — counter signal is undefined.

Sampling formula ("weighted")
------------------------------
    weight(b) = (1 - w) * pick_rate(b) + w * avg_counter_rate(b)

  pick_rate(b)        — empirical pick popularity from matchup DB
  avg_counter_rate(b) — average win rate of b against the existing opposing
                        brawlers (e.g. for an opponent pick, "against my team")
  w ∈ [0, 1]         — counter_weight parameter for this picker's side

When the opposing side has no picks yet, avg_counter_rate defaults to 0.5
for all brawlers (neutral), so weights reduce to pure pick_rate at that
point regardless of w.

Sparsity filter
---------------
Brawlers whose pick_rate falls below `min_pick_rate` are excluded from
the candidate pool before weights/counter rates are computed.  This prevents
brawlers with near-zero empirical support from dominating counter selections.

If the filter removes all candidates (extreme edge case), it is silently
disabled for that pick.

My-team vs. opponent parameters
---------------------------------
`my_counter_weight`  — counter weight for my team's future rollout picks
                        (only used when my_rollout_mode="weighted").
                        Default 0.3: mostly popularity-driven.

`opp_counter_weight` — counter weight for opponent picks
                        (only used when opp_rollout_mode="weighted").
                        Default 0.5: balanced.

`my_rollout_mode`    — "weighted" (default) or "top_k".  Use "top_k" to
                        model my team as picking optimally from their best
                        synergy/counter options.

`opp_rollout_mode`   — "weighted" (default) or "top_k".  Use "top_k" for
                        adversarial analysis where the opponent always
                        counter-picks strategically.

`my_top_k` / `opp_top_k` — k for top_k mode.  Default 3.  k=1 = greedy.

Typical usage
-------------
  Solo queue (default):     rollout(..., opp_rollout_mode="weighted", opp_counter_weight=0.5)
  Adversarial opponent:     rollout(..., opp_rollout_mode="top_k", opp_top_k=3)
  Worst-case (greedy opp):  rollout(..., opp_rollout_mode="top_k", opp_top_k=1)
  Coordinated both teams:   rollout(..., my_rollout_mode="top_k", my_top_k=3,
                                        opp_rollout_mode="top_k", opp_top_k=3)

Note on tree vs. rollout adversarial behavior
----------------------------------------------
The MCTS tree selection (UCB1 in mcts_node.py) is always adversarial:
opponent-turn nodes store values from the opponent's perspective, so
maximising UCB1 there automatically minimises my win probability.
The rollout mode parameters here only govern rollout sampling after reaching
a leaf — they do not change tree selection.
"""

from __future__ import annotations

import numpy as np

from bsdraft.data.matchup_db import MatchupDB, skill_ns_to_tier
from bsdraft.mcts.evaluator import FMEvaluator
from bsdraft.mcts.state import DraftState, apply_pick, available_actions

# Default rollout parameters.
DEFAULT_MY_COUNTER_WEIGHT: float = 0.3
DEFAULT_OPP_COUNTER_WEIGHT: float = 0.5
DEFAULT_MIN_PICK_RATE: float = 0.0   # no filter unless caller specifies
DEFAULT_ROLLOUT_MODE: str = "weighted"
DEFAULT_TOP_K: int = 3
DEFAULT_MIN_COUNTER_GAMES: int = 0   # no filter unless caller specifies


# ── Internal helpers ───────────────────────────────────────────────────────────

def _avg_counter_rate(
    brawler: str,
    against: frozenset[str],
    db: MatchupDB,
    mode: str,
    map_name: str,
    tier: int,
) -> float:
    """
    Average win rate of `brawler` against every brawler in `against`.

    Returns 0.5 (neutral) if `against` is empty or a lookup fails —
    correct behavior when there are no brawlers to counter yet.
    """
    if not against:
        return 0.5
    total = 0.0
    for opp in against:
        entry = db.counter_lookup(brawler, opp, mode, map_name, tier)
        total += entry["win_rate"] if entry else 0.5
    return total / len(against)


def _sampling_weights(
    candidates: list[str],
    context: frozenset[str],
    db: MatchupDB,
    mode: str,
    map_name: str,
    tier: int,
    counter_weight: float,
    min_pick_rate: float,
) -> np.ndarray:
    """
    Compute a normalised probability vector over `candidates`.

    weight(b) = (1 - counter_weight) * pick_rate(b)
              +      counter_weight   * avg_counter_rate(b, context)

    Brawlers with pick_rate < min_pick_rate are excluded.  If the filter
    removes all candidates, the filter is disabled for this call.

    Falls back to uniform if all weights collapse to zero after filtering.
    """
    n = len(candidates)
    pick_rates = np.empty(n, dtype=np.float64)
    counter_rates = np.empty(n, dtype=np.float64)

    for i, brawler in enumerate(candidates):
        entry = db.brawler_lookup(brawler, mode, map_name, tier)
        pick_rates[i] = entry["pick_rate"] if entry else (1.0 / n)
        counter_rates[i] = _avg_counter_rate(brawler, context, db, mode, map_name, tier)

    # Apply minimum pick_rate filter to screen out sparse brawlers.
    mask = pick_rates >= min_pick_rate
    if not mask.any():
        mask[:] = True  # all candidates below threshold — disable filter

    weights = np.where(
        mask,
        (1.0 - counter_weight) * pick_rates + counter_weight * counter_rates,
        0.0,
    )

    total = weights.sum()
    if total <= 0.0:
        # Uniform fallback over filtered candidates.
        weights = mask.astype(np.float64)
        total = float(mask.sum())

    return weights / total


def _top_k_pick(
    candidates: list[str],
    context: frozenset[str],
    db: MatchupDB,
    mode: str,
    map_name: str,
    tier: int,
    k: int,
    min_pick_rate: float,
    rng: np.random.Generator,
) -> str:
    """
    Slow-path top-k pick: choose uniformly from the k highest-counter-rate
    brawlers in `candidates` against `context`.

    k=1 is deterministic greedy (argmax).  Falls back to uniform over
    candidates when `context` is empty (no counter signal yet).
    """
    n = len(candidates)
    if not context:
        # No opposing picks yet — uniform fallback.
        return candidates[int(rng.integers(n))]

    pick_rates = np.empty(n, dtype=np.float64)
    counter_rates = np.empty(n, dtype=np.float64)
    for i, brawler in enumerate(candidates):
        entry = db.brawler_lookup(brawler, mode, map_name, tier)
        pick_rates[i] = entry["pick_rate"] if entry else (1.0 / n)
        counter_rates[i] = _avg_counter_rate(brawler, context, db, mode, map_name, tier)

    # Mask out brawlers below min_pick_rate by pushing them to -inf.
    if min_pick_rate > 0.0:
        mask = pick_rates >= min_pick_rate
        if mask.any():
            counter_rates = np.where(mask, counter_rates, -np.inf)

    effective_k = min(k, n)
    if effective_k == 1:
        chosen = int(np.argmax(counter_rates))
    else:
        top_k_idx = np.argpartition(counter_rates, -effective_k)[-effective_k:]
        chosen = int(rng.choice(top_k_idx))
    return candidates[chosen]


# ── Precomputed weight cache ───────────────────────────────────────────────────

class RolloutWeightCache:
    """
    Precomputed numpy arrays for O(1) rollout weight computation.

    Eliminates the per-candidate Python loops in _sampling_weights by building
    a pick_rates vector and counter_mat matrix once per (mode, map_name, tier)
    context, then using numpy indexing at rollout time.

    Build cost : O(V²) dict lookups (≈9025 for V=95) — done once per context,
                 lazily on first use.  Takes ~10–50 ms.
    Query cost : a handful of numpy array ops — effectively O(1).

    Usage
    -----
        cache = RolloutWeightCache(db, vocab)
        wp = rollout(state, evaluator, db, vocab, rng=rng, weight_cache=cache)
    """

    def __init__(
        self,
        db: MatchupDB,
        vocab: tuple[str, ...],
        min_counter_games: int = DEFAULT_MIN_COUNTER_GAMES,
    ) -> None:
        self._db = db
        self._vocab = vocab
        self._V = len(vocab)
        self._min_counter_games = min_counter_games
        # Public: rollout() reads this to convert brawler names → indices.
        self.brawler_idx: dict[str, int] = {b: i for i, b in enumerate(vocab)}
        self._arrays: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        # Pre-allocated CDF buffer — avoids per-call heap allocation in sample().
        self._cdf_buf = np.empty(len(vocab), dtype=np.float64)

    def _build(self, mode: str, map_name: str, tier: int) -> tuple[np.ndarray, np.ndarray]:
        """Build pick_rates (V,) and counter_mat (V, V) for one context."""
        V = self._V
        db = self._db
        vocab = self._vocab

        pick_rates = np.full(V, 1.0 / V, dtype=np.float64)
        for i, b in enumerate(vocab):
            entry = db.brawler_lookup(b, mode, map_name, tier)
            if entry is not None:
                pick_rates[i] = entry["pick_rate"]

        counter_mat = np.full((V, V), 0.5, dtype=np.float64)
        min_n = self._min_counter_games
        for i, b in enumerate(vocab):
            for j, opp in enumerate(vocab):
                if i == j:
                    continue
                entry = db.counter_lookup(b, opp, mode, map_name, tier)
                if entry is not None and entry["n"] >= min_n:
                    counter_mat[i, j] = entry["win_rate"]

        return pick_rates, counter_mat

    def _get(self, mode: str, map_name: str, tier: int) -> tuple[np.ndarray, np.ndarray]:
        key = (mode, map_name, tier)
        if key not in self._arrays:
            self._arrays[key] = self._build(mode, map_name, tier)
        return self._arrays[key]

    def sampling_weights(
        self,
        avail_idx: np.ndarray,
        context_idx: list[int],
        counter_weight: float,
        min_pick_rate: float,
        mode: str,
        map_name: str,
        tier: int,
    ) -> np.ndarray:
        """
        Return a normalised probability vector over available brawlers.

        Parameters
        ----------
        avail_idx      : integer indices (into vocab) of legal picks
        context_idx    : integer indices (into vocab) of the opposing team
        counter_weight : blend weight (0 = pick-rate only, 1 = counter-rate only)
        min_pick_rate  : exclude brawlers with pick_rate < this (0 = no filter)
        """
        pick_rates, counter_mat = self._get(mode, map_name, tier)

        pr = pick_rates[avail_idx]                                   # (n_avail,)
        if context_idx:
            cr = counter_mat[np.ix_(avail_idx, context_idx)].mean(axis=1)  # (n_avail,)
        else:
            cr = np.full(len(avail_idx), 0.5, dtype=np.float64)

        weights = (1.0 - counter_weight) * pr + counter_weight * cr

        if min_pick_rate > 0.0:
            mask = pr >= min_pick_rate
            if mask.any():
                weights = np.where(mask, weights, 0.0)

        total = weights.sum()
        if total <= 0.0:
            weights = np.ones(len(avail_idx), dtype=np.float64)
            total = float(len(avail_idx))

        return weights / total

    def sample(
        self,
        avail_idx: np.ndarray,
        context_idx: list[int],
        counter_weight: float,
        min_pick_rate: float,
        mode: str,
        map_name: str,
        tier: int,
        rng: np.random.Generator,
    ) -> int:
        """
        Sample one vocab index from the weighted distribution over avail_idx.

        Combines weight computation and sampling in one call.  Uses a
        pre-allocated CDF buffer and rng.random() + np.searchsorted, which
        is ~5–10× faster than the numpy.random.Generator.choice(n, p=...)
        path for small n (≈90 candidates).

        Returns
        -------
        int — index into vocab of the chosen brawler
        """
        n = len(avail_idx)
        pick_rates, counter_mat = self._get(mode, map_name, tier)

        pr = pick_rates[avail_idx]
        if context_idx:
            cr = counter_mat[np.ix_(avail_idx, context_idx)].mean(axis=1)
        else:
            cr = np.full(n, 0.5, dtype=np.float64)

        weights = (1.0 - counter_weight) * pr + counter_weight * cr

        if min_pick_rate > 0.0:
            mask = pr >= min_pick_rate
            if mask.any():
                weights *= mask

        # Build CDF in pre-allocated buffer — no heap allocation.
        cdf = self._cdf_buf[:n]
        np.cumsum(weights, out=cdf)
        total = cdf[n - 1]

        if total <= 0.0:
            # All weights zeroed (pathological): uniform fallback.
            np.arange(1, n + 1, dtype=np.float64, out=cdf)
            total = float(n)

        # rng.random() is a single C-level call (~0.1 μs).
        # np.searchsorted on a length-90 array is a C binary search (~0.5 μs).
        # Together ~5–10× faster than rng.choice(n, p=probs).
        pos = int(np.searchsorted(cdf, rng.random() * total, side='right'))
        return int(avail_idx[min(pos, n - 1)])

    def top_k_pick(
        self,
        avail_idx: np.ndarray,
        context_idx: list[int],
        k: int,
        min_pick_rate: float,
        mode: str,
        map_name: str,
        tier: int,
        rng: np.random.Generator,
    ) -> int:
        """
        Fast-path top-k pick: rank available brawlers by avg counter rate
        against context_idx, then sample uniformly from the top-k.

        k=1 is deterministic greedy (argmax over counter rates).
        Falls back to weighted sampling (pure pick_rate) when context_idx
        is empty — no counter signal available at draft start.

        Returns
        -------
        int — index into vocab of the chosen brawler
        """
        if not context_idx:
            # No opposing picks yet — fall back to pick-rate-only sampling.
            return self.sample(avail_idx, context_idx, 0.0, min_pick_rate,
                               mode, map_name, tier, rng)

        n = len(avail_idx)
        pick_rates, counter_mat = self._get(mode, map_name, tier)

        pr = pick_rates[avail_idx]
        cr = counter_mat[np.ix_(avail_idx, context_idx)].mean(axis=1)  # (n,)

        # Push min_pick_rate-filtered brawlers below any real counter rate.
        if min_pick_rate > 0.0:
            mask = pr >= min_pick_rate
            if mask.any():
                cr = np.where(mask, cr, -np.inf)

        effective_k = min(k, n)
        if effective_k == 1:
            chosen_local = int(np.argmax(cr))
        else:
            top_k_local = np.argpartition(cr, -effective_k)[-effective_k:]
            chosen_local = int(rng.choice(top_k_local))

        return int(avail_idx[chosen_local])


# ── Public API ─────────────────────────────────────────────────────────────────

def rollout(
    state: DraftState,
    evaluator: FMEvaluator,
    db: MatchupDB,
    vocab: tuple[str, ...],
    *,
    my_counter_weight: float = DEFAULT_MY_COUNTER_WEIGHT,
    opp_counter_weight: float = DEFAULT_OPP_COUNTER_WEIGHT,
    min_pick_rate: float = DEFAULT_MIN_PICK_RATE,
    opp_rollout_mode: str = DEFAULT_ROLLOUT_MODE,
    opp_top_k: int = DEFAULT_TOP_K,
    my_rollout_mode: str = DEFAULT_ROLLOUT_MODE,
    my_top_k: int = DEFAULT_TOP_K,
    min_counter_games: int = DEFAULT_MIN_COUNTER_GAMES,
    rng: np.random.Generator | None = None,
    weight_cache: RolloutWeightCache | None = None,
) -> float:
    """
    Complete the draft from `state` to terminal via stochastic rollout,
    then return P(my_team wins) from the FM.

    If `state` is already terminal, the FM is called immediately with no rollout.

    Parameters
    ----------
    state              : Starting draft state (partial or terminal).
    evaluator          : FMEvaluator for terminal state scoring.
    db                 : MatchupDB for pick-rate and counter-rate lookups.
    vocab              : Full brawler vocabulary (from FeatureSchema.vocab).
    my_counter_weight  : Counter weight for my team (only used when
                         my_rollout_mode="weighted").  Default 0.3.
    opp_counter_weight : Counter weight for opponent (only used when
                         opp_rollout_mode="weighted").  Default 0.5.
    min_pick_rate      : Exclude brawlers with pick_rate below this threshold.
                         Default 0.0 (no filter).  Typical value: 0.005.
    opp_rollout_mode   : "weighted" (default) — sample by blended weight.
                         "top_k"              — sample uniformly from top-k
                         counters; k=1 is greedy/deterministic.
    opp_top_k          : k for opp "top_k" mode (default 3).
    my_rollout_mode    : "weighted" (default) or "top_k" for my team's picks.
    my_top_k           : k for my "top_k" mode (default 3).
    rng                : NumPy random generator.  Created fresh if None.
    min_counter_games  : Only use counter_rate entries with at least this many
                         games in the matchup DB.  Entries below the threshold
                         fall back to 0.5 (neutral).  Default 0 (no filter).
                         Recommended: 10–20 to suppress sparse matchup noise,
                         especially important for greedy/top_k modes where a
                         1-win-0-loss entry would otherwise dominate selection.
    weight_cache       : Optional RolloutWeightCache.  10–50× faster rollout.
                         Create once per session: RolloutWeightCache(db, vocab).
                         Pass min_counter_games to the cache constructor too.

    Returns
    -------
    float — P(my_team wins) ∈ (0, 1).
    """
    if state.is_terminal:
        return evaluator.evaluate(state)

    if rng is None:
        rng = np.random.default_rng()

    tier = (
        skill_ns_to_tier(state.skill_ns, db.skill_tier_boundaries)
        if db.skill_tier_boundaries is not None
        else 1
    )

    while not state.is_terminal:
        if weight_cache is not None:
            # Fast path: vectorized — no Python loop over candidates.
            used = state.my_team | state.opp_team | state.bans
            avail_idx = np.array(
                [i for i, b in enumerate(vocab) if b not in used], dtype=np.intp
            )
            if len(avail_idx) == 0:
                break

            if state.whose_turn == "mine":
                context_idx = [weight_cache.brawler_idx[b] for b in state.opp_team]
                if my_rollout_mode == "top_k":
                    chosen = vocab[weight_cache.top_k_pick(
                        avail_idx, context_idx, my_top_k, min_pick_rate,
                        state.mode, state.map_name, tier, rng,
                    )]
                else:
                    chosen = vocab[weight_cache.sample(
                        avail_idx, context_idx, my_counter_weight, min_pick_rate,
                        state.mode, state.map_name, tier, rng,
                    )]
            else:
                context_idx = [weight_cache.brawler_idx[b] for b in state.my_team]
                if opp_rollout_mode == "top_k":
                    chosen = vocab[weight_cache.top_k_pick(
                        avail_idx, context_idx, opp_top_k, min_pick_rate,
                        state.mode, state.map_name, tier, rng,
                    )]
                else:
                    chosen = vocab[weight_cache.sample(
                        avail_idx, context_idx, opp_counter_weight, min_pick_rate,
                        state.mode, state.map_name, tier, rng,
                    )]
        else:
            # Slow path: original per-candidate Python loops.
            candidates = available_actions(state, vocab)
            if not candidates:  # guard: should never trigger in a valid draft
                break

            if state.whose_turn == "mine":
                if my_rollout_mode == "top_k":
                    chosen = _top_k_pick(
                        candidates, state.opp_team, db,
                        state.mode, state.map_name, tier,
                        my_top_k, min_pick_rate, rng,
                    )
                else:
                    probs = _sampling_weights(
                        candidates, context=state.opp_team, db=db,
                        mode=state.mode, map_name=state.map_name, tier=tier,
                        counter_weight=my_counter_weight, min_pick_rate=min_pick_rate,
                    )
                    chosen = candidates[int(rng.choice(len(candidates), p=probs))]
            else:
                if opp_rollout_mode == "top_k":
                    chosen = _top_k_pick(
                        candidates, state.my_team, db,
                        state.mode, state.map_name, tier,
                        opp_top_k, min_pick_rate, rng,
                    )
                else:
                    probs = _sampling_weights(
                        candidates, context=state.my_team, db=db,
                        mode=state.mode, map_name=state.map_name, tier=tier,
                        counter_weight=opp_counter_weight, min_pick_rate=min_pick_rate,
                    )
                    chosen = candidates[int(rng.choice(len(candidates), p=probs))]

        state = apply_pick(state, chosen)

    return evaluator.evaluate(state)


# ── Sanity checks ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Step 2.4 Sanity Checks ===\n")

    evaluator = FMEvaluator.load()
    schema = evaluator._fm.schema
    vocab = schema.vocab
    db = MatchupDB.load()

    _MAP  = schema.maps[0]
    _MODE = schema.modes[0]

    partial = DraftState(
        my_team=frozenset({vocab[0]}),
        opp_team=frozenset({vocab[3]}),
        mode=_MODE,
        map_name=_MAP,
        skill_ns=1.0,
        is_first_pick=True,
    )

    # ── Check 1: rollout completes draft and returns probability in (0, 1) ──────
    rng = np.random.default_rng(42)
    result = rollout(partial, evaluator, db, vocab, rng=rng)
    assert 0.0 < result < 1.0, f"Expected probability in (0,1), got {result}"
    print(f"Check 1: single rollout → win_prob = {result:.4f}")
    print(f"  starting state: my={sorted(partial.my_team)}, opp={sorted(partial.opp_team)}")
    print("  ✓\n")

    # ── Check 2: 10 rollouts produce variance (sampler is stochastic) ─────────
    rng = np.random.default_rng(0)
    scores = [rollout(partial, evaluator, db, vocab, rng=rng) for _ in range(10)]
    std = float(np.std(scores))
    print("Check 2: 10 rollouts produce varied FM scores (std > 0)")
    for i, s in enumerate(scores):
        print(f"  rollout {i+1:2d}: {s:.4f}")
    print(f"  std = {std:.5f}")
    assert std > 0.0, "All rollouts returned the same score — rollout sampler is broken"
    print("  ✓\n")

    # ── Check 3: adversarial opp (high opp_counter_weight) runs without error ──
    rng = np.random.default_rng(1)
    adv_scores = [
        rollout(partial, evaluator, db, vocab, opp_counter_weight=0.9, rng=rng)
        for _ in range(5)
    ]
    print(f"Check 3: adversarial opp (w=0.9) scores: {[f'{s:.4f}' for s in adv_scores]}")
    assert all(0.0 < s < 1.0 for s in adv_scores)
    print("  ✓\n")

    # ── Check 4: coordinated team (high my_counter_weight) runs without error ──
    rng = np.random.default_rng(2)
    coord_scores = [
        rollout(partial, evaluator, db, vocab, my_counter_weight=0.7, rng=rng)
        for _ in range(5)
    ]
    print(f"Check 4: coordinated team (w=0.7) scores: {[f'{s:.4f}' for s in coord_scores]}")
    assert all(0.0 < s < 1.0 for s in coord_scores)
    print("  ✓\n")

    # ── Check 5: min_pick_rate filter does not crash or exclude all candidates ──
    rng = np.random.default_rng(3)
    filtered_scores = [
        rollout(partial, evaluator, db, vocab, min_pick_rate=0.005, rng=rng)
        for _ in range(5)
    ]
    print(f"Check 5: min_pick_rate=0.005 scores: {[f'{s:.4f}' for s in filtered_scores]}")
    assert all(0.0 < s < 1.0 for s in filtered_scores)
    print("  ✓\n")

    # ── Check 6: terminal state skips rollout; second call hits cache ──────────
    terminal = DraftState(
        my_team=frozenset(vocab[:3]),
        opp_team=frozenset(vocab[3:6]),
        mode=_MODE,
        map_name=_MAP,
        skill_ns=1.0,
        is_first_pick=True,
    )
    hits_before = evaluator.cache_stats["hits"]
    _ = rollout(terminal, evaluator, db, vocab)
    _ = rollout(terminal, evaluator, db, vocab)
    hits_after = evaluator.cache_stats["hits"]
    assert hits_after > hits_before, "Expected cache hit on second terminal rollout"
    print("Check 6: terminal state goes straight to FM; second call hits cache  ✓\n")

    print("=== All checks passed ===")
