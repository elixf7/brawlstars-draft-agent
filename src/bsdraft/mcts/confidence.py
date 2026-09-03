"""
confidence.py — Step 2.6: Confidence Metrics

Two-layer confidence system for MCTS draft recommendations.

──────────────────────────────────────────────────────────────────────────
Layer 1 — Empirical data confidence (matchup DB sample sizes)

  The counter matrix stores {n, wins, win_rate} for every (my_brawler,
  opp_brawler) matchup.  A full 3v3 state has 9 cross-team pairs; a partial
  state has fewer.  If a pair rarely co-occurs in the data (e.g., one brawler
  is almost always banned on this map, or the combo is unusual at this tier),
  n is small and the stored win rate is noisy — the matchup DB is uncertain
  there, regardless of what the FM or MCTS says.

  We summarise coverage across all cross-team pairs via the harmonic mean of
  their sample sizes.  The harmonic mean is conservative: a single pair with
  n=10 pulls the whole number down even if the other 8 pairs each have n=5000.
  This is exactly right — one uncertain matchup undermines the rollout.

  95% CI half-width for a Beta proportion at p ≈ 0.5:
      n = 200  →  ± 7%   (threshold for "High")
      n = 50   →  ± 14%  (threshold for "Medium")
      n < 50   →  noisy  ("Low")

  Output includes the raw harmonic_n and the per-pair breakdown so the user
  can see exactly which brawler pair is the data bottleneck.

  Note on bans and empirical confidence: brawlers that are frequently banned
  appear less often in the matchup data than their real value would suggest.
  This shows up as low n for pairs involving those brawlers — the confidence
  layer surfaces this correctly rather than hiding it.

──────────────────────────────────────────────────────────────────────────
Layer 2 — MCTS convergence confidence (visit fraction distribution)

  After N simulations the top pick is the most-visited child of the root.
  Its visit fraction = visit_count(best_child) / N.  If one pick clearly
  dominates, visit counts concentrate — "dominant pick" signal.  If several
  picks are genuinely equivalent, visits spread — that's correct behavior,
  not a failure.

  We surface the top-N picks (default 5) with raw visit counts, visit
  fractions, and Q-values so the user can see the data directly.

  Visit fraction thresholds (applied to the top pick):
      ≥ 0.40  →  "Dominant pick"
      ≥ 0.25  →  "Strong pick"
      ≥ 0.15  →  "Solid pick"
      < 0.15  →  "Even matchup — multiple viable options"

──────────────────────────────────────────────────────────────────────────
Ban calibration utility

  sample_random_bans() draws a ban set weighted by pick rate for calibration
  and stress-testing the MCTS pipeline.  In ranked, brawlers with high pick
  rates tend to be targeted in bans; this is a plausible proxy.  Not used in
  production — call it when constructing test DraftStates.
"""

from __future__ import annotations

import numpy as np

from bsdraft.data.matchup_db import MatchupDB, skill_ns_to_tier
from bsdraft.mcts.node import MCTSNode
from bsdraft.mcts.state import DraftState

# ── Thresholds ────────────────────────────────────────────────────────────────

N_MIN_HIGH: int = 200    # harmonic_n ≥ 200 → ±7% 95% CI
N_MIN_MEDIUM: int = 50   # harmonic_n ≥ 50  → ±14% 95% CI

# Layer 2 convergence labels use *relative* visit fraction = actual / (1/n_children).
# This adapts to branching factor automatically.
#
# With ~95 brawlers at first pick, the random baseline is 1/95 ≈ 1.05%.  Asking
# for "40% of visits" is mathematically impossible at any reasonable simulation
# budget — the UCB1 exploration bonus is ~1.07 per visit when C=0.5, far larger
# than Q differences of ~0.1.  Absolute thresholds only apply when there are few
# choices (late draft with ≤ ~12 remaining brawlers).
#
# Formula: effective_threshold = min(absolute_threshold, relative_factor / n_children)
# For n=95: dominant=5.3%, strong=3.2%, solid=2.1% — achievable at ~5,000 sims.
# For n=5:  dominant=min(40%, 100%)=40%, strong=25%, solid=15% — absolute wins.
DOMINANT_ABS: float = 0.40   # absolute cap (applies when n < ~12)
STRONG_ABS:   float = 0.25
SOLID_ABS:    float = 0.15

DOMINANT_REL: float = 5.0    # ×random → "dominant" (top pick has 5× expected share)
STRONG_REL:   float = 3.0    # ×random
SOLID_REL:    float = 2.0    # ×random


# ── Layer 1: Empirical data confidence ───────────────────────────────────────

def layer1_confidence(state: DraftState, db: MatchupDB) -> dict:
    """
    Compute empirical data confidence for the cross-team pairs in `state`.

    Looks up the counter matrix for every (my_brawler, opp_brawler) pair
    currently in the draft and summarises sample sizes via harmonic mean.

    Works at any draft depth.  At 0v0 or 1v0 there are no cross-team pairs
    yet — returns confidence_label "N/A".  Most meaningful at terminal (all 9
    pairs checked), but valid and useful at any partial depth.

    Parameters
    ----------
    state : current draft state (partial or terminal)
    db    : loaded MatchupDB

    Returns
    -------
    dict:
      harmonic_n        float | None  harmonic mean of pair sample sizes
      n_pairs           int           number of cross-team pairs checked
      confidence_label  str           "High", "Medium", "Low", or "N/A"
      pairs             list[dict]    per-pair detail:
                          my_brawler    str
                          opp_brawler   str
                          n             int   sample size from matchup DB
                          win_rate      float | None  P(my team wins | pair)
                          fallback_level int | None   0=map+tier … 3=global
    """
    tier = (
        skill_ns_to_tier(state.skill_ns, db.skill_tier_boundaries)
        if db.skill_tier_boundaries is not None
        else 1
    )

    pairs = []
    for my_b in sorted(state.my_team):
        for opp_b in sorted(state.opp_team):
            entry = db.counter_lookup(
                my_b, opp_b, state.mode, state.map_name, tier
            )
            pairs.append({
                "my_brawler":     my_b,
                "opp_brawler":    opp_b,
                "n":              entry["n"] if entry else 0,
                "win_rate":       entry["win_rate"] if entry else None,
                "fallback_level": entry["level"] if entry else None,
            })

    if not pairs:
        return {
            "harmonic_n":       None,
            "n_pairs":          0,
            "confidence_label": "N/A",
            "pairs":            [],
        }

    # Harmonic mean: any n=0 collapses the whole thing to 0.
    ns = [p["n"] for p in pairs]
    if any(n == 0 for n in ns):
        harmonic_n = 0.0
    else:
        harmonic_n = len(ns) / sum(1.0 / n for n in ns)

    if harmonic_n >= N_MIN_HIGH:
        label = "High"
    elif harmonic_n >= N_MIN_MEDIUM:
        label = "Medium"
    else:
        label = "Low"

    return {
        "harmonic_n":       harmonic_n,
        "n_pairs":          len(pairs),
        "confidence_label": label,
        "pairs":            pairs,
    }


# ── Layer 2: MCTS convergence confidence ─────────────────────────────────────

def layer2_confidence(root: MCTSNode, n_top: int = 5) -> dict:
    """
    Compute MCTS convergence confidence from the visit distribution at `root`.

    Sorts all expanded children by visit count and returns the top-N with
    their raw counts, visit fractions, and Q-values.  The confidence label
    reflects how much the tree has converged on one pick vs. spread visits
    across many roughly-equivalent options.

    Parameters
    ----------
    root  : MCTS root node after simulations have been run
    n_top : number of top picks to surface (default 5)

    Returns
    -------
    dict:
      total_simulations  int         root.visit_count
      confidence_label   str         based on the top pick's visit fraction
      top_picks          list[dict]  up to n_top entries:
                           brawler            str    brawler name (action)
                           visit_count        int
                           visit_fraction     float  visit_count / total
                           estimated_win_prob float  Q = value_sum / visit_count

    Raises
    ------
    ValueError  if root has no expanded children or visit_count == 0
    """
    if not root.children:
        raise ValueError(
            "layer2_confidence: root has no expanded children. "
            "Run MCTS simulations before computing confidence."
        )
    if root.visit_count == 0:
        raise ValueError(
            "layer2_confidence: root.visit_count == 0. "
            "Root must be visited at least once."
        )

    total = root.visit_count
    # n_legal_actions accounts for lazy expansion: len(root.children) only counts
    # children created so far, but the random baseline needs the full legal count.
    n_children = root.n_legal_actions
    ranked = sorted(
        root.children.items(),
        key=lambda kv: kv[1].visit_count,
        reverse=True,
    )

    top_fraction = ranked[0][1].visit_count / total

    # Adaptive thresholds: min(absolute, relative/n_children).
    # With ~95 brawlers the absolute thresholds (40/25/15%) are unreachable at any
    # practical simulation budget — UCB1 exploration keeps all children in contention.
    # The relative thresholds (5×/3×/2× the random share of 1/n) lower the bar to
    # something achievable (~5k sims for a clearly dominant brawler at 95 choices).
    # When n is small (≤ 12 remaining choices) the absolute thresholds take over.
    random_frac = 1.0 / n_children if n_children > 0 else 1.0
    dom_thresh   = min(DOMINANT_ABS, DOMINANT_REL * random_frac)
    strong_thresh = min(STRONG_ABS,  STRONG_REL   * random_frac)
    solid_thresh  = min(SOLID_ABS,   SOLID_REL    * random_frac)

    if top_fraction >= dom_thresh:
        label = "Dominant pick"
    elif top_fraction >= strong_thresh:
        label = "Strong pick"
    elif top_fraction >= solid_thresh:
        label = "Solid pick"
    else:
        label = "Even matchup — multiple viable options"

    root_q = root.q_value  # expected win rate from the current state (tree average)

    top_picks = [
        {
            "brawler":                brawler,
            "visit_count":            node.visit_count,
            "visit_fraction":         node.visit_count / total,
            # How many times the random-baseline share (1/n_children) this pick has.
            # > 1.0 = visited more than random; the label thresholds use 2×/3×/5×.
            "relative_visit_fraction": (node.visit_count / total) / random_frac,
            "estimated_win_prob":     node.q_value,
            "win_prob_delta":         node.q_value - root_q,
        }
        for brawler, node in ranked[:n_top]
    ]

    return {
        "total_simulations": total,
        "n_available":       n_children,    # branching factor at root
        "random_fraction":   random_frac,   # expected share per child if visits were uniform
        "root_q":            root_q,        # baseline: expected win rate before committing
        "confidence_label":  label,
        "top_picks":         top_picks,
    }


# ── Ban calibration utility ───────────────────────────────────────────────────

def sample_random_bans(
    db: MatchupDB,
    vocab: tuple[str, ...],
    n_bans: int = 6,
    *,
    mode: str = "",
    map_name: str = "",
    tier: int = 1,
    rng: np.random.Generator | None = None,
) -> frozenset[str]:
    """
    Sample a realistic ban set for calibration and stress-testing.

    Draws n_bans brawlers without replacement, weighted by pick rate.
    High pick rate ≈ high ban rate — popular brawlers are typically targeted
    in the ban phase.  The MatchupDB fallback chain handles context gracefully:
    passing empty mode/map_name falls through to the global pick rate.

    Parameters
    ----------
    db       : loaded MatchupDB
    vocab    : full brawler vocabulary (e.g. fm.schema.vocab)
    n_bans   : brawlers to ban (default 6: 3 per team in ranked)
    mode     : game mode string for contextual pick rates (empty = global)
    map_name : map name (empty = global)
    tier     : skill tier 0–3 (default 1 = lower-mid)
    rng      : NumPy random generator (created fresh if None)

    Returns
    -------
    frozenset[str] — banned brawler names, suitable for DraftState.bans
    """
    if rng is None:
        rng = np.random.default_rng()

    weights = np.empty(len(vocab), dtype=np.float64)
    for i, brawler in enumerate(vocab):
        entry = db.brawler_lookup(brawler, mode, map_name, tier)
        weights[i] = entry["pick_rate"] if entry else (1.0 / len(vocab))

    total = weights.sum()
    if total <= 0.0:
        weights[:] = 1.0 / len(vocab)
    else:
        weights /= total

    n_bans = min(n_bans, len(vocab))
    indices = rng.choice(len(vocab), size=n_bans, replace=False, p=weights)
    return frozenset(vocab[i] for i in indices)


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_layer1(result: dict) -> str:
    """
    Compact one-line summary of a layer1_confidence result.

    Example:
      Layer 1 | High confidence (harmonic n=312, 9 pairs)
      Layer 1 | Medium confidence (harmonic n=87, 4 pairs) | bottleneck: POCO vs MORTIS (n=23)
    """
    if result["harmonic_n"] is None:
        return "Layer 1 | N/A (no cross-team pairs yet)"

    summary = (
        f"Layer 1 | {result['confidence_label']} confidence "
        f"(harmonic n={result['harmonic_n']:.0f}, {result['n_pairs']} pairs)"
    )
    if result["pairs"]:
        bottleneck = min(result["pairs"], key=lambda p: p["n"])
        if bottleneck["n"] < N_MIN_HIGH:
            summary += (
                f" | bottleneck: {bottleneck['my_brawler']} vs "
                f"{bottleneck['opp_brawler']} (n={bottleneck['n']})"
            )
    return summary


def format_layer2(result: dict) -> str:
    """
    Multi-line summary of a layer2_confidence result.

    Example:
      Layer 2 | Strong pick | 5000 sims | 89 choices | random=1.1%  (root Q: 0.503)
        1. MORTIS      visits=  182 (3.6%  3.3×rand)  win_prob=0.587  delta=+0.084
        2. CROW        visits=  124 (2.5%  2.2×rand)  win_prob=0.561  delta=+0.058
        ...
    """
    root_q   = result.get("root_q")
    rand_frac = result.get("random_fraction", 0.0)
    n_avail  = result.get("n_available", "?")
    baseline_str = f"  (root Q: {root_q:.3f})" if root_q is not None else ""
    lines = [
        f"Layer 2 | {result['confidence_label']} | "
        f"{result['total_simulations']} sims | "
        f"{n_avail} choices | random={rand_frac:.1%}"
        f"{baseline_str}"
    ]
    for rank, pick in enumerate(result["top_picks"], start=1):
        if pick["visit_count"] > 0:
            wr_str    = f"{pick['estimated_win_prob']:.3f}"
            delta_str = f"{pick['win_prob_delta']:+.3f}"
            rel_str   = f"{pick['relative_visit_fraction']:.1f}×rand"
        else:
            wr_str = delta_str = rel_str = "N/A"
        lines.append(
            f"  {rank}. {pick['brawler']:<12s} "
            f"visits={pick['visit_count']:>5d} "
            f"({pick['visit_fraction']:>5.1%} {rel_str:>8})  "
            f"win_prob={wr_str}  delta={delta_str}"
        )
    return "\n".join(lines)


# ── Sanity checks (run: python src/confidence.py) ────────────────────────────

if __name__ == "__main__":
    from bsdraft.mcts.evaluator import FMEvaluator
    from bsdraft.mcts.node import backpropagate, select
    from bsdraft.mcts.rollout import rollout

    print("=== Step 2.6 Sanity Checks ===\n")

    evaluator = FMEvaluator.load()
    schema    = evaluator._fm.schema
    vocab     = schema.vocab
    db        = MatchupDB.load()

    _MAP  = schema.maps[0]
    _MODE = schema.modes[0]
    _TIER = skill_ns_to_tier(1.0, db.skill_tier_boundaries) if db.skill_tier_boundaries else 1

    # ── Check 1: Layer 1 — N/A at draft start (no cross-team pairs) ──────────
    empty_state = DraftState(
        my_team=frozenset(), opp_team=frozenset(),
        mode=_MODE, map_name=_MAP, skill_ns=1.0, is_first_pick=True,
    )
    r1 = layer1_confidence(empty_state, db)
    assert r1["harmonic_n"] is None
    assert r1["confidence_label"] == "N/A"
    print("Check 1: Layer 1 at draft start")
    print(f"  {format_layer1(r1)}  ✓\n")

    # ── Check 2: Layer 1 — terminal state ────────────────────────────────────
    terminal = DraftState(
        my_team=frozenset(vocab[:3]),
        opp_team=frozenset(vocab[3:6]),
        mode=_MODE, map_name=_MAP, skill_ns=1.0, is_first_pick=True,
    )
    r2 = layer1_confidence(terminal, db)
    assert r2["n_pairs"] == 9, f"Expected 9 pairs, got {r2['n_pairs']}"
    assert r2["harmonic_n"] is not None
    assert r2["confidence_label"] in {"High", "Medium", "Low"}
    print(f"Check 2: Layer 1 at terminal state ({r2['n_pairs']} cross-team pairs)")
    print(f"  {format_layer1(r2)}")
    print("  Per-pair breakdown:")
    for p in r2["pairs"]:
        wr = f"{p['win_rate']:.3f}" if p["win_rate"] is not None else " N/A"
        print(
            f"    {p['my_brawler']:<12s} vs {p['opp_brawler']:<12s} "
            f"n={p['n']:>6}  win_rate={wr}  level={p['fallback_level']}"
        )
    print("  ✓\n")

    # ── Check 3: Layer 1 — partial state (1 pick each = 1 cross-team pair) ───
    partial = DraftState(
        my_team=frozenset({vocab[0]}),
        opp_team=frozenset({vocab[3]}),
        mode=_MODE, map_name=_MAP, skill_ns=1.0, is_first_pick=True,
    )
    r3_l1 = layer1_confidence(partial, db)
    assert r3_l1["n_pairs"] == 1
    print("Check 3: Layer 1 at partial state (1 pick each)")
    print(f"  {format_layer1(r3_l1)}  ✓\n")

    # ── Check 4: Layer 2 — run MCTS and check top-5 structure ────────────────
    def _run_mcts(state: DraftState, n_sims: int, seed: int = 42) -> MCTSNode:
        """Minimal MCTS loop for sanity checking."""
        root = MCTSNode(state=state)
        root.expand(vocab)
        rng_mc = np.random.default_rng(seed)
        for _ in range(n_sims):
            path = select(root)
            leaf = path[-1]
            if not leaf.state.is_terminal and leaf.is_leaf:
                leaf.expand(vocab)
                child_action, child = next(iter(leaf.children.items()))
                path.append(child)
                leaf = child
            win_prob = rollout(leaf.state, evaluator, db, vocab, rng=rng_mc)
            backpropagate(path, win_prob)
        return root

    root_state = DraftState(
        my_team=frozenset(), opp_team=frozenset(),
        mode=_MODE, map_name=_MAP, skill_ns=1.0, is_first_pick=True,
    )

    N_SIMS = 200
    root = _run_mcts(root_state, N_SIMS, seed=42)

    r4 = layer2_confidence(root, n_top=5)
    assert r4["total_simulations"] == N_SIMS, (
        f"Expected {N_SIMS} sims, got {r4['total_simulations']}"
    )
    assert 1 <= len(r4["top_picks"]) <= 5
    assert r4["confidence_label"] in {
        "Dominant pick", "Strong pick", "Solid pick",
        "Even matchup — multiple viable options"
    }
    total_frac = sum(p["visit_fraction"] for p in r4["top_picks"])
    assert total_frac <= 1.0 + 1e-9, f"Visit fractions sum > 1: {total_frac:.4f}"

    print(f"Check 4: Layer 2 after {N_SIMS} simulations")
    print(format_layer2(r4))
    print("  ✓\n")

    # ── Check 5: Layer 2 confidence changes between 50 and 500 sims ──────────
    root_50  = _run_mcts(root_state, 50,  seed=0)
    root_500 = _run_mcts(root_state, 500, seed=0)

    r_50  = layer2_confidence(root_50,  n_top=5)
    r_500 = layer2_confidence(root_500, n_top=5)

    frac_50  = r_50["top_picks"][0]["visit_fraction"]
    frac_500 = r_500["top_picks"][0]["visit_fraction"]
    print("Check 5: top-pick visit fraction comparison")
    print(f"  50 sims:  {frac_50:.3f}  ({r_50['confidence_label']})")
    print(f"  500 sims: {frac_500:.3f}  ({r_500['confidence_label']})")
    # Not guaranteed to increase if no dominant pick exists — log either way.
    direction = "increased" if frac_500 >= frac_50 else "decreased (no dominant pick on this map)"
    print(f"  visit fraction {direction}")
    print("  ✓\n")

    # ── Check 6: sample_random_bans structure ────────────────────────────────
    rng_ban = np.random.default_rng(7)
    bans = sample_random_bans(db, vocab, n_bans=6, mode=_MODE, map_name=_MAP, tier=_TIER, rng=rng_ban)
    assert isinstance(bans, frozenset)
    assert len(bans) == 6
    assert all(b in set(vocab) for b in bans)
    print(f"Check 6: sample_random_bans(n=6) → {sorted(bans)}")
    print("  6 unique brawlers from vocab  ✓\n")

    # ── Check 7: bans are respected in draft state + MCTS runs cleanly ────────
    banned_state = DraftState(
        my_team=frozenset(), opp_team=frozenset(),
        mode=_MODE, map_name=_MAP, skill_ns=1.0, is_first_pick=True,
        bans=bans,
    )
    root_banned = _run_mcts(banned_state, 50, seed=1)
    r7 = layer2_confidence(root_banned, n_top=5)
    # No banned brawler should appear in top picks
    assert not (set(p["brawler"] for p in r7["top_picks"]) & bans), (
        "A banned brawler appeared in top picks"
    )
    print(f"Check 7: MCTS with {len(bans)} bans — top picks exclude all banned brawlers")
    print(format_layer2(r7))
    print("  ✓\n")

    print("=== All checks passed ===")
