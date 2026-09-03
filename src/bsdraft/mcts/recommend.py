"""
recommend.py — Step 2.7: Integration & Interface

Top-level recommend() function that runs MCTS from a partial draft state and
returns a RecommendResult containing ranked pick recommendations, win probability
estimates, and both confidence layers.

Usage
-----
    from bsdraft.mcts.recommend import recommend

    result = recommend(
        my_picks      = ["MORTIS"],
        opp_picks     = ["BELLE"],
        mode          = "gemGrab",
        map_name      = "Double Swoosh",
        skill_ns      = 1.2,
        is_first_pick = True,   # required — coin flip at match start
    )
    print(result.summary())

"Whose turn" derivation
-----------------------
Pick order in Brawl Stars ranked:
    P1 (first-pick team):  [mine, opp, opp, mine, mine, opp]
    P2 (second-pick team): [opp, mine, mine, opp, opp, mine]

P1 vs P2 is determined by a coin flip — the caller MUST pass is_first_pick.
whose_turn is derived internally from pick_number = |my_picks| + |opp_picks|.

When it is the opponent's turn, recommend() still runs: MCTS models the
opponent's optimal pick. result.whose_turn == 'opp' signals that the top_picks
list is a prediction of what the opponent will select, not a pick recommendation
for you.

Pre-loading resources
---------------------
For repeated calls in a session (e.g. across multiple draft picks), pass
pre-loaded evaluator and db to avoid reloading from disk each call:

    evaluator = FMEvaluator.load()
    db = MatchupDB.load()
    result1 = recommend(..., evaluator=evaluator, db=db)
    result2 = recommend(..., evaluator=evaluator, db=db)
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from bsdraft.data.matchup_db import MatchupDB, skill_ns_to_tier
from bsdraft.mcts.confidence import (
    format_layer1,
    format_layer2,
    layer1_confidence,
    layer2_confidence,
)
from bsdraft.mcts.evaluator import FMEvaluator
from bsdraft.mcts.node import (
    UCB1_C,
    MCTSNode,
    backpropagate,
    select,
    select_child,
)
from bsdraft.mcts.rollout import (
    DEFAULT_MIN_COUNTER_GAMES,
    DEFAULT_MIN_PICK_RATE,
    DEFAULT_MY_COUNTER_WEIGHT,
    DEFAULT_OPP_COUNTER_WEIGHT,
    DEFAULT_ROLLOUT_MODE,
    DEFAULT_TOP_K,
    RolloutWeightCache,
    rollout,
)
from bsdraft.mcts.state import DraftState, available_actions
from bsdraft.selfplay.joint_net import JointNetInference
from bsdraft.selfplay.policy_net import PolicyInference

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RecommendResult:
    """
    Output of recommend().

    Fields
    ------
    state       : DraftState used as the MCTS root.
    whose_turn  : 'mine'  — top_picks is a recommendation for my next pick.
                  'opp'   — top_picks is MCTS's prediction of the opponent's pick.
    layer1      : dict from layer1_confidence() — empirical matchup DB coverage.
    layer2      : dict from layer2_confidence() — MCTS visit-count convergence.
    elapsed_sec : Wall time for the simulation run.
    """

    state: DraftState
    whose_turn: str
    layer1: dict
    layer2: dict
    elapsed_sec: float

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def top_picks(self) -> list[dict]:
        """Shortcut to layer2['top_picks'] — ranked candidate brawlers."""
        return self.layer2["top_picks"]

    @property
    def best(self) -> str:
        """The single top-ranked brawler by visit count."""
        return self.layer2["top_picks"][0]["brawler"]

    # ── Formatting ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """
        Human-readable multi-line summary.

        Example output (my turn):
            Draft: my=[] opp=[] | gemGrab / Double Swoosh | P1 | pick 0 → my pick
            Layer 1 | N/A (no cross-team pairs yet)
            Layer 2 | Strong pick | 1000 sims | 89 choices | random=1.1%  (root Q: 0.504)
              1. CROW         visits=   55 ( 5.5%  5.2×rand)  win_prob=0.551  delta=+0.047
              2. MORTIS       visits=   41 ( 4.1%  3.9×rand)  win_prob=0.538  delta=+0.034
            [1.24s]
        """
        p1p2 = "P1" if self.state.is_first_pick else "P2"
        turn_label = "my pick" if self.whose_turn == "mine" else "opponent's predicted pick"
        bans_str = f" | bans={sorted(self.state.bans)}" if self.state.bans else ""
        header = (
            f"Draft: my={sorted(self.state.my_team)} opp={sorted(self.state.opp_team)}"
            f" | {self.state.mode} / {self.state.map_name}"
            f" | {p1p2}{bans_str}"
            f" | pick {self.state.pick_number} → {turn_label}"
        )
        return "\n".join([
            header,
            format_layer1(self.layer1),
            format_layer2(self.layer2),
            f"[{self.elapsed_sec:.2f}s]",
        ])


# ── PUCT prior helper ─────────────────────────────────────────────────────────

def _compute_puct_priors(
    state: DraftState,
    expand_vocab: tuple[str, ...],
    weight_cache: RolloutWeightCache,
    puct_alpha: float,
    min_pick_rate: float,
    tier: int,
    policy: PolicyInference | None = None,
) -> np.ndarray | None:
    """
    Compute normalized prior weights for an opponent-turn node expansion (Step 3.2.1).

    When policy is provided, uses the trained policy network's predicted pick
    distribution as the prior (policy takes precedence over counter-rate prior).

    When policy is None, falls back to the counter-rate blended prior:
      prior(b) = (1 - alpha) * pick_rate(b) + alpha * avg_counter_rate(b, against=my_team)

    The blended formula keeps the prior grounded in empirical pick popularity so
    rare brawlers with noisy counter-rate estimates don't dominate exploration.
    Normalization is handled by RolloutWeightCache.sampling_weights().

    Returns None if the node has no available actions (should not occur in valid draft).
    """
    actions = available_actions(state, expand_vocab)
    if not actions:
        return None
    if policy is not None:
        return policy.predict_prior(state, actions)
    avail_idx = np.array(
        [weight_cache.brawler_idx[b] for b in actions], dtype=np.intp
    )
    context_idx = [weight_cache.brawler_idx[b] for b in state.my_team]
    return weight_cache.sampling_weights(
        avail_idx, context_idx, puct_alpha, min_pick_rate,
        state.mode, state.map_name, tier,
    )


# ── Internal MCTS runner ──────────────────────────────────────────────────────

def _run_mcts(
    state: DraftState,
    evaluator: FMEvaluator,
    db: MatchupDB,
    vocab: tuple[str, ...],
    n_simulations: int,
    my_counter_weight: float,
    opp_counter_weight: float,
    min_pick_rate: float,
    opp_rollout_mode: str,
    opp_top_k: int,
    my_rollout_mode: str,
    my_top_k: int,
    min_counter_games: int,
    c: float,
    rng: np.random.Generator,
    puct_alpha: float = 0.7,
    tree_vocab: tuple[str, ...] | None = None,
    weight_cache: RolloutWeightCache | None = None,
    policy: PolicyInference | None = None,
    value_net: JointNetInference | None = None,
) -> MCTSNode:
    """
    Run `n_simulations` MCTS iterations from `state` and return the root node.

    Each iteration:
      1. select()        — traverse root → leaf via UCB1 (my-turn) or PUCT (opp-turn)
      2. expand()        — if leaf is non-terminal, add all legal children
      3. evaluate        — value_net.evaluate() if provided, else rollout() to terminal
      4. backpropagate() — update visit counts and value sums up to root

    Uses RolloutWeightCache so rollout weight computation is vectorized
    rather than per-candidate Python loops.

    Parameters
    ----------
    puct_alpha : prior blend weight for opponent-turn nodes (Step 3.2).
        prior(b) = (1-alpha)*pick_rate(b) + alpha*avg_counter_rate(b, against=my_team).
        0.0 disables PUCT (UCB1 everywhere); 1.0 makes priors counter-rate-only.
        Default 0.7.
    tree_vocab : Optional filtered vocabulary for tree expansion.  When
        provided, only brawlers in tree_vocab are added as children during
        expand() — effectively removing low-pick-rate brawlers from the
        MCTS tree so sims are not wasted on them.  The full vocab is still
        used for the RolloutWeightCache and rollout() so that counter-rate
        lookups on already-picked brawlers always succeed.
    weight_cache : Optional pre-built RolloutWeightCache.  When provided
        (e.g. from self-play where the same map/mode/tier is used for all
        6 picks in a game), the O(V²) matrix construction is skipped,
        saving ~50-300ms per game.  Must be built with the full vocab and
        the same min_counter_games as this call.  Built fresh if None.
    value_net : Optional JointNetInference.  When provided, replaces rollout()
        entirely: every leaf is evaluated directly via value_net.evaluate()
        without simulating any remaining picks.  This eliminates rollout
        variance and allows the tree to converge with fewer simulations.
        Pass the same JointNetInference as both `policy` and `value_net`
        to use learned priors and rollout-free evaluation together.
    """
    _expand_vocab = tree_vocab if tree_vocab is not None else vocab

    # Build rollout weight cache — reused for rollout() and PUCT prior computation.
    # When called from self-play, a pre-warmed cache is passed to avoid
    # rebuilding the O(V²) counter matrix for every pick in the same game.
    if weight_cache is None:
        weight_cache = RolloutWeightCache(db, vocab, min_counter_games=min_counter_games)

    # Precompute skill tier once; constant for the whole draft.
    _tier = (
        skill_ns_to_tier(state.skill_ns, db.skill_tier_boundaries)
        if db.skill_tier_boundaries is not None
        else 1
    )

    def _expand_node(node: MCTSNode) -> None:
        """Expand node, attaching PUCT priors at opponent-turn nodes."""
        pw: np.ndarray | None = None
        if node.state.whose_turn == "opp" and (policy is not None or puct_alpha > 0.0):
            pw = _compute_puct_priors(
                node.state, _expand_vocab, weight_cache, puct_alpha, min_pick_rate, _tier,
                policy=policy,
            )
        node.expand(_expand_vocab, prior_weights=pw)

    root = MCTSNode(state=state)
    _expand_node(root)

    for _ in range(n_simulations):
        path = select(root, c=c)
        leaf = path[-1]

        # Expand if the leaf is non-terminal and not yet expanded.
        # expand() is lazy: it stores available actions but creates no children.
        # select_child() then creates exactly one child on demand (one apply_pick),
        # rather than the old approach of eagerly creating all ~90 children.
        if not leaf.state.is_terminal and leaf.is_leaf:
            _expand_node(leaf)
            if not leaf.is_leaf:
                # expand() succeeded — create the first child lazily.
                _, child = select_child(leaf, c=c)
                path.append(child)
                leaf = child

        if value_net is not None:
            # Rollout-free: directly evaluate the leaf with the value network.
            # No random picks needed — the value head returns P(my_team wins)
            # for any partial (or terminal) state.
            win_prob = value_net.evaluate(leaf.state)
        else:
            win_prob = rollout(
                leaf.state,
                evaluator,
                db,
                vocab,
                my_counter_weight=my_counter_weight,
                opp_counter_weight=opp_counter_weight,
                min_pick_rate=min_pick_rate,
                opp_rollout_mode=opp_rollout_mode,
                opp_top_k=opp_top_k,
                my_rollout_mode=my_rollout_mode,
                my_top_k=my_top_k,
                min_counter_games=min_counter_games,
                rng=rng,
                weight_cache=weight_cache,
            )
        backpropagate(path, win_prob)

    return root


# ── Public API ────────────────────────────────────────────────────────────────

def recommend(
    my_picks: Iterable[str],
    opp_picks: Iterable[str],
    mode: str,
    map_name: str,
    skill_ns: float,
    *,
    is_first_pick: bool,
    bans: Iterable[str] = (),
    n_simulations: int = 1000,
    n_top: int = 5,
    my_counter_weight: float = DEFAULT_MY_COUNTER_WEIGHT,
    opp_counter_weight: float = DEFAULT_OPP_COUNTER_WEIGHT,
    min_pick_rate: float = DEFAULT_MIN_PICK_RATE,
    opp_rollout_mode: str = DEFAULT_ROLLOUT_MODE,
    opp_top_k: int = DEFAULT_TOP_K,
    my_rollout_mode: str = DEFAULT_ROLLOUT_MODE,
    my_top_k: int = DEFAULT_TOP_K,
    min_counter_games: int = DEFAULT_MIN_COUNTER_GAMES,
    ucb1_c: float = UCB1_C,
    puct_alpha: float = 0.7,
    evaluator: FMEvaluator | None = None,
    db: MatchupDB | None = None,
    rng: np.random.Generator | None = None,
    policy: PolicyInference | None = None,
    value_net: JointNetInference | None = None,
) -> RecommendResult:
    """
    Run MCTS from the given partial draft state and return pick recommendations.

    Parameters
    ----------
    my_picks      : brawlers my team has already picked (0–2; must be < 3 and
                    the draft must not be complete)
    opp_picks     : brawlers the opponent has already picked (0–2)
    mode          : game mode string (e.g. "gemGrab")
    map_name      : map name string (e.g. "Double Swoosh")
    skill_ns      : continuous skill score for the session (logit-ECDF percentile).
                    Constant across the entire draft.
    is_first_pick : True  = my team is P1 (picks first after coin flip).
                    False = my team is P2.
                    REQUIRED — cannot be derived from pick counts; the caller
                    must specify this based on the actual match assignment.
    bans          : brawlers excluded from the pick pool (default: none).
    n_simulations : MCTS simulation budget (default 1000). Increase for stronger
                    recommendations; decrease for faster response.
    n_top         : number of top candidates to surface in Layer 2 (default 5).
    my_counter_weight  : rollout blend weight for my team's future picks.
                    0 = popularity-driven, 1 = counter-rate-driven (default 0.3).
                    Only used when my_rollout_mode="weighted".
    opp_counter_weight : rollout blend weight for opponent picks (default 0.5).
                    Only used when opp_rollout_mode="weighted".
    min_pick_rate  : exclude brawlers with pick_rate < this before sampling
                    (default 0.0 = no filter).
    min_counter_games : only trust counter_rate entries backed by at least
                    this many games; sparse entries fall back to 0.5 neutral.
                    Default 0 (no filter).  Recommended 10–20 for top_k/greedy
                    modes to suppress noise from new or rarely-matched brawlers.
    opp_rollout_mode : "weighted" (default) — sample by blended weight.
                    "top_k" — sample uniformly from top-k counters (k=1 = greedy).
    opp_top_k      : k for opp top_k mode (default 3).
    my_rollout_mode  : "weighted" (default) or "top_k" for my team's picks.
    my_top_k       : k for my top_k mode (default 3).
    ucb1_c        : UCB1/PUCT exploration constant (default 0.5).
    puct_alpha    : prior blend weight for PUCT at opponent-turn nodes (default 0.7).
                    prior(b) = (1-alpha)*pick_rate(b) + alpha*avg_counter_rate(b).
                    Higher alpha concentrates exploration on stronger counters.
                    0.0 disables PUCT entirely (plain UCB1 at all nodes).
    evaluator     : pre-loaded FMEvaluator; loaded from disk if None.
    db            : pre-loaded MatchupDB; loaded from disk if None.
    rng           : NumPy random generator; created fresh if None.
    policy        : optional trained PolicyInference (or JointNetInference) to use
                    as PUCT prior at opponent-turn nodes. When provided, replaces
                    the counter-rate prior from Part 3.2 with the learned policy
                    prior. When None, falls back to the counter-rate prior (if
                    puct_alpha > 0) or uniform.
    value_net     : optional JointNetInference for rollout-free MCTS. When provided,
                    replaces rollout() entirely: every leaf is evaluated directly
                    via value_net.evaluate() (no random pick simulation). Lower
                    variance → faster convergence. Pass the same JointNetInference
                    as both `policy` and `value_net` for maximum benefit.

    Returns
    -------
    RecommendResult with:
      - top_picks: top-N brawlers by MCTS visit count (layer2['top_picks'])
      - layer1: empirical data confidence for current cross-team pairs
      - layer2: MCTS convergence confidence + visit distribution
      - whose_turn: 'mine' (recommendation) or 'opp' (prediction)

    Raises
    ------
    ValueError  if the draft is already complete, if more than 3 brawlers are
                given for either team, or if the same brawler appears on both
                teams or in bans.
    """
    my_team  = frozenset(b.upper() for b in my_picks)
    opp_team = frozenset(b.upper() for b in opp_picks)
    ban_set  = frozenset(b.upper() for b in bans)

    if len(my_team) > 3:
        raise ValueError(f"my_picks has {len(my_team)} brawlers; max is 3.")
    if len(opp_team) > 3:
        raise ValueError(f"opp_picks has {len(opp_team)} brawlers; max is 3.")
    overlap = my_team & opp_team
    if overlap:
        raise ValueError(f"Same brawler on both teams: {sorted(overlap)}")
    picked_and_banned = (my_team | opp_team) & ban_set
    if picked_and_banned:
        raise ValueError(f"Brawler(s) both picked and banned: {sorted(picked_and_banned)}")

    state = DraftState(
        my_team=my_team,
        opp_team=opp_team,
        mode=mode,
        map_name=map_name,
        skill_ns=skill_ns,
        is_first_pick=is_first_pick,
        bans=ban_set,
    )

    if state.is_terminal:
        raise ValueError(
            "Draft is already complete (both teams have 3 brawlers). "
            "Nothing to recommend."
        )

    if evaluator is None:
        evaluator = FMEvaluator.load()
    if db is None:
        db = MatchupDB.load()
    if rng is None:
        rng = np.random.default_rng()

    vocab = evaluator._fm.schema.vocab

    # Build a filtered vocab for tree expansion when min_pick_rate > 0.
    # This removes low-pick-rate brawlers from the MCTS tree entirely so
    # simulations are not wasted visiting them.  The full vocab is kept for
    # the rollout weight cache so counter-rate lookups on already-picked
    # brawlers always succeed regardless of their pick rate.
    if min_pick_rate > 0.0 and db.skill_tier_boundaries is not None:
        tier = skill_ns_to_tier(state.skill_ns, db.skill_tier_boundaries)
        _filtered: list[str] = []
        for b in vocab:
            entry = db.brawler_lookup(b, mode, map_name, tier)
            if entry is None or entry["pick_rate"] >= min_pick_rate:
                _filtered.append(b)
        tree_vocab: tuple[str, ...] | None = tuple(_filtered)
    else:
        tree_vocab = None  # signals _run_mcts to use full vocab

    t0 = time.perf_counter()
    root = _run_mcts(
        state=state,
        evaluator=evaluator,
        db=db,
        vocab=vocab,
        n_simulations=n_simulations,
        my_counter_weight=my_counter_weight,
        opp_counter_weight=opp_counter_weight,
        min_pick_rate=min_pick_rate,
        opp_rollout_mode=opp_rollout_mode,
        opp_top_k=opp_top_k,
        my_rollout_mode=my_rollout_mode,
        my_top_k=my_top_k,
        min_counter_games=min_counter_games,
        c=ucb1_c,
        rng=rng,
        puct_alpha=puct_alpha,
        tree_vocab=tree_vocab,
        policy=policy,
        value_net=value_net,
    )
    elapsed = time.perf_counter() - t0

    l1 = layer1_confidence(state, db)
    l2 = layer2_confidence(root, n_top=n_top)

    return RecommendResult(
        state=state,
        whose_turn=state.whose_turn,
        layer1=l1,
        layer2=l2,
        elapsed_sec=elapsed,
    )


# ── Smoke tests ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Step 2.7 Smoke Tests ===\n")

    # Load once; reuse across all calls.
    evaluator = FMEvaluator.load()
    db = MatchupDB.load()
    schema = evaluator._fm.schema
    vocab = schema.vocab

    MAP_A  = schema.maps[0]
    MAP_B  = schema.maps[1]   # second map — recommendations should differ
    MODE   = schema.modes[0]
    SKILL  = 1.0

    # ── Test 1: Structural validity at draft start ────────────────────────────
    # Verify recommend() returns a well-formed RecommendResult.
    print("Test 1: structural validity at draft start (500 sims)")
    result = recommend(
        my_picks=[], opp_picks=[],
        mode=MODE, map_name=MAP_A, skill_ns=SKILL,
        is_first_pick=True,
        n_simulations=500,
        evaluator=evaluator, db=db, rng=np.random.default_rng(42),
    )

    assert result.whose_turn in {"mine", "opp"}, f"Unexpected whose_turn: {result.whose_turn}"
    assert 1 <= len(result.top_picks) <= 5
    assert result.layer2["total_simulations"] == 500
    assert result.elapsed_sec > 0.0

    # All top picks must be available brawlers (in vocab, not picked, not banned).
    available = set(vocab)
    for pick in result.top_picks:
        b = pick["brawler"]
        assert b in available, f"Recommended brawler {b!r} not in vocab"
        assert b not in result.state.my_team, f"Recommended own team's pick: {b}"
        assert b not in result.state.opp_team, f"Recommended opp's pick: {b}"
        assert 0.0 < pick["estimated_win_prob"] < 1.0, f"Win prob out of range: {pick}"

    print(result.summary())
    print("  ✓\n")

    # ── Test 2: Pick/ban exclusion ────────────────────────────────────────────
    # After picking brawlers, they must not appear in recommendations.
    # Banned brawlers must never appear.
    print("Test 2: picked and banned brawlers excluded from recommendations")
    my_pick   = vocab[0]
    opp_pick  = vocab[1]
    banned    = vocab[2]

    result2 = recommend(
        my_picks=[my_pick], opp_picks=[opp_pick],
        mode=MODE, map_name=MAP_A, skill_ns=SKILL,
        is_first_pick=True,
        bans=[banned],
        n_simulations=300,
        evaluator=evaluator, db=db, rng=np.random.default_rng(0),
    )

    recommended_brawlers = {p["brawler"] for p in result2.top_picks}
    assert my_pick  not in recommended_brawlers, f"{my_pick!r} (own pick) in recommendations"
    assert opp_pick not in recommended_brawlers, f"{opp_pick!r} (opp pick) in recommendations"
    assert banned   not in recommended_brawlers, f"{banned!r} (banned) in recommendations"
    print(f"  my_pick={my_pick}, opp_pick={opp_pick}, banned={banned}")
    print(f"  top_picks: {[p['brawler'] for p in result2.top_picks]}")
    print("  ✓\n")

    # ── Test 3: Context sensitivity — opponent pick shifts recommendations ─────
    # After the opponent claims a brawler, that brawler disappears from the pool
    # and recommendations shift. We verify the top-5 sets differ (at minimum the
    # opponent's brawler is gone).
    print("Test 3: recommendations shift after opponent claims a brawler")
    rng_base = np.random.default_rng(7)
    result_before = recommend(
        my_picks=[], opp_picks=[],
        mode=MODE, map_name=MAP_A, skill_ns=SKILL,
        is_first_pick=False,    # P2: first pick is the opponent's
        n_simulations=500,
        evaluator=evaluator, db=db, rng=rng_base,
    )
    # The opponent will have just picked something; simulate them taking the
    # top recommendation from a fresh run with is_first_pick=True.
    rng_opp = np.random.default_rng(8)
    opp_result = recommend(
        my_picks=[], opp_picks=[],
        mode=MODE, map_name=MAP_A, skill_ns=SKILL,
        is_first_pick=True,
        n_simulations=300,
        evaluator=evaluator, db=db, rng=rng_opp,
    )
    opp_claimed = opp_result.best   # treat the top pick as "opponent took this"

    rng_after = np.random.default_rng(9)
    result_after = recommend(
        my_picks=[], opp_picks=[opp_claimed],
        mode=MODE, map_name=MAP_A, skill_ns=SKILL,
        is_first_pick=False,
        n_simulations=500,
        evaluator=evaluator, db=db, rng=rng_after,
    )

    picks_before = {p["brawler"] for p in result_before.top_picks}
    picks_after  = {p["brawler"] for p in result_after.top_picks}
    assert opp_claimed not in picks_after, (
        f"Opponent's claimed brawler {opp_claimed!r} still in recommendations"
    )
    changed = picks_before.symmetric_difference(picks_after)
    print(f"  opponent claimed: {opp_claimed}")
    print(f"  top-5 before:  {sorted(picks_before)}")
    print(f"  top-5 after:   {sorted(picks_after)}")
    print(f"  changed brawlers between the two sets: {sorted(changed)}")
    print("  ✓\n")

    # ── Test 4: Map context is used — top picks differ across maps ────────────
    # This is informational: if the FM has learned map-specific patterns, the
    # top-5 should differ between two different maps. Failure here is suspicious
    # but not a hard correctness error (maps could genuinely have identical ranking).
    print("Test 4: map context — top-5 comparison across two maps (informational)")
    result_map_a = recommend(
        my_picks=[], opp_picks=[],
        mode=MODE, map_name=MAP_A, skill_ns=SKILL,
        is_first_pick=True,
        n_simulations=500,
        evaluator=evaluator, db=db, rng=np.random.default_rng(10),
    )
    result_map_b = recommend(
        my_picks=[], opp_picks=[],
        mode=MODE, map_name=MAP_B, skill_ns=SKILL,
        is_first_pick=True,
        n_simulations=500,
        evaluator=evaluator, db=db, rng=np.random.default_rng(11),
    )
    top_a = [p["brawler"] for p in result_map_a.top_picks]
    top_b = [p["brawler"] for p in result_map_b.top_picks]
    overlap_count = len(set(top_a) & set(top_b))
    differs = (top_a != top_b)
    print(f"  map A ({MAP_A}): {top_a}")
    print(f"  map B ({MAP_B}): {top_b}")
    print(f"  lists identical: {not differs}  |  overlapping brawlers: {overlap_count}/5")
    if not differs:
        print("  [WARN] top-5 lists are identical across maps — FM may not be using map context")
    else:
        print("  Map context is influencing recommendations  ✓")
    print()

    # ── Test 5: P2 initial state (opponent picks first) ───────────────────────
    # When is_first_pick=False, the first pick is the opponent's.
    # Verify whose_turn == 'opp' at draft start.
    print("Test 5: P2 draft start — whose_turn should be 'opp'")
    result_p2 = recommend(
        my_picks=[], opp_picks=[],
        mode=MODE, map_name=MAP_A, skill_ns=SKILL,
        is_first_pick=False,
        n_simulations=300,
        evaluator=evaluator, db=db, rng=np.random.default_rng(20),
    )
    assert result_p2.whose_turn == "opp", (
        f"Expected whose_turn='opp' for P2 at pick 0, got {result_p2.whose_turn!r}"
    )
    print(f"  whose_turn = {result_p2.whose_turn!r}  (P2 draft start → opponent picks first)")
    print(f"  top predicted opp picks: {[p['brawler'] for p in result_p2.top_picks]}")
    print("  ✓\n")

    # ── Test 6: ValueError on already-complete draft ──────────────────────────
    print("Test 6: ValueError raised on complete draft")
    try:
        recommend(
            my_picks=vocab[:3], opp_picks=vocab[3:6],
            mode=MODE, map_name=MAP_A, skill_ns=SKILL,
            is_first_pick=True,
            evaluator=evaluator, db=db,
        )
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        print(f"  Caught ValueError: {e}")
    print("  ✓\n")

    print("=== All smoke tests passed ===")
