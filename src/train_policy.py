"""
train_policy.py — Part 4.3: Iterative Self-Play Improvement

One iteration of the self-play training loop:
  1. Generate N self-play games using MCTS + current policy (iter 0 = no policy).
  2. Train policy network on ALL accumulated data, from scratch.
  3. Evaluate new policy vs. old policy: 200 head-to-head games on a fixed
     held-out map set. Winner = agent with higher mean FM win probability.
  4. Promote new policy if mean FM win prob >= promotion_threshold (default 0.525).
  5. Repeat for n_iterations.

Sim budget
----------
Iteration 0 uses n_sims_iter0 (default 2 000) because the policy is random and
more sims produce no better training signal.  Iteration 1+ uses n_sims_later
(default 5 000): the policy prior is now informative, so extra sims meaningfully
improve data quality for backup picks 5-10.

At recommendation time, users should run recommend() with 10k-20k sims + the
trained policy loaded to get quality top-5 to top-10 picks.  Self-play sims
determine training data quality, not final output quality.

Data retention
--------------
All self-play games from every iteration are kept.  The policy is retrained from
scratch each iteration to avoid bias toward early weak games.  Games live in the
ReplayBuffer; each iteration appends a new batch to data_dir/self_play/.

Evaluation
----------
A fixed held-out eval map set (deterministically chosen from all available maps)
is used for all iterations so the promotion decision is always an apples-to-apples
comparison.  Eval games use greedy pick selection (temperature=0) for
reproducibility.  Mean FM win probability (not binary win/loss) is used as the
metric — this is more stable with 200 games.

Persistence
-----------
  data_dir/policy/policy_iter_{n:03d}.pkl  — saved policy at each iteration
  data_dir/policy/policy_best.pkl          — currently promoted policy
  data_dir/policy/training_log.json        — list of IterationResult dicts
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from src.draft_state import DraftState, apply_pick
    from src.fm_integration import FMEvaluator
    from src.fm_model import FMInference
    from src.matchup_db import MatchupDB, skill_ns_to_tier
    from src.mcts_node import UCB1_C
    from src.policy_net import PolicyInference, train_policy
    from src.recommend import _run_mcts
    from src.rollout import (
        DEFAULT_MIN_COUNTER_GAMES,
        DEFAULT_MY_COUNTER_WEIGHT,
        DEFAULT_OPP_COUNTER_WEIGHT,
        RolloutWeightCache,
    )
    from src.self_play import (
        DEFAULT_SKILL_NS_CHOICES,
        DEFAULT_VISIT_DIST_TOP_N,
        ReplayBuffer,
        _build_tree_vocab,
        _sample_pick,
        _SP_BAN_P,
        _SP_BAN_TOP_N,
        _SP_MAX_BANS,
        _SP_MIN_COUNTER_GAMES,
        _SP_MIN_PICK_RATE,
        _SP_OPP_ROLLOUT_MODE,
        _SP_OPP_TOP_K,
        load_map_mode_pairs,
        run_self_play_batch,
        sample_bans,
    )
except ModuleNotFoundError:
    from draft_state import DraftState, apply_pick
    from fm_integration import FMEvaluator
    from fm_model import FMInference
    from matchup_db import MatchupDB, skill_ns_to_tier
    from mcts_node import UCB1_C
    from policy_net import PolicyInference, train_policy
    from recommend import _run_mcts
    from rollout import (
        DEFAULT_MIN_COUNTER_GAMES,
        DEFAULT_MY_COUNTER_WEIGHT,
        DEFAULT_OPP_COUNTER_WEIGHT,
        RolloutWeightCache,
    )
    from self_play import (
        DEFAULT_SKILL_NS_CHOICES,
        DEFAULT_VISIT_DIST_TOP_N,
        ReplayBuffer,
        _build_tree_vocab,
        _sample_pick,
        _SP_BAN_P,
        _SP_BAN_TOP_N,
        _SP_MAX_BANS,
        _SP_MIN_COUNTER_GAMES,
        _SP_MIN_PICK_RATE,
        _SP_OPP_ROLLOUT_MODE,
        _SP_OPP_TOP_K,
        load_map_mode_pairs,
        run_self_play_batch,
        sample_bans,
    )


# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_N_GAMES_PER_ITER: int = 500
"""
Games generated per iteration.

At 2k sims/pick × 6 picks × ~0.05 s/sim = ~0.6 s/game (serial), 500 games
takes ~5 min serial or ~1.5 min with 4 workers.  Later iterations at 5k sims
take proportionally longer.  Increase to 1000+ for higher-quality policies.
"""

DEFAULT_N_SIMS_ITER0: int = 2_000
"""Sim budget per pick for iteration 0 (no policy prior — more sims are wasteful)."""

DEFAULT_N_SIMS_LATER: int = 5_000
"""
Sim budget per pick for iteration 1+ (policy prior is informative).

5k sims with a good policy prior concentrates visits on the top 10-15 picks,
giving each of the top-10 candidates 100-300 visits — enough for a stable
training target for backup picks.
"""

DEFAULT_N_EVAL_GAMES: int = 200
"""
Head-to-head games for policy evaluation.

With continuous FM win probability as the metric, 200 games gives a standard
error of ~±0.011 (σ≈0.15 / √200).  This reliably detects improvements of
≥0.02 above the 0.525 promotion threshold.
"""

DEFAULT_PROMOTION_THRESHOLD: float = 0.525
"""
New policy must achieve ≥ this mean FM win probability to be promoted.

0.525 gives the new policy modest slack above the symmetric baseline of 0.5.
A policy that wins 52.5% of FM-evaluated terminal states on the eval map set
is considered improved.
"""

DEFAULT_N_EVAL_MAPS_PER_MODE: int = 2
"""Maps per mode in the fixed held-out evaluation set (2 maps × N modes)."""

DEFAULT_N_SIMS_EVAL: int = 2_000
"""
Sim budget per pick during policy evaluation.

Lower than self-play to keep 200 eval games fast.  Eval games use greedy
temperature=0 selection, so the argmax is stable even with fewer sims.
"""


# ── Configuration dataclass ────────────────────────────────────────────────────

@dataclass
class IterationConfig:
    """
    Configuration for one self-play training loop.

    All parameters have sensible defaults.  Increase n_games_per_iter and
    n_sims_later for a stronger policy at the cost of more compute.

    Parameters
    ----------
    n_games_per_iter      : games generated each iteration (default 500).
    n_sims_iter0          : sims/pick for iteration 0 (default 2 000).
    n_sims_later          : sims/pick for iteration 1+ (default 5 000).
    n_eval_games          : head-to-head games for evaluation (default 200).
    n_sims_eval           : sims/pick during evaluation (default 2 000).
    promotion_threshold   : new policy mean FM win prob required for promotion
                            (default 0.525).
    n_eval_maps_per_mode  : maps per mode in the held-out eval set (default 2).
    n_workers             : parallel workers for data generation (default 1).
    opp_rollout_mode      : rollout policy for opponent picks (default "top_k").
    opp_top_k             : k for top_k opponent rollout (default 3).
    min_pick_rate         : brawler tree-vocab filter (default 0.005).
    min_counter_games     : minimum games to trust counter_rate (default 10).
    skill_ns_choices      : skill tiers sampled per game (default 1.0/2.0/3.0).
    seed                  : root RNG seed for reproducibility (default None).
    ban_top_n             : top-pick-rate brawlers eligible for ban sampling
                            (default 15). Dominant brawlers (Sirius, Bull, etc.)
                            naturally rank here. Set to 0 to disable bans.
    ban_p                 : per-brawler ban probability for each eligible brawler
                            (default 0.35). Each top-N brawler is absent from ~35%
                            of games, preventing "first-to-pick-S-tier-wins" collapse.
    max_bans              : hard cap on total bans per game (default 6).
    """

    n_games_per_iter: int = DEFAULT_N_GAMES_PER_ITER
    n_sims_iter0: int = DEFAULT_N_SIMS_ITER0
    n_sims_later: int = DEFAULT_N_SIMS_LATER
    n_eval_games: int = DEFAULT_N_EVAL_GAMES
    n_sims_eval: int = DEFAULT_N_SIMS_EVAL
    promotion_threshold: float = DEFAULT_PROMOTION_THRESHOLD
    n_eval_maps_per_mode: int = DEFAULT_N_EVAL_MAPS_PER_MODE
    n_workers: int = 1
    opp_rollout_mode: str = _SP_OPP_ROLLOUT_MODE
    opp_top_k: int = _SP_OPP_TOP_K
    min_pick_rate: float = _SP_MIN_PICK_RATE
    min_counter_games: int = _SP_MIN_COUNTER_GAMES
    skill_ns_choices: tuple = DEFAULT_SKILL_NS_CHOICES
    seed: Optional[int] = None
    ban_top_n: int = _SP_BAN_TOP_N
    ban_p: float = _SP_BAN_P
    max_bans: int = _SP_MAX_BANS


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class IterationResult:
    """
    Output of one self-play training iteration.

    Fields
    ------
    iteration         : 0-based iteration index.
    n_games_generated : games generated this iteration.
    n_games_total     : total games in the replay buffer after this iteration.
    train_kl          : final training cross-entropy (proxy for KL divergence).
    val_kl            : final validation cross-entropy (early-stopping metric).
    eval_win_prob     : mean FM win probability of new policy vs. old (or vs.
                        no-policy baseline for iteration 0).
    promoted          : True if new policy exceeded promotion_threshold.
    elapsed_sec       : wall time for the full iteration.
    n_sims_per_pick   : sim budget used for data generation this iteration.
    """

    iteration: int
    n_games_generated: int
    n_games_total: int
    train_kl: float
    val_kl: float
    eval_win_prob: float
    promoted: bool
    elapsed_sec: float
    n_sims_per_pick: int
    eval_opponent: str = "best_ever"   # "best_ever" | "baseline" (iter 0, no prior policy)
    loss_history: list = field(default_factory=list)  # (epoch, train_kl, val_kl) tuples


# ── Eval map selection ─────────────────────────────────────────────────────────

def select_eval_maps(
    all_pairs: list[tuple[str, str]],
    n_per_mode: int = DEFAULT_N_EVAL_MAPS_PER_MODE,
) -> list[tuple[str, str]]:
    """
    Deterministically select a held-out evaluation map set.

    Groups all (map_name, mode) pairs by mode, then takes the first
    `n_per_mode` maps per mode (alphabetically by map name).  The result
    is stable across iterations so all policies are evaluated on the same maps.

    Parameters
    ----------
    all_pairs  : full list of (map_name, mode) pairs from load_map_mode_pairs().
    n_per_mode : maps to include per mode (default 2).

    Returns
    -------
    Sorted list of (map_name, mode) tuples for the eval set.
    """
    by_mode: dict[str, list[str]] = defaultdict(list)
    for map_name, mode in all_pairs:
        by_mode[mode].append(map_name)

    result: list[tuple[str, str]] = []
    for mode in sorted(by_mode):
        maps_sorted = sorted(set(by_mode[mode]))
        for m in maps_sorted[:n_per_mode]:
            result.append((m, mode))
    return result


# ── Head-to-head evaluation game ──────────────────────────────────────────────

def _run_eval_game(
    new_policy: Optional[PolicyInference],
    old_policy: Optional[PolicyInference],
    evaluator: FMEvaluator,
    db: MatchupDB,
    game_idx: int,
    eval_map_mode_pairs: list[tuple[str, str]],
    *,
    n_sims_per_pick: int = DEFAULT_N_SIMS_EVAL,
    skill_ns_choices: tuple = DEFAULT_SKILL_NS_CHOICES,
    opp_rollout_mode: str = _SP_OPP_ROLLOUT_MODE,
    opp_top_k: int = _SP_OPP_TOP_K,
    min_pick_rate: float = _SP_MIN_PICK_RATE,
    min_counter_games: int = _SP_MIN_COUNTER_GAMES,
    ban_top_n: int = _SP_BAN_TOP_N,
    ban_p: float = _SP_BAN_P,
    max_bans: int = _SP_MAX_BANS,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Run one evaluation game: new_policy agent vs. old_policy agent.

    The new_policy agent plays as the "primary team" (P1 for even game_idx,
    P2 for odd).  Returns the FM win probability from the new_policy's
    perspective at the terminal state.

    Picks are made greedily (temperature=0 — argmax of visit distribution)
    for reproducibility.  This maximises each agent's play strength so the
    evaluation cleanly measures policy quality rather than sampling variance.

    Parameters
    ----------
    new_policy : newly trained policy (None = no policy, counter-rate prior only).
    old_policy : previous iteration's policy (None = no policy).
    evaluator  : pre-loaded FMEvaluator.
    db         : pre-loaded MatchupDB.
    game_idx   : sequential index; determines P1/P2 assignment and RNG seeding.
    eval_map_mode_pairs : held-out (map, mode) pairs from select_eval_maps().
    n_sims_per_pick     : MCTS sim budget per pick slot.
    rng        : NumPy generator (fresh if None).

    Returns
    -------
    float — FM win probability for the new_policy agent at the terminal state.
    """
    if rng is None:
        rng = np.random.default_rng()

    vocab = evaluator._fm.schema.vocab

    # Deterministically pick a map from the eval set by game_idx.
    map_name, mode = eval_map_mode_pairs[game_idx % len(eval_map_mode_pairs)]
    skill_ns = float(skill_ns_choices[int(rng.integers(len(skill_ns_choices)))])
    is_first_pick = (game_idx % 2 == 0)  # alternating P1/P2

    tier = (
        skill_ns_to_tier(skill_ns, db.skill_tier_boundaries)
        if db.skill_tier_boundaries is not None
        else 1
    )

    tree_vocab = _build_tree_vocab(vocab, db, mode, map_name, tier, min_pick_rate)
    weight_cache = RolloutWeightCache(db, vocab, min_counter_games=min_counter_games)
    weight_cache._get(mode, map_name, tier)

    # Sample bans for this eval game — same distribution as training games so
    # the evaluation measures policy quality on the same state distribution.
    bans = sample_bans(
        vocab, db, mode, map_name, tier, rng,
        ban_top_n=ban_top_n, ban_p=ban_p, max_bans=max_bans,
    )

    game_state = DraftState(
        my_team=frozenset(),
        opp_team=frozenset(),
        mode=mode,
        map_name=map_name,
        skill_ns=skill_ns,
        is_first_pick=is_first_pick,
        bans=bans,
    )

    _mcts_kwargs = dict(
        evaluator=evaluator,
        db=db,
        vocab=vocab,
        n_simulations=n_sims_per_pick,
        my_counter_weight=DEFAULT_MY_COUNTER_WEIGHT,
        opp_counter_weight=DEFAULT_OPP_COUNTER_WEIGHT,
        min_pick_rate=min_pick_rate,
        opp_rollout_mode=opp_rollout_mode,
        opp_top_k=opp_top_k,
        my_rollout_mode="weighted",
        my_top_k=3,
        min_counter_games=min_counter_games,
        c=UCB1_C,
        rng=rng,
        puct_alpha=0.7,
        tree_vocab=tree_vocab,
        weight_cache=weight_cache,
    )

    for _ in range(6):
        current_turn = game_state.whose_turn  # "mine" = new_policy's turn

        if current_turn == "mine":
            mcts_state = game_state
            active_policy = new_policy
        else:
            # Mirror state so the old_policy agent evaluates from its own perspective.
            mcts_state = DraftState(
                my_team=game_state.opp_team,
                opp_team=game_state.my_team,
                mode=mode,
                map_name=map_name,
                skill_ns=skill_ns,
                is_first_pick=not is_first_pick,
                bans=bans,
            )
            active_policy = old_policy

        root = _run_mcts(state=mcts_state, policy=active_policy, **_mcts_kwargs)

        # Greedy pick (temperature=0) for evaluation — maximises each agent's strength.
        pick = _sample_pick(root, temperature=0.0, rng=rng)
        game_state = apply_pick(game_state, pick)

    # Terminal evaluation from new_policy's perspective (is_first_pick = was new_policy P1?).
    terminal_win_prob = evaluator.evaluate(game_state)
    return terminal_win_prob


# ── Policy evaluation ──────────────────────────────────────────────────────────

def evaluate_policies(
    new_policy: Optional[PolicyInference],
    old_policy: Optional[PolicyInference],
    evaluator: FMEvaluator,
    db: MatchupDB,
    eval_map_mode_pairs: list[tuple[str, str]],
    *,
    n_games: int = DEFAULT_N_EVAL_GAMES,
    n_sims_per_pick: int = DEFAULT_N_SIMS_EVAL,
    skill_ns_choices: tuple = DEFAULT_SKILL_NS_CHOICES,
    opp_rollout_mode: str = _SP_OPP_ROLLOUT_MODE,
    opp_top_k: int = _SP_OPP_TOP_K,
    min_pick_rate: float = _SP_MIN_PICK_RATE,
    min_counter_games: int = _SP_MIN_COUNTER_GAMES,
    ban_top_n: int = _SP_BAN_TOP_N,
    ban_p: float = _SP_BAN_P,
    max_bans: int = _SP_MAX_BANS,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> float:
    """
    Run n_games head-to-head games between new_policy and old_policy.

    Returns the mean FM win probability of new_policy across all games.
    A symmetric baseline (two equal agents) produces ~0.5.  The promotion
    threshold of 0.525 accounts for random variance with 200 games.

    Games are split evenly: half with new_policy as P1, half as P2.  This
    prevents draft-order bias from influencing the evaluation result.

    Parameters
    ----------
    new_policy : newly trained PolicyInference (None = no policy).
    old_policy : previous iteration's PolicyInference (None = no policy baseline).
    evaluator  : pre-loaded FMEvaluator.
    db         : pre-loaded MatchupDB.
    eval_map_mode_pairs : held-out map set from select_eval_maps().
    n_games    : number of eval games (default 200).
    n_sims_per_pick : sims per pick during eval (default 2 000).
    seed       : RNG seed for reproducibility (default None).
    verbose    : print progress every 50 games (default True).

    Returns
    -------
    float — mean FM win probability for new_policy (0.0–1.0).
    """
    if not eval_map_mode_pairs:
        raise ValueError("eval_map_mode_pairs must not be empty.")

    ss = np.random.SeedSequence(seed)
    child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_games)]

    win_probs: list[float] = []
    for i in range(n_games):
        wp = _run_eval_game(
            new_policy=new_policy,
            old_policy=old_policy,
            evaluator=evaluator,
            db=db,
            game_idx=i,
            eval_map_mode_pairs=eval_map_mode_pairs,
            n_sims_per_pick=n_sims_per_pick,
            skill_ns_choices=skill_ns_choices,
            opp_rollout_mode=opp_rollout_mode,
            opp_top_k=opp_top_k,
            min_pick_rate=min_pick_rate,
            min_counter_games=min_counter_games,
            ban_top_n=ban_top_n,
            ban_p=ban_p,
            max_bans=max_bans,
            rng=np.random.default_rng(child_seeds[i]),
        )
        win_probs.append(wp)
        if verbose and (i + 1) % 50 == 0:
            mean_so_far = float(np.mean(win_probs))
            print(f"  eval [{i+1}/{n_games}]  mean_win_prob={mean_so_far:.4f}")

    mean_wp = float(np.mean(win_probs))
    if verbose:
        print(f"  eval complete: mean_win_prob={mean_wp:.4f}  (n={n_games})")
    return mean_wp


# ── Persistence helpers ────────────────────────────────────────────────────────

def _policy_dir(data_dir: Path) -> Path:
    d = data_dir / "policy"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_training_log(data_dir: Path) -> list[dict]:
    log_path = _policy_dir(data_dir) / "training_log.json"
    if log_path.exists():
        with open(log_path) as fh:
            return json.load(fh)
    return []


def _save_training_log(data_dir: Path, log: list[dict]) -> None:
    log_path = _policy_dir(data_dir) / "training_log.json"
    with open(log_path, "w") as fh:
        json.dump(log, fh, indent=2)


def _load_best_policy(data_dir: Path) -> Optional[PolicyInference]:
    """Load the most recently promoted policy (used for self-play data generation)."""
    best_path = _policy_dir(data_dir) / "policy_best.pkl"
    if best_path.exists():
        return PolicyInference.load(best_path)
    return None


def _load_best_ever_policy(data_dir: Path) -> Optional[PolicyInference]:
    """
    Load the all-time best promoted policy (used as the promotion eval opponent).

    Distinct from policy_best.pkl: best_ever is only updated when a new policy
    beats it.  In the non-cyclic case these are identical; when a weaker policy
    somehow gets promoted and then is beaten by a stronger one, best_ever remains
    the high-water mark while policy_best tracks the most recent promotion.
    """
    path = _policy_dir(data_dir) / "policy_best_ever.pkl"
    if path.exists():
        return PolicyInference.load(path)
    return None


def _save_policy(
    policy: PolicyInference,
    data_dir: Path,
    iteration: int,
    *,
    promoted: bool,
    is_new_best_ever: bool,
) -> None:
    """
    Persist policy artifacts after one iteration.

    Always saved   : policy_iter_NNN.pkl  — snapshot for every iteration
    When promoted  : policy_best.pkl      — most recently promoted (data-gen prior)
    When best-ever : policy_best_ever.pkl — high-water mark (eval opponent)
    """
    pdir = _policy_dir(data_dir)
    policy.save(pdir / f"policy_iter_{iteration:03d}.pkl")
    if promoted:
        policy.save(pdir / "policy_best.pkl")
    if is_new_best_ever:
        policy.save(pdir / "policy_best_ever.pkl")


# ── Single iteration ───────────────────────────────────────────────────────────

def run_iteration(
    iteration_num: int,
    replay_buffer: ReplayBuffer,
    current_policy: Optional[PolicyInference],
    best_ever_policy: Optional[PolicyInference],
    evaluator: FMEvaluator,
    db: MatchupDB,
    map_mode_pairs: list[tuple[str, str]],
    eval_map_mode_pairs: list[tuple[str, str]],
    config: IterationConfig,
    data_dir: Path,
) -> tuple[PolicyInference, IterationResult]:
    """
    Run one complete training iteration and return the (new policy, result).

    Steps
    -----
    1. Generate n_games_per_iter self-play games using current_policy.
    2. Train a fresh PolicyNet on ALL records in the replay buffer.
    3. Evaluate new policy vs. best_ever_policy on the held-out eval map set.
    4. Promote if new policy exceeds the threshold; update best_ever if so.

    Two distinct policies are tracked:
      current_policy   — most recently promoted; used for self-play data generation
                         so the agent always generates data with its best current play.
      best_ever_policy — the all-time highest-performing promoted policy; used as the
                         evaluation opponent so the promotion bar never lowers.  A newly
                         trained policy must beat the best seen so far, not just the most
                         recent one, preventing cyclic regressions where A beats B, B beats
                         C, and C then beats A even though C < A.

    Parameters
    ----------
    iteration_num      : 0-based iteration index (determines sim budget).
    replay_buffer      : in-memory + on-disk game store; updated in place.
    current_policy     : most recently promoted policy (None for iter 0).
                         Used for self-play data generation only.
    best_ever_policy   : all-time best promoted policy (None for iter 0).
                         Used as the evaluation opponent for promotion decisions.
    evaluator          : pre-loaded FMEvaluator.
    db                 : pre-loaded MatchupDB.
    map_mode_pairs     : full (map, mode) pool for data generation.
    eval_map_mode_pairs: fixed held-out eval set from select_eval_maps().
    config             : IterationConfig with all hyperparameters.
    data_dir           : root directory for saving policies and batches.

    Returns
    -------
    (new_policy, IterationResult) — if result.promoted, new_policy is both the
    new current_policy AND the new best_ever_policy; the caller should update both.
    """
    t_iter_start = time.perf_counter()

    n_sims = config.n_sims_iter0 if iteration_num == 0 else config.n_sims_later

    # ── Step 1: Generate self-play games ──────────────────────────────────────
    print(
        f"\n=== Iteration {iteration_num} | "
        f"{config.n_games_per_iter} games | {n_sims} sims/pick ==="
    )
    print("  [1/3] Generating self-play games...")
    t0 = time.perf_counter()

    # Use a seeded sequence derived from the root seed + iteration for reproducibility.
    iter_seed = (
        None if config.seed is None
        else int(np.random.SeedSequence(config.seed).spawn(iteration_num + 1)[-1].generate_state(1)[0])
    )

    run_self_play_batch(
        n_games=config.n_games_per_iter,
        data_dir=data_dir,
        map_mode_pairs=map_mode_pairs,
        replay_buffer=replay_buffer,
        evaluator=evaluator,
        db=db,
        n_workers=config.n_workers,
        start_game_idx=replay_buffer.n_games,  # globally unique game IDs
        seed=iter_seed,
        n_sims_per_pick=n_sims,
        skill_ns_choices=config.skill_ns_choices,
        opp_rollout_mode=config.opp_rollout_mode,
        opp_top_k=config.opp_top_k,
        min_pick_rate=config.min_pick_rate,
        min_counter_games=config.min_counter_games,
        visit_dist_top_n=DEFAULT_VISIT_DIST_TOP_N,
        pick_temperature=1.0,
        policy=current_policy,
        ban_top_n=config.ban_top_n,
        ban_p=config.ban_p,
        max_bans=config.max_bans,
    )
    games_elapsed = time.perf_counter() - t0
    print(f"     done in {games_elapsed:.1f}s  ({replay_buffer.n_games} total games in buffer)")

    # ── Step 2: Train policy ──────────────────────────────────────────────────
    print("  [2/3] Training policy network (all accumulated data)...")
    t0 = time.perf_counter()

    records = replay_buffer.all_records()
    schema = evaluator._fm.schema
    loss_hist: list = []
    new_policy = train_policy(
        records=records,
        schema=schema,
        verbose=True,
        loss_history=loss_hist,
    )
    train_elapsed = time.perf_counter() - t0

    # Extract final train / val KL from history
    final_train_kl = loss_hist[-1][1] if loss_hist else float("nan")
    final_val_kl   = loss_hist[-1][2] if loss_hist else float("nan")
    best_val_kl    = min((e[2] for e in loss_hist), default=float("nan"))
    print(
        f"     done in {train_elapsed:.1f}s  "
        f"train_kl={final_train_kl:.4f}  best_val_kl={best_val_kl:.4f}"
    )

    # ── Step 3: Evaluate new policy vs. best-ever policy ─────────────────────
    _opponent_label = "best-ever" if best_ever_policy else "baseline (no policy)"
    print(
        f"  [3/3] Evaluating new vs. {_opponent_label}  "
        f"({config.n_eval_games} games, {config.n_sims_eval} sims/pick)..."
    )
    t0 = time.perf_counter()

    eval_seed = (
        None if config.seed is None
        else int(np.random.SeedSequence(config.seed + 9999).spawn(iteration_num + 1)[-1].generate_state(1)[0])
    )
    eval_win_prob = evaluate_policies(
        new_policy=new_policy,
        old_policy=best_ever_policy,
        evaluator=evaluator,
        db=db,
        eval_map_mode_pairs=eval_map_mode_pairs,
        n_games=config.n_eval_games,
        n_sims_per_pick=config.n_sims_eval,
        skill_ns_choices=config.skill_ns_choices,
        opp_rollout_mode=config.opp_rollout_mode,
        opp_top_k=config.opp_top_k,
        min_pick_rate=config.min_pick_rate,
        min_counter_games=config.min_counter_games,
        ban_top_n=config.ban_top_n,
        ban_p=config.ban_p,
        max_bans=config.max_bans,
        seed=eval_seed,
        verbose=True,
    )
    eval_elapsed = time.perf_counter() - t0

    promoted = eval_win_prob >= config.promotion_threshold
    status = "PROMOTED" if promoted else "rejected (keeping old policy)"
    print(
        f"     eval win prob = {eval_win_prob:.4f}  "
        f"threshold = {config.promotion_threshold:.3f}  →  {status}"
    )
    print(f"     eval done in {eval_elapsed:.1f}s")

    # ── Save policy ───────────────────────────────────────────────────────────
    _save_policy(
        new_policy, data_dir, iteration_num,
        promoted=promoted,
        is_new_best_ever=promoted,
    )

    total_elapsed = time.perf_counter() - t_iter_start
    print(f"  Iteration {iteration_num} complete in {total_elapsed:.1f}s total\n")

    result = IterationResult(
        iteration=iteration_num,
        n_games_generated=config.n_games_per_iter,
        n_games_total=replay_buffer.n_games,
        train_kl=final_train_kl,
        val_kl=best_val_kl,
        eval_win_prob=eval_win_prob,
        promoted=promoted,
        elapsed_sec=total_elapsed,
        n_sims_per_pick=n_sims,
        eval_opponent="baseline" if best_ever_policy is None else "best_ever",
        loss_history=loss_hist,
    )

    return new_policy, result


# ── Full iteration loop ────────────────────────────────────────────────────────

def run_iteration_loop(
    n_iterations: int,
    data_dir: Path,
    map_mode_pairs: list[tuple[str, str]],
    config: Optional[IterationConfig] = None,
    *,
    evaluator: Optional[FMEvaluator] = None,
    db: Optional[MatchupDB] = None,
    resume: bool = True,
    verbose: bool = True,
) -> list[IterationResult]:
    """
    Run the full iterative self-play training loop.

    Orchestrates data generation, policy training, and evaluation across
    `n_iterations` rounds.  Results are persisted to disk so the loop can
    be interrupted and resumed.

    Parameters
    ----------
    n_iterations    : total iterations to run (no hard cap — run as many as needed;
                      marginal gains typically plateau after 5-10 depending on
                      data volume and sim budget).
    data_dir        : root directory.  Sub-paths used:
                        data_dir/self_play/   — replay buffer batch files
                        data_dir/policy/      — saved policies + training log
    map_mode_pairs  : full (map, mode) pool from load_map_mode_pairs().
    config          : IterationConfig (defaults to IterationConfig() if None).
    evaluator       : pre-loaded FMEvaluator (loaded from data_dir if None).
    db              : pre-loaded MatchupDB (loaded from data_dir if None).
    resume          : if True, load existing replay buffer and training log from
                      data_dir and continue from the last completed iteration.
    verbose         : print iteration summaries (default True).

    Returns
    -------
    list of IterationResult, one per iteration run in this call.
    """
    if config is None:
        config = IterationConfig()

    if evaluator is None:
        evaluator = FMEvaluator(FMInference.load(data_dir / "fm_model.pkl"))
    if db is None:
        db = MatchupDB.load(data_dir / "matchup_db.pkl")

    # Fixed held-out eval map set — same for all iterations.
    eval_map_mode_pairs = select_eval_maps(map_mode_pairs, config.n_eval_maps_per_mode)
    if verbose:
        print(f"Eval map set ({len(eval_map_mode_pairs)} maps): {eval_map_mode_pairs}")

    # Replay buffer — resume from disk if available.
    sp_dir = data_dir / "self_play"
    replay_buffer = (
        ReplayBuffer.load(sp_dir) if resume and (sp_dir / "manifest.json").exists()
        else ReplayBuffer(sp_dir)
    )
    if verbose and replay_buffer.n_games > 0:
        print(f"Resumed replay buffer: {replay_buffer.n_games} games, {replay_buffer.n_records} records")

    # Load existing training log and current best policy if resuming.
    existing_log = _load_training_log(data_dir) if resume else []
    start_iteration = len(existing_log)
    current_policy: Optional[PolicyInference] = (
        _load_best_policy(data_dir) if resume else None
    )
    best_ever_policy: Optional[PolicyInference] = (
        _load_best_ever_policy(data_dir) if resume else None
    )
    if verbose and current_policy is not None:
        print(f"Loaded existing best policy (resuming from iteration {start_iteration})")

    new_results: list[IterationResult] = []

    for i in range(start_iteration, start_iteration + n_iterations):
        new_policy, result = run_iteration(
            iteration_num=i,
            replay_buffer=replay_buffer,
            current_policy=current_policy,
            best_ever_policy=best_ever_policy,
            evaluator=evaluator,
            db=db,
            map_mode_pairs=map_mode_pairs,
            eval_map_mode_pairs=eval_map_mode_pairs,
            config=config,
            data_dir=data_dir,
        )

        # Promote or keep.
        if result.promoted:
            current_policy = new_policy
            best_ever_policy = new_policy
        # If not promoted, both policies remain unchanged.
        # The new (rejected) policy was still saved as iter snapshot for analysis.

        # Persist training log after every iteration (safe to interrupt).
        result_dict = asdict(result)
        existing_log.append(result_dict)
        _save_training_log(data_dir, existing_log)

        new_results.append(result)

        if verbose:
            _print_iteration_summary(result)

    return new_results


# ── Summary helper ─────────────────────────────────────────────────────────────

def _print_iteration_summary(result: IterationResult) -> None:
    status = "✓ promoted" if result.promoted else "✗ rejected"
    print(
        f"  Iter {result.iteration:2d} | {result.n_games_generated} new games "
        f"({result.n_games_total} total) | {result.n_sims_per_pick} sims/pick | "
        f"val_kl={result.val_kl:.4f} | eval_win={result.eval_win_prob:.4f} | {status}"
    )


def load_training_log(data_dir: Path) -> list[dict]:
    """Load and return the persisted training log as a list of dicts."""
    return _load_training_log(data_dir)
