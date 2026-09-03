"""
workflow_utils.py — Helper functions for training_workflow.ipynb.

Keeps heavy or repetitive logic out of notebook cells so cells stay short
and focused on configuration and results.
"""

from __future__ import annotations

import json
from pathlib import Path


# ── Pipeline status ───────────────────────────────────────────────────────────

def check_pipeline_status(data_dir: Path) -> dict:
    """
    Scan data_dir for pipeline artifacts and print a checklist.

    Artifacts checked (in pipeline order):
      1. fm_model.pkl        — trained Factorization Machine
      2. feature_schema.pkl  — FM feature schema (built alongside the FM)
      3. matchup_db.pkl      — empirical counter/synergy lookup tables
      4. self_play/          — replay buffer (tracks total games generated)
      5. joint_net.pkl       — final joint policy+value network

    Returns a dict with keys:
      'fm'          : bool
      'schema'      : bool
      'matchup_db'  : bool
      'self_play_games' : int   (0 if manifest absent)
      'joint_net'   : bool
      'next_section': str       (human-readable label for where to jump in)
    """
    data_dir = Path(data_dir)

    fm_ok       = (data_dir / "fm_model.pkl").exists()
    schema_ok   = (data_dir / "feature_schema.pkl").exists()
    db_ok       = (data_dir / "matchup_db.pkl").exists()
    joint_ok    = (data_dir / "joint_net.pkl").exists()

    # Count total games from manifest (each game = 6 records)
    manifest_path = data_dir / "self_play" / "manifest.json"
    sp_games = 0
    if manifest_path.exists():
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        batches = manifest.get("batches", [])
        # n_records per batch ÷ 6 picks/game → game count
        sp_games = sum(b.get("n_records", 0) for b in batches) // 6

    # Determine where to resume
    if not fm_ok:
        next_section = "Section 1 — Train the Factorization Machine"
    elif not db_ok:
        next_section = "Section 2 — Build the Matchup Database"
    elif sp_games == 0:
        next_section = "Section 3 — Self-Play Data Generation (iteration 0)"
    elif not joint_ok:
        next_section = "Section 4 — Train the Joint Network"
    else:
        next_section = "All done! Use draft_playground.ipynb to run the agent."

    # ── Print checklist ───────────────────────────────────────────────────────
    w = 52
    print(f"Pipeline Status — {data_dir}")
    print("─" * w)
    _row("fm_model.pkl",       fm_ok,   "Factorization Machine")
    _row("feature_schema.pkl", schema_ok, "FM feature schema")
    _row("matchup_db.pkl",     db_ok,   "Matchup database")

    sp_label = f"{sp_games:,} games recorded" if sp_games else "0 games — run iteration 0"
    sp_ok = sp_games > 0
    _row("self_play/", sp_ok, sp_label)

    joint_label = "trained joint network" if joint_ok else "not yet trained"
    _row("joint_net.pkl", joint_ok, joint_label)

    print("─" * w)
    arrow = "▶" if next_section != "All done! Use draft_playground.ipynb to run the agent." else "✓"
    print(f"{arrow}  Next: {next_section}")

    return {
        "fm":             fm_ok,
        "schema":         schema_ok,
        "matchup_db":     db_ok,
        "self_play_games": sp_games,
        "joint_net":      joint_ok,
        "next_section":   next_section,
    }


def _row(label: str, ok: bool, detail: str = "") -> None:
    mark = "✓" if ok else "✗"
    detail_str = f"  ({detail})" if detail else ""
    print(f"  [{mark}] {label:<28}{detail_str}")


# ── Self-play speed benchmark ─────────────────────────────────────────────────

def benchmark_mcts_speed(
    data_dir: Path,
    map_mode_pairs: list,
    sim_counts: list[int] | None = None,
    *,
    n_test_games: int = 3,
    n_games_per_iter: int = 500,
    n_iters: int = 5,
    n_workers: int = 1,
) -> list[dict]:
    """
    Benchmark iteration-0 MCTS speed (no joint net, pure FM rollouts) at
    several simulation budgets.

    Runs ``n_test_games`` games serially at each sim count, then extrapolates
    an estimated wall-clock time for a full ``n_games_per_iter × n_iters`` run.
    When ``n_workers > 1``, a parallel estimate is also shown (assumes linear
    scaling, which is a lower bound — startup overhead makes real speedup
    slightly less than n_workers×).

    Call this cell once before committing to a long self-play run, then adjust
    ``N_SIMS_ITER0`` and ``N_GAMES_PER_ITER`` in cell 3.1 accordingly.

    Parameters
    ----------
    data_dir         : directory containing ``fm_model.pkl`` and ``matchup_db.pkl``.
    map_mode_pairs   : valid (map_name, mode) pairs from ``load_map_mode_pairs()``.
    sim_counts       : list of simulation budgets to test. Default: [500, 1000, 2000, 5000].
    n_test_games     : games to run per sim count (3 is enough for a stable estimate).
    n_games_per_iter : games per iteration used for the time extrapolation.
    n_iters          : iteration count used for the time extrapolation.
    n_workers        : if > 1, also prints a parallel time estimate (serial ÷ n_workers).

    Returns
    -------
    List of dicts with keys: ``n_sims``, ``secs_per_game``, ``games_per_min``,
    ``est_hrs_serial``, ``est_hrs_parallel`` (None when n_workers == 1).
    """
    import time

    # Lazy imports — avoids loading torch at workflow_utils import time.
    try:
        from fm_model import FMInference
        from fm_integration import FMEvaluator
        from matchup_db import MatchupDB
        from self_play import run_self_play_batch
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from fm_model import FMInference
        from fm_integration import FMEvaluator
        from matchup_db import MatchupDB
        from self_play import run_self_play_batch

    if sim_counts is None:
        sim_counts = [500, 1000, 2000, 5000]

    data_dir = Path(data_dir)
    total_games = n_games_per_iter * n_iters

    evaluator = FMEvaluator(FMInference.load(data_dir / "fm_model.pkl"))
    db = MatchupDB.load(data_dir / "matchup_db.pkl")

    show_parallel = n_workers > 1
    w = 74 if show_parallel else 60
    header = (
        f"{'sims/pick':>9}  {'sec/game':>9}  {'games/min':>9}  "
        f"{'serial hrs':>11}"
        + (f"  {'parallel hrs':>12}" if show_parallel else "")
    )
    label = f"({n_games_per_iter} games × {n_iters} iters = {total_games:,} total)"
    print(f"MCTS Speed Benchmark  —  {n_test_games} test games per sim count")
    print(f"Extrapolation target  :  {label}")
    print("─" * w)
    print(header)
    print("─" * w)

    results = []
    for n_sims in sim_counts:
        t0 = time.perf_counter()
        run_self_play_batch(
            n_games        = n_test_games,
            data_dir       = data_dir,
            map_mode_pairs = map_mode_pairs,
            evaluator      = evaluator,
            db             = db,
            n_workers      = 1,           # always serial for the benchmark itself
            seed           = 0,
            n_sims_per_pick= n_sims,
        )
        elapsed = time.perf_counter() - t0

        secs_per_game   = elapsed / n_test_games
        games_per_min   = 60.0 / secs_per_game if secs_per_game > 0 else 0.0
        est_hrs_serial  = secs_per_game * total_games / 3600.0
        est_hrs_par     = est_hrs_serial / n_workers if show_parallel else None

        marker = "  ← iter-0 default" if n_sims == 2000 else ""
        row = (
            f"{n_sims:>9,}  {secs_per_game:>9.2f}  {games_per_min:>9.1f}  "
            f"{est_hrs_serial:>11.2f}h"
            + (f"  {est_hrs_par:>12.2f}h" if show_parallel else "")
            + marker
        )
        print(row)
        results.append({
            "n_sims":           n_sims,
            "secs_per_game":    secs_per_game,
            "games_per_min":    games_per_min,
            "est_hrs_serial":   est_hrs_serial,
            "est_hrs_parallel": est_hrs_par,
        })

    print("─" * w)
    if show_parallel:
        print(f"Parallel estimate assumes linear scaling across {n_workers} workers.")
    else:
        print("Tip: set N_WORKERS > 1 in cell 3.1 to see parallel time estimates.")
    return results


# ── Head-to-head evaluation ───────────────────────────────────────────────────

def run_eval_games(
    new_joint,
    baseline_joint,
    data_dir: Path,
    db_path: Path,
    n_games: int,
    *,
    n_sims: int = 500,
    min_pick_rate: float = 0.005,
    puct_alpha: float = 0.7,
    seed: int = 42,
) -> dict:
    """
    Run ``n_games`` asymmetric head-to-head evaluation games: new_joint vs.
    baseline_joint (or FM-only MCTS when baseline_joint is None).

    For each game, new_joint controls one team and baseline_joint controls the
    other. P1/P2 assignment alternates by game index to eliminate first-pick
    advantage. Each team runs its own independent MCTS tree using its own
    policy and value head (or rollout for FM-only baseline).

    The metric is the FM-evaluated terminal win probability for new_joint's
    team, averaged across all games. A value > 0.5 means new_joint produces
    better drafts than the baseline; the threshold for promotion is set by
    the caller (PROMOTION_WIN_THR in cell 4.1).

    Parameters
    ----------
    new_joint       : newly trained JointNetInference to evaluate.
    baseline_joint  : current best JointNetInference, or None for FM-only.
    data_dir        : directory containing fm_model.pkl and matchup_db.pkl.
    db_path         : path to the season's SQLite database (for map/mode pairs).
    n_games         : number of head-to-head games to run (serial).
    n_sims          : MCTS simulation budget per pick (default 500; less than
                      training since we only need pick quality, not diversity).
    min_pick_rate   : MCTS tree vocab filter (default 0.005).
    puct_alpha      : PUCT prior weight for new_joint's tree (default 0.7).
    seed            : RNG seed for reproducibility (default 42).

    Returns
    -------
    dict with:
      'new_net_win_rate' : float        — mean P(new_net team wins)
      'n_games'          : int
      'win_probs'        : list[float]  — per-game FM win probabilities
      'elapsed_sec'      : float
    """
    import time
    import numpy as np

    # Lazy imports — avoid loading torch/numpy at module import time.
    try:
        from fm_model import FMInference
        from fm_integration import FMEvaluator
        from matchup_db import MatchupDB, skill_ns_to_tier
        from draft_state import DraftState, apply_pick
        from recommend import _run_mcts
        from rollout import (
            RolloutWeightCache,
            DEFAULT_MY_COUNTER_WEIGHT,
            DEFAULT_OPP_COUNTER_WEIGHT,
        )
        from mcts_node import UCB1_C
        from self_play import (
            _build_tree_vocab,
            _sample_pick,
            DEFAULT_SKILL_NS_CHOICES,
            load_map_mode_pairs,
        )
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from fm_model import FMInference
        from fm_integration import FMEvaluator
        from matchup_db import MatchupDB, skill_ns_to_tier
        from draft_state import DraftState, apply_pick
        from recommend import _run_mcts
        from rollout import (
            RolloutWeightCache,
            DEFAULT_MY_COUNTER_WEIGHT,
            DEFAULT_OPP_COUNTER_WEIGHT,
        )
        from mcts_node import UCB1_C
        from self_play import (
            _build_tree_vocab,
            _sample_pick,
            DEFAULT_SKILL_NS_CHOICES,
            load_map_mode_pairs,
        )

    data_dir = Path(data_dir)
    evaluator      = FMEvaluator(FMInference.load(data_dir / "fm_model.pkl"))
    db             = MatchupDB.load(data_dir / "matchup_db.pkl")
    map_mode_pairs = load_map_mode_pairs(db_path, known_maps=set(evaluator._fm.schema.maps))
    vocab          = evaluator._fm.schema.vocab

    rng = np.random.default_rng(seed)

    def _run_one_game(game_idx: int) -> float:
        """Return P(new_joint team wins) from the FM."""
        map_name, mode = map_mode_pairs[rng.integers(len(map_mode_pairs))]
        skill_ns       = float(DEFAULT_SKILL_NS_CHOICES[rng.integers(len(DEFAULT_SKILL_NS_CHOICES))])
        new_is_first   = (game_idx % 2 == 0)

        tier = (
            skill_ns_to_tier(skill_ns, db.skill_tier_boundaries)
            if db.skill_tier_boundaries is not None
            else 1
        )
        tree_vocab   = _build_tree_vocab(vocab, db, mode, map_name, tier, min_pick_rate)
        weight_cache = RolloutWeightCache(db, vocab)
        weight_cache._get(mode, map_name, tier)

        game_state = DraftState(
            my_team=frozenset(), opp_team=frozenset(),
            mode=mode, map_name=map_name, skill_ns=skill_ns,
            is_first_pick=new_is_first,
        )

        for _ in range(6):
            current_turn = game_state.whose_turn

            if current_turn == "mine":
                # new_joint's team picks
                mcts_state    = game_state
                pick_policy   = new_joint
                pick_value    = new_joint
            else:
                # baseline's team picks — mirror state so whose_turn == "mine"
                mcts_state = DraftState(
                    my_team=game_state.opp_team, opp_team=game_state.my_team,
                    mode=mode, map_name=map_name, skill_ns=skill_ns,
                    is_first_pick=not new_is_first,
                )
                pick_policy = baseline_joint   # None → counter-rate prior
                pick_value  = baseline_joint   # None → FM rollout

            root = _run_mcts(
                state               = mcts_state,
                evaluator           = evaluator,
                db                  = db,
                vocab               = vocab,
                n_simulations       = n_sims,
                my_counter_weight   = DEFAULT_MY_COUNTER_WEIGHT,
                opp_counter_weight  = DEFAULT_OPP_COUNTER_WEIGHT,
                min_pick_rate       = min_pick_rate,
                opp_rollout_mode    = "top_k",
                opp_top_k           = 3,
                my_rollout_mode     = "weighted",
                my_top_k            = 3,
                min_counter_games   = 10,
                c                   = UCB1_C,
                rng                 = rng,
                puct_alpha          = puct_alpha,
                tree_vocab          = tree_vocab,
                weight_cache        = weight_cache,
                policy              = pick_policy,
                value_net           = pick_value,
            )

            # Greedy pick in evaluation (no exploration noise)
            pick = _sample_pick(root, temperature=0.0, rng=rng)
            game_state = apply_pick(game_state, pick)

        # game_state is terminal; evaluate from new_joint's team perspective.
        return evaluator.evaluate(game_state)

    t0        = time.perf_counter()
    win_probs = [_run_one_game(i) for i in range(n_games)]
    elapsed   = time.perf_counter() - t0

    return {
        "new_net_win_rate": float(np.mean(win_probs)),
        "n_games":          n_games,
        "win_probs":        win_probs,
        "elapsed_sec":      elapsed,
    }


# ── Joint network validation ───────────────────────────────────────────────────

def validate_joint_net(
    joint_net,
    records,
    val_game_fraction: float = 0.1,
    n_policy_samples: int = 5,
    seed: int = 0,
) -> dict:
    """
    Run value-calibration and policy-sanity checks on a newly trained joint net.

    Replicates the logic from cell 4.4 of training_workflow.ipynb so that the
    Section 5 automated loop can call it without duplicating code.

    Parameters
    ----------
    joint_net         : JointNetInference to evaluate.
    records           : list of SelfPlayRecord (output of ReplayBuffer.all_records()).
    val_game_fraction : fraction of games held out for validation; must match the
                        val_game_fraction used in train_joint() so the val split
                        is the same and results are not over-optimistic.
    n_policy_samples  : number of val records to sample for policy top-1 agreement.
    seed              : RNG seed for the policy sample selection.

    Returns
    -------
    dict with keys:
        'cal_err'   : float  — mean calibration MAE across occupied 10-bin buckets
        'cal_pass'  : bool   — cal_err < 0.05
        'pol_top1'  : int    — top-1 agreement count (out of n_policy_samples)
        'pol_pass'  : bool   — pol_top1 >= ceil(n_policy_samples * 3/5)  [≥3/5 for n=5]
    """
    try:
        import numpy as np
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        import numpy as np

    # Replicate train_joint's chronological val split
    all_ids     = sorted({r.game_id for r in records})
    n_val_games = max(1, int(len(all_ids) * val_game_fraction))
    val_ids     = set(all_ids[-n_val_games:])
    val_records = [r for r in records if r.game_id in val_ids]

    # ── Value head calibration ─────────────────────────────────────────────────
    pred_probs = np.array([joint_net.evaluate(r.state) for r in val_records])
    true_probs = np.array([r.terminal_win_prob for r in val_records])

    n_bins    = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_mads  = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (pred_probs >= lo) & (pred_probs < hi)
        if mask.sum() == 0:
            continue
        bin_mads.append(abs(float(pred_probs[mask].mean()) - float(true_probs[mask].mean())))

    cal_err  = float(np.mean(bin_mads)) if bin_mads else 1.0
    cal_pass = cal_err < 0.05

    # ── Policy head top-1 agreement ────────────────────────────────────────────
    policy_records = [r for r in val_records if r.visit_dist]

    if len(policy_records) < n_policy_samples:
        # Not enough val records with visit distributions — don't block promotion.
        pol_top1 = n_policy_samples
        pol_pass = True
    else:
        rng        = np.random.default_rng(seed)
        sample_idx = rng.choice(len(policy_records), size=n_policy_samples, replace=False)
        sample     = [policy_records[i] for i in sample_idx]

        n_agree = 0
        for r in sample:
            used      = r.state.my_team | r.state.opp_team | (r.state.bans or frozenset())
            available = [b for b in joint_net.schema.vocab if b not in used]
            priors    = joint_net.predict_prior(r.state, available)
            joint_top = available[int(np.argmax(priors))]
            mcts_top  = max(r.visit_dist, key=r.visit_dist.get)
            n_agree  += int(joint_top == mcts_top)

        pol_top1 = n_agree
        # Pass threshold: ≥ 3/5 for n=5 (or ≥60% generally)
        pol_pass = n_agree >= max(1, n_policy_samples * 3 // 5)

    return {
        "cal_err":  cal_err,
        "cal_pass": cal_pass,
        "pol_top1": pol_top1,
        "pol_pass": pol_pass,
    }


# ── Rollout-free speed benchmark ───────────────────────────────────────────────

def benchmark_rollout_free_speed(
    data_dir: Path,
    db_path: Path,
    sim_counts: list[int] | None = None,
    *,
    n_warmup: int = 1,
    n_timed: int = 3,
    ucb1_c: float = 0.5,
    puct_alpha: float = 0.7,
    min_pick_rate: float = 0.005,
) -> list[dict]:
    """
    Benchmark per-pick MCTS speed for FM-only vs joint-net (rollout-free) modes.

    For each sim count, runs ``n_timed`` independent ``_run_mcts`` calls in each
    mode (after ``n_warmup`` discarded warm-up calls) and reports median time per
    pick.  The speedup factor shows how much faster rollout-free MCTS is at the
    same simulation budget.

    Parameters
    ----------
    data_dir     : directory containing ``fm_model.pkl``, ``matchup_db.pkl``,
                   and ``joint_net.pkl``.
    db_path      : path to the season's SQLite database (for map/mode pairs).
    sim_counts   : list of simulation budgets to test.
                   Default: [500, 1000, 2000, 5000, 10000].
    n_warmup     : calls to discard before timing (warms up Python / JIT).
    n_timed      : calls to time per mode per sim count.
    ucb1_c       : MCTS exploration constant.
    puct_alpha   : PUCT prior weight for the joint-net tree.
    min_pick_rate: brawlers below this pick rate are excluded from the tree vocab.

    Returns
    -------
    List of dicts with keys:
        ``n_sims``, ``fm_sec_per_pick``, ``joint_sec_per_pick``, ``speedup``.
        ``joint_sec_per_pick`` and ``speedup`` are None when joint_net.pkl is absent.
    """
    import time
    import statistics

    try:
        from fm_model      import FMInference
        from fm_integration import FMEvaluator
        from matchup_db    import MatchupDB, skill_ns_to_tier
        from draft_state   import DraftState
        from recommend     import _run_mcts
        from rollout       import (
            RolloutWeightCache,
            DEFAULT_MY_COUNTER_WEIGHT,
            DEFAULT_OPP_COUNTER_WEIGHT,
        )
        from mcts_node     import UCB1_C
        from self_play     import (
            _build_tree_vocab,
            DEFAULT_SKILL_NS_CHOICES,
            load_map_mode_pairs,
        )
        from joint_net     import JointNetInference
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from fm_model      import FMInference
        from fm_integration import FMEvaluator
        from matchup_db    import MatchupDB, skill_ns_to_tier
        from draft_state   import DraftState
        from recommend     import _run_mcts
        from rollout       import (
            RolloutWeightCache,
            DEFAULT_MY_COUNTER_WEIGHT,
            DEFAULT_OPP_COUNTER_WEIGHT,
        )
        from mcts_node     import UCB1_C
        from self_play     import (
            _build_tree_vocab,
            DEFAULT_SKILL_NS_CHOICES,
            load_map_mode_pairs,
        )
        from joint_net     import JointNetInference

    import numpy as np

    if sim_counts is None:
        sim_counts = [500, 1000, 2000, 5000, 10000]

    data_dir = Path(data_dir)

    evaluator      = FMEvaluator(FMInference.load(data_dir / "fm_model.pkl"))
    db             = MatchupDB.load(data_dir / "matchup_db.pkl")
    map_mode_pairs = load_map_mode_pairs(db_path, known_maps=set(evaluator._fm.schema.maps))
    vocab          = evaluator._fm.schema.vocab

    joint_path    = data_dir / "joint_net.pkl"
    joint_net     = JointNetInference.load(joint_path) if joint_path.exists() else None
    has_joint     = joint_net is not None

    # Fixed benchmark state: empty board, first available map/mode, mid-tier skill.
    map_name, mode = map_mode_pairs[0]
    skill_ns       = float(DEFAULT_SKILL_NS_CHOICES[len(DEFAULT_SKILL_NS_CHOICES) // 2])
    tier = (
        skill_ns_to_tier(skill_ns, db.skill_tier_boundaries)
        if db.skill_tier_boundaries is not None
        else 1
    )
    state        = DraftState(
        my_team=frozenset(), opp_team=frozenset(),
        mode=mode, map_name=map_name, skill_ns=skill_ns, is_first_pick=True,
    )
    tree_vocab   = _build_tree_vocab(vocab, db, mode, map_name, tier, min_pick_rate)
    weight_cache = RolloutWeightCache(db, vocab)
    weight_cache._get(mode, map_name, tier)
    rng          = np.random.default_rng(0)

    def _time_one(policy, value_net, n_sims):
        _run_mcts(
            state              = state,
            evaluator          = evaluator,
            db                 = db,
            vocab              = vocab,
            n_simulations      = n_sims,
            my_counter_weight  = DEFAULT_MY_COUNTER_WEIGHT,
            opp_counter_weight = DEFAULT_OPP_COUNTER_WEIGHT,
            min_pick_rate      = min_pick_rate,
            opp_rollout_mode   = "top_k",
            opp_top_k          = 3,
            my_rollout_mode    = "weighted",
            my_top_k           = 3,
            min_counter_games  = 10,
            c                  = ucb1_c,
            rng                = rng,
            puct_alpha         = puct_alpha,
            tree_vocab         = tree_vocab,
            weight_cache       = weight_cache,
            policy             = policy,
            value_net          = value_net,
        )

    # ── Print header ───────────────────────────────────────────────────────────
    w = 68 if has_joint else 46
    print(f"Rollout-Free Speed Benchmark  —  {map_name} / {mode}")
    print(f"  FM-only  : rollout simulations (baseline)")
    if has_joint:
        print(f"  Joint net: value head replaces rollouts (rollout-free)")
    print(f"  Timing   : {n_warmup} warm-up + {n_timed} timed calls, median reported")
    print("─" * w)
    hdr = f"  {'sims/pick':>9}  {'FM sec/pick':>11}  {'FM picks/min':>12}"
    if has_joint:
        hdr += f"  {'JN sec/pick':>11}  {'speedup':>7}"
    print(hdr)
    print("─" * w)

    results = []
    for n_sims in sim_counts:
        # Warm up
        for _ in range(n_warmup):
            _time_one(None, None, n_sims)

        # Time FM-only
        fm_times = []
        for _ in range(n_timed):
            t0 = time.perf_counter()
            _time_one(None, None, n_sims)
            fm_times.append(time.perf_counter() - t0)
        fm_sec = statistics.median(fm_times)

        # Time joint net (rollout-free)
        jn_sec    = None
        speedup   = None
        if has_joint:
            for _ in range(n_warmup):
                _time_one(joint_net, joint_net, n_sims)
            jn_times = []
            for _ in range(n_timed):
                t0 = time.perf_counter()
                _time_one(joint_net, joint_net, n_sims)
                jn_times.append(time.perf_counter() - t0)
            jn_sec  = statistics.median(jn_times)
            speedup = fm_sec / jn_sec if jn_sec > 0 else None

        row = (
            f"  {n_sims:>9,}  {fm_sec:>11.3f}  {60.0 / fm_sec:>12.1f}"
        )
        if has_joint:
            row += f"  {jn_sec:>11.3f}  {speedup:>6.2f}×"
        print(row)

        results.append({
            "n_sims":             n_sims,
            "fm_sec_per_pick":    fm_sec,
            "joint_sec_per_pick": jn_sec,
            "speedup":            speedup,
        })

    print("─" * w)

    # Recommend a production budget: largest sim count where joint picks/min ≥ 12
    # (i.e., pick decision in ≤ 5 s, comfortable for a live draft).
    if has_joint:
        recommended = None
        for r in results:
            if r["joint_sec_per_pick"] is not None and r["joint_sec_per_pick"] <= 5.0:
                recommended = r["n_sims"]
        if recommended:
            print(f"\nRecommended production budget: {recommended:,} sims/pick "
                  f"(rollout-free, ≤5s per pick)")
        else:
            print("\nNo sim count achieved ≤5s per pick on this hardware. "
                  "Consider reducing min_pick_rate or using N_SIMS < 500.")
    else:
        print("\njoint_net.pkl not found — train and promote a joint net (Sections 4–5)")
        print("to see rollout-free speed gains.")

    return results
