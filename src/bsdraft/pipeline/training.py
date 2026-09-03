"""
training_utils.py — Helpers for training_guide.ipynb

Keeps notebook cells short and the logic testable. All heavy lifting (self-play
generation, training, evaluation) stays in the original src/ modules; this file
only adds convenience wrappers and display utilities.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

# ── Status checking ────────────────────────────────────────────────────────────

def check_status(data_dir: Path) -> dict:
    """
    Inspect data_dir and return a summary dict of what training artifacts exist.

    Keys
    ----
    fm_model, matchup_db, feature_schema, joint_net, policy_best : bool
    n_policy_iters : int   (number of policy_iter_*.pkl files found)
    n_sp_games     : int   (total games in the replay buffer)
    n_sp_records   : int   (total records = n_sp_games × 6)
    training_log   : list  (IterationResult dicts from train_policy.py, may be [])
    """
    d = Path(data_dir)

    # Self-play buffer stats via manifest
    sp_dir = d / "self_play"
    manifest_path = sp_dir / "manifest.json"
    n_sp_records = 0
    if manifest_path.exists():
        with open(manifest_path) as fh:
            mf = json.load(fh)
        n_sp_records = sum(b.get("n_records", 0) for b in mf.get("batches", []))
    n_sp_games = n_sp_records // 6  # each game always produces exactly 6 records

    # Policy iterations
    policy_dir = d / "policy"
    n_policy_iters = 0
    if policy_dir.exists():
        n_policy_iters = len(sorted(policy_dir.glob("policy_iter_*.pkl")))

    # Training log (from run_iteration_loop)
    log_path = d / "policy" / "training_log.json"
    training_log: list = []
    if log_path.exists():
        try:
            with open(log_path) as fh:
                training_log = json.load(fh)
        except Exception:
            pass

    return {
        "fm_model":       (d / "fm_model.pkl").exists(),
        "matchup_db":     (d / "matchup_db.pkl").exists(),
        "feature_schema": (d / "feature_schema.pkl").exists(),
        "joint_net":      (d / "joint_net.pkl").exists(),
        "policy_best":    (policy_dir / "policy_best.pkl").exists() if policy_dir.exists() else False,
        "n_policy_iters": n_policy_iters,
        "n_sp_games":     n_sp_games,
        "n_sp_records":   n_sp_records,
        "training_log":   training_log,
    }


def print_status(status: dict) -> None:
    """Pretty-print the dict returned by check_status()."""
    def ok(v):
        return "✓" if v else "✗"
    print("=" * 50)
    print("  Training Status")
    print("=" * 50)
    print(f"  {ok(status['fm_model'])}  Factorization Machine trained")
    print(f"  {ok(status['matchup_db'])}  Matchup DB built")
    print(f"  {ok(status['feature_schema'])}  Feature schema saved")
    print()
    print(f"     Self-play games : {status['n_sp_games']:>6,}")
    print(f"     Records in buf  : {status['n_sp_records']:>6,}")
    print(f"     Policy iters    : {status['n_policy_iters']:>6}")
    print(f"  {ok(status['policy_best'])}  Best standalone policy saved")
    print(f"  {ok(status['joint_net'])}  Joint policy+value net trained")
    print("=" * 50)

    # Contextual next-step hint
    if not status["fm_model"]:
        print("\n→ Start at Phase 1: train the FM and build the matchup DB.")
    elif status["n_sp_games"] < 500:
        needed = 500 - status["n_sp_games"]
        print(f"\n→ Phase 2: generate self-play data (~{needed} more games needed for a first policy).")
    elif not status["policy_best"] and status["n_sp_games"] < 2000:
        print("\n→ Phase 2/3: run initial policy iterations (train_policy loop).")
    elif not status["joint_net"]:
        print("\n→ Phase 4: train the joint policy+value network.")
    else:
        print(f"\n→ Phase 5: run more joint iterations to improve ({status['n_sp_games']:,} games so far).")


# ── Speed benchmarking ─────────────────────────────────────────────────────────

def benchmark_fm(evaluator, n_calls: int = 10_000) -> float:
    """Return FM evaluations per second (uses the built-in FMInference.benchmark)."""
    return evaluator._fm.benchmark(n_calls=n_calls)


def benchmark_joint_net(joint, n_calls: int = 5_000) -> float:
    """
    Return joint net value-head evaluations per second.

    Uses a representative partial state (pick_number=3, both teams have 1 brawler).
    """
    from bsdraft.mcts.state import DraftState

    state = DraftState(
        my_team=frozenset({"BULL"}),
        opp_team=frozenset({"SHELLY"}),
        mode="gemGrab",
        map_name="Double Swoosh",
        skill_ns=2.0,
        is_first_pick=True,
    )
    # Warm up
    for _ in range(20):
        joint.evaluate(state)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        joint.evaluate(state)
    elapsed = time.perf_counter() - t0
    return n_calls / elapsed


def benchmark_self_play_serial(
    evaluator,
    db,
    map_mode_pairs: list,
    *,
    n_probe: int = 3,
    n_sims_per_pick: int = 2_000,
    value_net=None,
    policy=None,
) -> dict:
    """
    Time n_probe self-play games serially and return speed estimates.

    Returns dict with keys:
      sec_per_game  : float
      games_per_hr  : float  (serial)
      sims_per_sec  : float  (total MCTS simulations per second)
    """
    from bsdraft.selfplay.generate import run_self_play_game

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for i in range(n_probe):
        run_self_play_game(
            evaluator, db,
            game_idx=999_000 + i,
            map_mode_pairs=map_mode_pairs,
            n_sims_per_pick=n_sims_per_pick,
            rng=rng,
            policy=policy,
            value_net=value_net,
        )
    elapsed = time.perf_counter() - t0

    sec_per_game = elapsed / n_probe
    games_per_hr = 3600.0 / sec_per_game
    picks_per_game = 6
    sims_per_sec = (n_sims_per_pick * picks_per_game) / sec_per_game

    return {
        "sec_per_game": sec_per_game,
        "games_per_hr": games_per_hr,
        "sims_per_sec": sims_per_sec,
    }


def print_speed_report(
    fm_eps: float,
    serial_stats: dict,
    n_workers: int,
    joint_eps: float | None = None,
) -> None:
    """Print a formatted speed report."""
    n_cpu = os.cpu_count() or 4
    eff = 0.80  # typical multiprocessing efficiency on macOS/Linux

    print("=" * 55)
    print("  Speed Report")
    print("=" * 55)
    print(f"  FM evaluations/sec         : {fm_eps:>10,.0f}")
    if joint_eps is not None:
        print(f"  Joint net evals/sec (value): {joint_eps:>10,.0f}")
    print()
    print("  Self-play (serial, 1 worker):")
    print(f"    {serial_stats['sec_per_game']:.1f}s / game  "
          f"→  {serial_stats['games_per_hr']:.0f} games/hr  "
          f"  ({serial_stats['sims_per_sec']:,.0f} sims/sec)")
    print()
    print("  Estimated parallel speed (multiprocessing):")
    for nw in [2, 4, 8]:
        if nw <= n_cpu:
            gph = serial_stats["games_per_hr"] * nw * eff
            hrs_per_1k = 1000 / gph
            print(f"    {nw} workers: {gph:>6.0f} games/hr  "
                  f"(1,000 games ≈ {hrs_per_1k:.1f} hr)")
    print("=" * 55)

    rec_workers = min(max(2, n_cpu - 1), 8)
    print(f"\n  Recommended n_workers for this machine: {rec_workers}  "
          f"({n_cpu} logical CPUs detected)")


# ── Joint network training iteration ─────────────────────────────────────────

def run_joint_iteration(
    n_games: int,
    data_dir: Path,
    map_mode_pairs: list,
    schema,
    *,
    n_workers: int = 4,
    n_sims_per_pick: int = 2_000,
    lambda_v: float = 1.0,
    lr: float = 1e-3,
    max_epochs: int = 50,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple:
    """
    One complete joint training iteration:

      1. Generate n_games self-play games.
         Workers automatically load joint_net.pkl from disk (rollout-free MCTS).
         Serial mode (n_workers=1) uses FM rollouts — use n_workers >= 2 to get
         the value net speedup.
      2. Retrain joint net from scratch on ALL accumulated records.
      3. Overwrite joint_net.pkl with the new weights.

    Returns
    -------
    joint       : JointNetInference  (the freshly trained model)
    n_total_games : int              (total games now in buffer)
    loss_history  : list             ([(epoch, pol_train, val_train, val_comb), ...])
    """
    from bsdraft.selfplay.generate import ReplayBuffer, run_self_play_batch
    from bsdraft.selfplay.joint_net import train_joint

    data_dir = Path(data_dir)
    buf = ReplayBuffer.load(data_dir / "self_play")
    start_idx = buf.n_games

    if verbose:
        jnet_status = "joint_net.pkl exists — workers will use rollout-free MCTS" \
                      if (data_dir / "joint_net.pkl").exists() \
                      else "no joint_net.pkl yet — workers will use FM rollouts"
        print(f"Generating {n_games:,} games  ({jnet_status})")
        print(f"  Buffer before: {buf.n_games:,} games  |  {n_workers} workers  "
              f"|  {n_sims_per_pick:,} sims/pick")

    t0 = time.perf_counter()
    run_self_play_batch(
        n_games=n_games,
        data_dir=data_dir,
        map_mode_pairs=map_mode_pairs,
        replay_buffer=buf,
        n_workers=n_workers,
        start_game_idx=start_idx,
        seed=seed,
        n_sims_per_pick=n_sims_per_pick,
    )
    gen_time = time.perf_counter() - t0

    if verbose:
        print(f"  Generated in {gen_time / 60:.1f} min  "
              f"→  buffer now: {buf.n_games:,} games")

    records = buf.all_records()
    if verbose:
        print(f"\nTraining joint net on {len(records):,} records "
              f"(+ {len(records):,} augmented for value head)...")

    loss_history: list = []
    t1 = time.perf_counter()
    joint = train_joint(
        records, schema,
        lambda_v=lambda_v,
        lr=lr,
        max_epochs=max_epochs,
        verbose=verbose,
        loss_history=loss_history,
    )
    train_time = time.perf_counter() - t1

    joint.save(data_dir / "joint_net.pkl")
    if verbose:
        print(f"  Trained in {train_time:.0f}s  →  joint_net.pkl saved")
        print(f"  Total wall time: {(gen_time + train_time) / 60:.1f} min")

    return joint, buf.n_games, loss_history


# ── Progress visualisation ────────────────────────────────────────────────────

def plot_training_log(training_log: list, ax=None) -> None:
    """
    Plot eval win prob per policy iteration from a training_log list
    (as saved by run_iteration_loop in train_policy.py).
    """
    import matplotlib.pyplot as plt

    if not training_log:
        print("No training log entries to plot.")
        return

    iters = [r.get("iteration", i) for i, r in enumerate(training_log)]
    win_probs = [r.get("eval_mean_win_prob", float("nan")) for r in training_log]
    promoted = [r.get("promoted", False) for r in training_log]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    ax.plot(iters, win_probs, marker="o", color="steelblue", label="Eval win prob")
    ax.axhline(0.525, color="crimson", linestyle="--", lw=1, label="Promotion threshold (0.525)")
    ax.axhline(0.5, color="gray", linestyle=":", lw=1, label="Baseline (0.5)")

    for i, wp, pr in zip(iters, win_probs, promoted, strict=True):
        if pr:
            ax.annotate("promoted", (i, wp), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color="forestgreen")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean FM win prob (eval games)")
    ax.set_title("Policy iteration history")
    ax.legend(fontsize=8)
    ax.set_ylim(0.45, 0.65)


def plot_joint_loss(loss_history: list, ax1=None, ax2=None) -> None:
    """Plot policy and value loss curves from a joint net training run."""
    import matplotlib.pyplot as plt

    if not loss_history:
        print("No loss history to plot.")
        return

    epochs    = [h[0] for h in loss_history]
    pol_train = [h[1] for h in loss_history]
    val_train = [h[2] for h in loss_history]
    val_comb  = [h[3] for h in loss_history]

    if ax1 is None or ax2 is None:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, pol_train, color="steelblue", label="Policy (train)")
    ax1.set_title("Policy head — cross-entropy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, val_train, color="darkorange", label="Value (train)")
    ax2.plot(epochs, val_comb,  color="gray", linestyle="--", label="Combined (val)")
    ax2.axhline(0.693, color="crimson", linestyle=":", lw=1, label="Random baseline")
    ax2.set_title("Value head — BCE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()


# ── Recommendation quality spot-check ─────────────────────────────────────────

def spot_check_recommendations(
    states: list,
    evaluator,
    db,
    schema,
    *,
    n_simulations: int = 2_000,
    policy=None,
    value_net=None,
    n_rollout_eval: int = 30,
    rng=None,
) -> list[dict]:
    """
    For each DraftState in `states`, run recommend() and then evaluate the top-1
    pick with n_rollout_eval random completions.

    Returns a list of dicts with keys:
      state, top1, visit_frac, rollout_fm_mean
    """
    from bsdraft.mcts.recommend import recommend
    from bsdraft.mcts.state import apply_pick, available_actions

    if rng is None:
        rng = np.random.default_rng(0)

    results = []
    for state in states:
        result = recommend(
            state.my_team, state.opp_team,
            state.mode, state.map_name, state.skill_ns,
            is_first_pick=state.is_first_pick, bans=state.bans,
            n_simulations=n_simulations, n_top=5,
            evaluator=evaluator, db=db,
            policy=policy, value_net=value_net,
        )
        top1 = result.top_picks[0]["brawler"]
        visit_frac = result.top_picks[0]["visit_fraction"]

        # Evaluate top-1 via random completions
        state_after = apply_pick(state, top1)
        fm_scores = []
        for _ in range(n_rollout_eval):
            s = state_after
            while not s.is_terminal:
                avail = available_actions(s, schema.vocab)
                s = apply_pick(s, rng.choice(avail))
            fm_scores.append(evaluator.evaluate(s))

        results.append({
            "state": state,
            "top1": top1,
            "visit_frac": visit_frac,
            "rollout_fm_mean": float(np.mean(fm_scores)),
            "top5": [(p["brawler"], p["visit_fraction"]) for p in result.top_picks],
        })
    return results


def print_spot_check(results: list[dict], label: str = "") -> None:
    """Pretty-print spot_check_recommendations output."""
    if label:
        print(f"\n{label}")
    print(f"{'#':>2}  {'Depth':>5}  {'Top-1 pick':>14}  {'Visit%':>7}  {'FM eval':>8}")
    print("-" * 45)
    fm_vals = []
    for i, r in enumerate(results):
        depth = r["state"].pick_number
        print(f"{i+1:>2}  {depth:>5}  {r['top1']:>14}  {r['visit_frac']:>6.1%}  "
              f"{r['rollout_fm_mean']:>8.4f}")
        fm_vals.append(r["rollout_fm_mean"])
    print(f"{'Mean':>25}  {'':>7}  {np.mean(fm_vals):>8.4f}")
