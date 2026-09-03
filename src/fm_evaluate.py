"""
fm_evaluate.py — Step 1.5: Calibration & Evaluation

Three evaluations on the held-out validation set:

  1.5.1  Scalar metrics: log-loss, AUC-ROC, Brier score.
  1.5.2  Calibration curves: overall + per-mode (subplots) + per-map (ECE table).
  1.5.3  Inference speed: encoding and forward pass profiled separately;
         MCTS simulation budget estimated from combined throughput.

Also provides FMEncoder — the optimised state encoder for MCTS use.

Usage:
  python src/fm_evaluate.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_prep import build_game_dataset  # noqa: E402
from feature_engineering import (  # noqa: E402
    FeatureSchema,
    build_feature_matrix,
    chronological_split,
)
from fm_model import FMInference, _csr_to_compact  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"


# ─── Batch inference ──────────────────────────────────────────────────────────

def batch_predict(
    fm: FMInference,
    idx: np.ndarray,
    val: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    """
    Vectorised FM forward pass over a (N, nnz) compact sparse matrix.

    Returns float32 predicted probabilities of shape (N,).
    Uses pure NumPy — no PyTorch overhead, fast for large datasets.
    """
    N = len(idx)
    probs = np.empty(N, dtype=np.float32)
    V_mat = fm.V          # (D, k) float32
    w0 = fm.w0            # scalar
    w_lin = fm.w_linear   # (D,)  float32

    for start in range(0, N, batch_size):
        ib = idx[start : start + batch_size]   # (B, nnz)  int32
        vb = val[start : start + batch_size]   # (B, nnz)  float32

        # Second-order interactions — vectorised over the batch
        emb = V_mat[ib]                           # (B, nnz, k)
        Vx = emb * vb[:, :, None]                 # (B, nnz, k)
        sum_Vx = Vx.sum(axis=1)                   # (B, k)

        linear = w0 + (w_lin[ib] * vb).sum(axis=1)          # (B,)
        interaction = 0.5 * (
            (sum_Vx * sum_Vx).sum(axis=1)
            - (Vx * Vx).sum(axis=(1, 2))
        )                                                     # (B,)

        logits = linear + interaction
        # Numerically stable sigmoid
        probs[start : start + batch_size] = np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-logits)),
            np.exp(logits) / (1.0 + np.exp(logits)),
        )

    return probs


# ─── Data loading helper ──────────────────────────────────────────────────────

def _load_val_predictions(fm: FMInference) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build the val set, run batch inference, and return three aligned objects:
      val_df   — original game-level DataFrame (N rows), with mode/map columns
      probs    — predicted P(team1 wins) for each game (N,)
      labels   — actual team1_wins for each game (N,)

    The feature matrix has 2N rows (original + flipped).  Only original
    (even-indexed) rows are returned, so probs[i] corresponds to val_df.iloc[i].
    """
    print("Loading validation data...")
    df_all, _ = build_game_dataset()
    _, val_df = chronological_split(df_all)
    del df_all

    schema = fm.schema
    X_val_csr, y_val = build_feature_matrix(val_df, schema)
    val_idx, val_val_arr = _csr_to_compact(X_val_csr)
    del X_val_csr

    print(f"  Running batch inference over {len(val_idx):,} augmented rows...")
    all_probs = batch_predict(fm, val_idx, val_val_arr)

    # Even rows = original orientation (team1 as 'my team'), odd = flipped
    orig_probs  = all_probs[0::2]
    orig_labels = y_val[0::2].astype(np.int8)

    assert len(orig_probs) == len(val_df), (
        f"Length mismatch: probs={len(orig_probs)}, val_df={len(val_df)}"
    )
    return val_df.reset_index(drop=True), orig_probs, orig_labels


# ─── Step 1.5.1 — Scalar metrics ─────────────────────────────────────────────

def evaluate_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Print and return log-loss, AUC-ROC, Brier score."""
    y = labels.astype(np.float32)
    logloss = float(log_loss(y, probs))
    auc     = float(roc_auc_score(y, probs))
    brier   = float(np.mean((probs - y) ** 2))

    print("\n── 1.5.1: Validation Metrics ──────────────────────────────────")
    print(f"  Log-loss : {logloss:.4f}  (random ≈ 0.693 | target < 0.685)")
    print(f"  AUC-ROC  : {auc:.4f}  (random = 0.500 | target > 0.550)")
    print(f"  Brier    : {brier:.4f}  (random = 0.250 | target < 0.245)")

    return {"logloss": logloss, "auc": auc, "brier": brier}


# ─── Step 1.5.2 — Calibration curves ─────────────────────────────────────────

def _calibration_bins(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Equal-frequency binning: returns (mean_pred, actual_frac, counts) per bin.
    Uses quantiles so each bin has roughly equal sample count.
    """
    order = np.argsort(probs)
    p_sorted = probs[order]
    y_sorted = labels[order].astype(np.float32)

    edges = np.linspace(0, len(p_sorted), n_bins + 1, dtype=int)
    mean_pred = np.empty(n_bins, dtype=np.float64)
    actual    = np.empty(n_bins, dtype=np.float64)
    counts    = np.empty(n_bins, dtype=np.int64)

    for i in range(n_bins):
        sl = slice(edges[i], edges[i + 1])
        mean_pred[i] = p_sorted[sl].mean()
        actual[i]    = y_sorted[sl].mean()
        counts[i]    = edges[i + 1] - edges[i]

    return mean_pred, actual, counts


def _ece(mean_pred: np.ndarray, actual: np.ndarray, counts: np.ndarray) -> float:
    """Expected Calibration Error (weighted mean absolute error across bins)."""
    w = counts / counts.sum()
    return float(np.dot(w, np.abs(mean_pred - actual)))


def _draw_cal_axis(
    ax: plt.Axes,
    probs: np.ndarray,
    labels: np.ndarray,
    title: str,
    n_bins: int = 10,
) -> float:
    """Draw one calibration subplot.  Returns ECE (for annotation)."""
    if len(probs) < n_bins * 5:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title(title, fontsize=9)
        return float("nan")

    mp, af, counts = _calibration_bins(probs, labels, n_bins)
    ece = _ece(mp, af, counts)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    ax.plot(mp, af, "o-", ms=4, lw=1.5)
    ax.fill_between(mp, mp, af, alpha=0.12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability", fontsize=8)
    ax.set_ylabel("Actual win rate", fontsize=8)
    ax.set_title(f"{title}\n(n={len(probs):,}, ECE={ece:.4f})", fontsize=8)
    ax.tick_params(labelsize=7)
    return ece


def plot_calibration_curves(
    val_df: pd.DataFrame,
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    save: bool = True,
) -> None:
    """
    Plot calibration curves: 1 overall + 1 per mode.
    Saves to figures/calibration_curves.png.
    """
    print("\n── 1.5.2: Calibration Curves ──────────────────────────────────")

    modes = sorted(val_df["mode"].unique())
    n_cols = len(modes) + 1           # overall + one per mode
    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 3.5))
    fig.suptitle("FM Calibration: predicted probability vs. actual win rate",
                 fontsize=10, y=1.01)

    # Overall
    ece_overall = _draw_cal_axis(axes[0], probs, labels, "Overall", n_bins)

    # Per mode
    mode_eces: dict[str, float] = {}
    for ax, mode in zip(axes[1:], modes):
        mask = val_df["mode"].values == mode
        ece_m = _draw_cal_axis(ax, probs[mask], labels[mask], mode, n_bins)
        mode_eces[mode] = ece_m

    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "calibration_curves.png"
        plt.savefig(path, dpi=130, bbox_inches="tight")
        print(f"  Saved → {path}")

    plt.close(fig)

    print(f"  Overall ECE : {ece_overall:.4f}")
    for mode, ece in sorted(mode_eces.items(), key=lambda x: x[1]):
        print(f"  {mode:<20}  ECE={ece:.4f}")


def print_permap_calibration(
    val_df: pd.DataFrame,
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> None:
    """
    Print a per-map calibration quality table sorted by ECE.

    ECE = expected calibration error (lower = better calibrated).
    Also shows mean predicted probability and actual win rate for sanity.
    """
    maps = sorted(val_df["map"].unique())
    rows: list[dict] = []

    for m in maps:
        mask = val_df["map"].values == m
        n = mask.sum()
        if n < n_bins * 5:
            continue
        p, y = probs[mask], labels[mask]
        mp, af, cnts = _calibration_bins(p, y, n_bins)
        ece = _ece(mp, af, cnts)
        mode_for_map = val_df.loc[mask, "mode"].iloc[0]
        rows.append({
            "map": m,
            "mode": mode_for_map,
            "n": n,
            "ECE": ece,
            "mean_pred": float(p.mean()),
            "actual_wr": float(y.mean()),
        })

    df_out = pd.DataFrame(rows).sort_values("ECE")

    print("\n── Per-map calibration (ECE: lower = better calibrated) ───────")
    print(f"{'Map':<28}  {'Mode':<14}  {'N':>7}  {'ECE':>6}  {'MeanPred':>9}  {'ActualWR':>9}")
    print("─" * 82)
    for _, r in df_out.iterrows():
        print(
            f"{r['map']:<28}  {r['mode']:<14}  {r['n']:>7,}  "
            f"{r['ECE']:>6.4f}  {r['mean_pred']:>9.4f}  {r['actual_wr']:>9.4f}"
        )


# ─── FMEncoder — optimised state encoder for MCTS ────────────────────────────

class FMEncoder:
    """
    Converts a draft state into the (indices, values) arrays expected by
    FMInference.evaluate_sparse().

    All index offsets are pre-computed at construction time.  Per-call cost is
    9 dict lookups + 9 integer writes into a pre-allocated buffer — no Python
    list construction or numpy array allocation.

    Convention (must match FMInference.evaluate_sparse):
        my_brawlers  → t1 feature slots
        opp_brawlers → t2 feature slots

    Usage:
        encoder = FMEncoder(fm)
        idx, val = encoder.encode(my_brawlers, opp_brawlers, map_name, mode_name, skill_ns)
        prob = fm.evaluate_sparse(idx, val)
    """

    __slots__ = (
        "_t1", "_t2", "_map_idx", "_mode_idx",
        "_skill_offset", "_idx_buf", "_val_buf",
    )

    def __init__(self, fm: FMInference) -> None:
        schema = fm.schema
        self._t1       = {b: schema.t1_offset  + i for i, b in enumerate(schema.vocab)}
        self._t2       = {b: schema.t2_offset  + i for i, b in enumerate(schema.vocab)}
        self._map_idx  = {m: schema.map_offset  + i for i, m in enumerate(schema.maps)}
        self._mode_idx = {m: schema.mode_offset + i for i, m in enumerate(schema.modes)}
        self._skill_offset = schema.skill_offset

        # Pre-allocated output buffers.  Binary features stay 1.0; skill_ns
        # is the only entry that changes between calls.
        self._idx_buf = np.empty(9, dtype=np.int32)
        self._val_buf = np.ones(9, dtype=np.float32)

    def encode(
        self,
        my_brawlers: tuple[str, str, str],
        opp_brawlers: tuple[str, str, str],
        map_name: str,
        mode_name: str,
        skill_ns: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (idx_buf, val_buf) pointing into pre-allocated internal arrays.

        Do NOT retain references across calls — the buffers are mutated in place.
        Copy if you need to store the result:  idx = encoder.encode(...)[0].copy()
        """
        b = self._idx_buf
        b[0] = self._t1[my_brawlers[0]]
        b[1] = self._t1[my_brawlers[1]]
        b[2] = self._t1[my_brawlers[2]]
        b[3] = self._t2[opp_brawlers[0]]
        b[4] = self._t2[opp_brawlers[1]]
        b[5] = self._t2[opp_brawlers[2]]
        b[6] = self._map_idx[map_name]
        b[7] = self._mode_idx[mode_name]
        b[8] = self._skill_offset
        self._val_buf[8] = skill_ns
        return self._idx_buf, self._val_buf


# ─── Step 1.5.3 — Inference speed benchmark ──────────────────────────────────

def benchmark_inference(fm: FMInference, n_calls: int = 30_000) -> dict[str, float]:
    """
    Profile the full MCTS evaluation pipeline in three stages:

      1. Feature encoding only   — FMEncoder.encode()
      2. FM forward pass only    — FMInference.evaluate_sparse()
      3. Combined pipeline       — encoding + forward pass end-to-end

    Reports evaluations/sec for each stage, identifies the bottleneck, and
    estimates MCTS simulation capacity within a 2-second budget.

    Returns a dict with keys: 'encoding', 'forward', 'combined' (evals/sec).
    """
    schema = fm.schema
    encoder = FMEncoder(fm)
    rng = np.random.default_rng(42)

    # Pre-generate a pool of realistic states to cycle through
    POOL = min(n_calls, 500)
    vocab = schema.vocab
    maps  = schema.maps
    modes = schema.modes

    pool_states = [
        (
            tuple(rng.choice(vocab, 3, replace=False)),
            tuple(rng.choice(vocab, 3, replace=False)),
            maps[int(rng.integers(len(maps)))],
            modes[int(rng.integers(len(modes)))],
            float(rng.normal(0.5, 1.0)),
        )
        for _ in range(POOL)
    ]

    # Pre-encode states for the forward-pass-only benchmark
    # (make copies so the buffer mutation doesn't matter)
    pre_encoded = [
        (encoder.encode(*s)[0].copy(), encoder.encode(*s)[1].copy())
        for s in pool_states
    ]

    print("\n── 1.5.3: Inference Speed Benchmark ───────────────────────────")

    # ── 1. Encoding only ──
    t0 = time.perf_counter()
    for i in range(n_calls):
        encoder.encode(*pool_states[i % POOL])
    enc_elapsed = time.perf_counter() - t0
    enc_speed = n_calls / enc_elapsed
    print(f"  Feature encoding :  {enc_speed:>12,.0f}  calls/sec")

    # ── 2. Forward pass only ──
    t0 = time.perf_counter()
    for i in range(n_calls):
        idx, val = pre_encoded[i % POOL]
        fm.evaluate_sparse(idx, val)
    fwd_elapsed = time.perf_counter() - t0
    fwd_speed = n_calls / fwd_elapsed
    print(f"  FM forward pass  :  {fwd_speed:>12,.0f}  evals/sec")

    # ── 3. Combined pipeline (encoding + inference) — MCTS figure of merit ──
    t0 = time.perf_counter()
    for i in range(n_calls):
        idx, val = encoder.encode(*pool_states[i % POOL])
        fm.evaluate_sparse(idx, val)
    combined_elapsed = time.perf_counter() - t0
    combined_speed = n_calls / combined_elapsed

    TARGET = 10_000
    status = "PASS" if combined_speed >= TARGET else "FAIL — consider caching or batching"
    print(f"  Combined pipeline:  {combined_speed:>12,.0f}  evals/sec  [{status}]")

    sims_2s = int(combined_speed * 2)
    branch  = 90  # approximate branching factor at start of draft
    print(f"\n  2-second MCTS budget  → ~{sims_2s:>6,} simulations")
    print(f"  Per root child (~{branch} brawlers)  → ~{sims_2s // branch:>4,} visits")

    bottleneck = "encoding" if enc_elapsed > fwd_elapsed else "FM forward pass"
    print(f"  Bottleneck: {bottleneck}")

    return {"encoding": enc_speed, "forward": fwd_speed, "combined": combined_speed}


# ─── Main entry point ─────────────────────────────────────────────────────────

# ─── Part 5.1.1 — FM calibration gate for value network training ──────────────

def check_fm_calibration(
    fm: FMInference,
    logloss_threshold: float = 0.685,
) -> dict:
    """
    Check stored FM validation metrics against the thresholds required for safe
    value network training (Part 5.1.1).

    Uses metrics recorded at FM training time — no val set reload needed.
    Returns a dict with metric values and a ``'pass'`` key (True = safe to proceed).

    If ``'pass'`` is False, the FM is poorly calibrated and value labels from
    ``terminal_win_prob`` will be too noisy for reliable value head training.

    Parameters
    ----------
    fm                : trained FMInference (metrics stored from training time).
    logloss_threshold : fail if val_logloss >= this value (default 0.685).
    """
    logloss = fm.val_logloss
    auc     = fm.val_auc
    brier   = fm.val_brier

    passed = not math.isnan(logloss) and logloss < logloss_threshold

    result: dict = {
        "val_logloss":       logloss,
        "val_auc":           auc,
        "val_brier":         brier,
        "logloss_threshold": logloss_threshold,
        "pass":              passed,
    }

    status = "PASS" if passed else "FAIL"
    print(f"FM calibration check [{status}]:")
    print(f"  val_logloss : {logloss:.4f}  (threshold < {logloss_threshold:.3f})")
    print(f"  val_auc     : {auc:.4f}  (target > 0.550)")
    print(f"  val_brier   : {brier:.4f}  (target < 0.245)")
    if not passed:
        print(
            "  WARNING: FM log-loss does not meet the threshold. "
            "Value network labels (terminal_win_prob) may be too noisy. "
            "Consider retraining the FM before proceeding to Part 5."
        )
    return result


def run_evaluation(save_figures: bool = True) -> None:
    """Run all Step 1.5 evaluations in order."""
    print("=== Step 1.5: Calibration & Evaluation ===\n")

    # Load trained model
    fm = FMInference.load()
    print(f"Loaded FM: val_logloss={fm.val_logloss:.4f}, val_auc={fm.val_auc:.4f}, "
          f"val_brier={fm.val_brier:.4f}")

    # Load val predictions once (shared across all sub-steps)
    val_df, probs, labels = _load_val_predictions(fm)
    print(f"  Val set: {len(val_df):,} games, label balance = {labels.mean():.4f}")

    # 1.5.1 — Scalar metrics
    evaluate_metrics(probs, labels)

    # 1.5.2 — Calibration curves
    plot_calibration_curves(val_df, probs, labels, save=save_figures)
    print_permap_calibration(val_df, probs, labels)

    # 1.5.3 — Inference speed
    benchmark_inference(fm)

    print("\n=== Step 1.5 complete ===")


if __name__ == "__main__":
    run_evaluation()
