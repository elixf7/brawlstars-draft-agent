"""
fm_model.py — Step 1.4: FM Model Design & Training

Factorization Machine for Brawl Stars 3v3 draft win-probability prediction.

Design decisions (see IMPLEMENTATION_CHECKLIST.md §1.4 for full rationale):
  k=32  — rich enough for 95 brawlers' within-archetype variation across all
            interaction axes (synergy, counter, map fit, skill scaling). Inference
            cost is O(9×32)=288 multiply-adds — negligible vs. Python call overhead.

  Uniform λ=1e-4 L2  — maps occur at roughly equal rates (randomly assigned in
                        ranked). Sparse brawler interactions regularize toward zero
                        naturally (appropriate prior: uncertain ≈ neutral).

  Chronological train/val split  — correct evaluation methodology; prevents
                                   temporal leakage regardless of retraining policy.

  Sparse-aware forward pass  — compact (N, 9) index+value arrays instead of dense
                               (N, 223) matrices, keeping memory usage under 500 MB.

  NumPy inference wrapper  — ~40–80k single-sample evals/sec at MCTS time, vs
                             ~2–5k/sec going through PyTorch autograd per sample.

Usage:
  # Train (one-time):
  python src/fm_model.py

  # Load for MCTS inference:
  from bsdraft.fm.model import FMInference
  inf = FMInference.load()
  prob = inf.evaluate_sparse(indices, values)  # indices/values for 9 non-zero features
"""

from __future__ import annotations

import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score

from bsdraft.data.prep import build_game_dataset
from bsdraft.data.sources import DatasetRef
from bsdraft.features.engineering import (
    SCHEMA_PATH,
    FeatureSchema,
    build_feature_matrix,
    chronological_split,
)

# src/bsdraft/<subpackage>/<module>.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
MODEL_PATH = DATA_DIR / "fm_model.pkl"

# ── Hyperparameters ──────────────────────────────────────────────────────────
K = 32              # embedding dimension
LR = 1e-3           # Adam learning rate
WEIGHT_DECAY = 1e-4  # uniform L2 regularisation (λ)
BATCH_SIZE = 4096
MAX_EPOCHS = 50
PATIENCE = 5        # consecutive epochs without val improvement before stopping


# ── Model ────────────────────────────────────────────────────────────────────

class FactorizationMachine(nn.Module):
    """
    Logistic FM over sparse inputs (index + value format).

    FM(x) = σ(w₀ + w·x + 0.5 · Σₖ [(Σᵢ Vᵢₖxᵢ)² − Σᵢ (Vᵢₖxᵢ)²])

    forward(indices, values) → logits   (apply sigmoid for probabilities)

    Parameters
    ----------
    indices : (B, nnz) int64  — non-zero feature column indices per row
    values  : (B, nnz) float32 — corresponding non-zero values
    """

    def __init__(self, n_features: int, k: int) -> None:
        super().__init__()
        self.k = k
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w_linear = nn.Parameter(torch.zeros(n_features))
        self.V = nn.Parameter(torch.randn(n_features, k) * 0.01)

    def forward(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # indices: (B, nnz)   int64
        # values:  (B, nnz)   float32

        # First-order: w₀ + Σᵢ wᵢxᵢ
        linear = self.w0 + (self.w_linear[indices] * values).sum(dim=1)  # (B,)

        # Second-order: 0.5 · Σₖ [(Σᵢ Vᵢₖxᵢ)² − Σᵢ (Vᵢₖxᵢ)²]
        emb = self.V[indices]                    # (B, nnz, k)
        Vx = emb * values.unsqueeze(-1)          # (B, nnz, k)
        sum_Vx = Vx.sum(dim=1)                   # (B, k)
        interaction = 0.5 * (
            sum_Vx.pow(2).sum(dim=1) - Vx.pow(2).sum(dim=(1, 2))
        )                                        # (B,)

        return linear + interaction              # logits, shape (B,)


# ── NumPy fast inference for MCTS ─────────────────────────────────────────────

@dataclass
class FMInference:
    """
    NumPy-based FM inference — bypasses PyTorch autograd overhead for the
    single-sample evaluations that MCTS requires.

    Typical speed: 40–80k evaluations/second on CPU.
    Loaded once at MCTS startup: ``inf = FMInference.load()``.

    The key method is ``evaluate_sparse(indices, values)``, where ``indices``
    and ``values`` encode the 9 non-zero features of one complete 3v3 draft
    state (see feature_engineering.py for the feature layout).
    """

    w0: float
    w_linear: np.ndarray     # (D,)  float32
    V: np.ndarray            # (D, k) float32
    schema: FeatureSchema
    val_logloss: float = float("nan")
    val_auc: float = float("nan")
    val_brier: float = float("nan")
    # Training metadata (populated by train_fm; absent when loading pre-trained)
    n_train: int = 0        # augmented training rows (2 × games)
    n_val: int = 0          # augmented val rows (2 × games)
    split_time: str = ""    # ISO-8601 string of first val-set battle_time
    train_history: list = field(default_factory=list)  # [{epoch, train_loss, val_loss, val_auc}]

    @classmethod
    def from_model(
        cls,
        model: FactorizationMachine,
        schema: FeatureSchema,
        val_logloss: float = float("nan"),
        val_auc: float = float("nan"),
        val_brier: float = float("nan"),
        n_train: int = 0,
        n_val: int = 0,
        split_time: str = "",
        train_history: list | None = None,
    ) -> FMInference:
        with torch.no_grad():
            return cls(
                w0=model.w0.item(),
                w_linear=model.w_linear.detach().cpu().numpy().astype(np.float32),
                V=model.V.detach().cpu().numpy().astype(np.float32),
                schema=schema,
                val_logloss=val_logloss,
                val_auc=val_auc,
                val_brier=val_brier,
                n_train=n_train,
                n_val=n_val,
                split_time=split_time,
                train_history=train_history or [],
            )

    def _init_buffers(self) -> None:
        """Initialise pre-allocated inference buffers (called lazily)."""
        k = self.V.shape[1]
        object.__setattr__(self, "_emb_buf", np.empty((9, k), dtype=np.float32))
        object.__setattr__(self, "_sum_buf", np.empty(k,      dtype=np.float32))

    def __setstate__(self, state: dict) -> None:
        """Restore from pickle and re-create inference buffers."""
        for k, v in state.items():
            object.__setattr__(self, k, v)
        self._init_buffers()

    def evaluate_sparse(self, indices: np.ndarray, values: np.ndarray) -> float:
        """
        Evaluate one FM state from sparse (indices, values) arrays.

        O(nnz × k) arithmetic — for nnz=9, k=32: 288 multiply-adds.

        Convention: always pass *my* team as t1 features (t1_ prefix indices)
        and the *opponent* as t2 features. Swapping returns ≈ 1 − result.

        Parameters
        ----------
        indices : int32 array of shape (nnz,) — non-zero feature column indices
        values  : float32 array of shape (nnz,) — corresponding values

        Returns
        -------
        P(my_team wins) ∈ (0, 1)
        """
        # Use pre-allocated buffers to eliminate heap allocations on the hot path.
        # _emb_buf: (9, k) scratch space for V[indices] and Vx in-place.
        # _sum_buf: (k,)   scratch space for sum_Vx.
        try:
            emb_buf = self._emb_buf
            sum_buf = self._sum_buf
        except AttributeError:
            self._init_buffers()
            emb_buf = self._emb_buf
            sum_buf = self._sum_buf

        np.take(self.V, indices, axis=0, out=emb_buf)     # emb_buf = V[indices]
        np.multiply(emb_buf, values[:, None], out=emb_buf) # emb_buf = Vx  (in-place)
        emb_buf.sum(axis=0, out=sum_buf)                   # sum_buf = sum_Vx

        linear = self.w0 + np.dot(self.w_linear[indices], values)
        rv = emb_buf.ravel()   # view — no copy
        interaction = 0.5 * (np.dot(sum_buf, sum_buf) - np.dot(rv, rv))
        return 1.0 / (1.0 + math.exp(-float(linear + interaction)))

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"FM inference model saved → {path}")

    @staticmethod
    def load(path: Path = MODEL_PATH) -> FMInference:
        import importlib

        class _Unpickler(pickle.Unpickler):
            """Remap __main__ → fm_model so spawn workers can unpickle."""

            def find_class(self, module: str, name: str):
                if module == "__main__":
                    try:
                        mod = importlib.import_module("fm_model")
                    except ModuleNotFoundError:
                        mod = importlib.import_module("src.fm_model")
                    return getattr(mod, name)
                return super().find_class(module, name)

        with open(path, "rb") as f:
            return _Unpickler(f).load()

    def benchmark(self, n_calls: int = 10_000) -> float:
        """
        Measure single-sample inference speed.

        Returns evaluations per second and prints a PASS/FAIL vs. the
        10,000 evals/sec target needed for viable MCTS budgets.
        """
        rng = np.random.default_rng(0)
        D = len(self.w_linear)
        # Realistic 9-feature sparse input: 6 brawler bits, 1 map, 1 mode, 1 skill_ns
        indices = rng.choice(D, size=9, replace=False).astype(np.int32)
        values = np.array([1.0] * 8 + [1.2], dtype=np.float32)

        start = time.perf_counter()
        for _ in range(n_calls):
            self.evaluate_sparse(indices, values)
        elapsed = time.perf_counter() - start

        evals_per_sec = n_calls / elapsed
        target = 10_000
        status = "PASS" if evals_per_sec >= target else "FAIL (consider numpy optimisation)"
        print(f"Inference speed: {evals_per_sec:,.0f} evals/sec — {status}")
        return evals_per_sec


# ── Training helpers ──────────────────────────────────────────────────────────

def _csr_to_compact(X: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a CSR matrix with *uniform* non-zeros per row to compact
    (N, nnz) index and value arrays.

    Our feature matrix has exactly 9 non-zeros per row (or 8 without mode),
    so ``X.indices`` and ``X.data`` are both flat arrays of length 9·N that
    reshape cleanly without any data copy.

    ~15× more memory-efficient than densifying (9 stored cols vs 223 total).
    """
    n_rows = X.shape[0]
    nnz_counts = np.diff(X.indptr)
    nnz = int(nnz_counts[0])
    if not np.all(nnz_counts == nnz):
        raise ValueError(
            f"Non-uniform nnz per row (min={nnz_counts.min()}, max={nnz_counts.max()}). "
            "Feature matrix must have exactly nnz non-zeros per row."
        )
    idx = X.indices.reshape(n_rows, nnz).astype(np.int32)
    val = X.data.reshape(n_rows, nnz).astype(np.float32)
    return idx, val


def _eval_metrics(
    model: FactorizationMachine,
    idx: np.ndarray,
    val: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> dict[str, float]:
    """Compute log-loss, AUC-ROC, and Brier score over a full dataset."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_probs, total_loss, n_batches = [], 0.0, 0

    with torch.no_grad():
        for i in range(0, len(y), batch_size):
            ib = torch.from_numpy(idx[i: i + batch_size]).long().to(device)
            vb = torch.from_numpy(val[i: i + batch_size]).to(device)
            yb = torch.from_numpy(y[i: i + batch_size].astype(np.float32)).to(device)
            logits = model(ib, vb)
            total_loss += criterion(logits, yb).item()
            n_batches += 1
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.concatenate(all_probs)
    logloss = total_loss / n_batches
    auc = float(roc_auc_score(y.astype(np.int32), probs))
    brier = float(np.mean((probs - y.astype(np.float32)) ** 2))
    return {"logloss": logloss, "auc": auc, "brier": brier}


# ── Main training function ────────────────────────────────────────────────────

def train_fm(
    k: int = K,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    model_path: Path = MODEL_PATH,
    schema_path: Path = SCHEMA_PATH,
    source: DatasetRef | str | Path | None = None,
    elo_min: int | None = None,
    elo_max: int | None = None,
) -> FMInference:
    """
    Train the FM and save the inference artifact to ``data/fm_model.pkl``.

    Steps:
      1. Load game dataset; apply chronological 80/20 train/val split.
      2. Build sparse feature matrices; convert to compact (N, nnz) format.
      3. Train with BCEWithLogitsLoss + Adam + early stopping on val log-loss.
      4. Evaluate best checkpoint; print log-loss, AUC, Brier.
      5. Save FMInference; run inference speed benchmark.

    Returns
    -------
    FMInference ready for MCTS evaluation (also persisted to disk).
    """
    print("=== Step 1.4: FM Training ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    from bsdraft.data.prep import DB_PATH as _DEFAULT_DB
    from bsdraft.data.prep import ELO_MAX as _EMAX
    from bsdraft.data.prep import ELO_MIN as _EMIN

    print("\nLoading dataset...")
    df, _ = build_game_dataset(
        source if source is not None else _DEFAULT_DB,
        elo_min=elo_min if elo_min is not None else _EMIN,
        elo_max=elo_max if elo_max is not None else _EMAX,
    )
    train_df, val_df = chronological_split(df)

    # Load or build schema. Build from df when Step 1.3 artefacts don't exist yet.
    if schema_path.exists():
        schema = FeatureSchema.load(schema_path)
    else:
        from bsdraft.features.engineering import build_schema
        print("feature_schema.pkl not found — building schema from dataset...")
        schema = build_schema(df)
        schema.save(schema_path)
    del df  # no longer needed

    print(f"Feature schema: D={schema.n_features}, k={k}\n")

    # Capture split boundary for FMInference metadata.
    # val_df is already time-sorted by chronological_split; iloc[0] is the boundary.
    split_time_str = str(val_df["battle_time"].iloc[0])

    print("Building feature matrices (with symmetry augmentation)...")
    X_train_csr, y_train = build_feature_matrix(train_df, schema)
    X_val_csr, y_val = build_feature_matrix(val_df, schema)
    del train_df, val_df
    print(f"  Train: {X_train_csr.shape[0]:,} rows")
    print(f"  Val:   {X_val_csr.shape[0]:,} rows")

    print("Converting to compact sparse arrays...")
    train_idx, train_val = _csr_to_compact(X_train_csr)
    val_idx, val_val = _csr_to_compact(X_val_csr)
    del X_train_csr, X_val_csr  # CSR matrices no longer needed
    print(
        f"  Memory: {(train_idx.nbytes + train_val.nbytes) / 1e6:.0f} MB train, "
        f"{(val_idx.nbytes + val_val.nbytes) / 1e6:.0f} MB val"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FactorizationMachine(n_features=schema.n_features, k=k).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters (k={k})")
    print(f"  w0: 1  |  w_linear: {schema.n_features}  |  V: {schema.n_features}×{k}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    N_train = len(y_train)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state: dict | None = None
    no_improve = 0
    train_history: list[dict] = []
    rng = np.random.default_rng(42)

    print(f"\nTraining (max {max_epochs} epochs, patience={patience}, "
          f"batch={batch_size:,})...\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'AUC':>7}  {'Status'}")
    print("-" * 54)

    for epoch in range(1, max_epochs + 1):
        model.train()
        shuffled = rng.permutation(N_train)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, N_train, batch_size):
            bi = shuffled[start: start + batch_size]
            ib = torch.from_numpy(train_idx[bi]).long().to(device)
            vb = torch.from_numpy(train_val[bi]).to(device)
            yb = torch.from_numpy(y_train[bi].astype(np.float32)).to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ib, vb), yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches
        metrics = _eval_metrics(model, val_idx, val_val, y_val, device)
        val_loss = metrics["logloss"]

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {key: v.clone() for key, v in model.state_dict().items()}
            no_improve = 0
            status = "✓ best"
        else:
            no_improve += 1
            status = f"({no_improve}/{patience})"

        train_history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "val_auc": metrics["auc"],
        })

        print(f"{epoch:>5}  {epoch_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{metrics['auc']:>7.4f}  {status}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    # ── Restore best checkpoint and final evaluation ───────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    final = _eval_metrics(model, val_idx, val_val, y_val, device)
    print("\n── Final validation metrics ─────────────────────────────")
    print(f"  Log-loss : {final['logloss']:.4f}  (random baseline ≈ 0.693)")
    print(f"  AUC-ROC  : {final['auc']:.4f}  (random baseline = 0.500)")
    print(f"  Brier    : {final['brier']:.4f}  (random baseline = 0.250)")

    # ── Save ──────────────────────────────────────────────────────────────────
    inference = FMInference.from_model(
        model,
        schema,
        val_logloss=final["logloss"],
        val_auc=final["auc"],
        val_brier=final["brier"],
        n_train=N_train,
        n_val=len(y_val),
        split_time=split_time_str,
        train_history=train_history,
    )
    inference.save(model_path)

    print()
    inference.benchmark()

    print("\n=== Step 1.4 complete ===")
    return inference


if __name__ == "__main__":
    import argparse

    from bsdraft.data.prep import SEASON_CONFIGS

    parser = argparse.ArgumentParser(description="Train FM for a given season.")
    parser.add_argument(
        "--season",
        choices=list(SEASON_CONFIGS),
        default="s42",
        help="Which season config to use (default: s42)",
    )
    args = parser.parse_args()
    cfg = SEASON_CONFIGS[args.season]
    data_dir = cfg["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Season: {args.season}  |  DB: {cfg['db_path']}  |  ELO [{cfg['elo_min']}, {cfg['elo_max']}]")
    print(f"Output dir: {data_dir}\n")
    train_fm(
        model_path=data_dir / "fm_model.pkl",
        schema_path=data_dir / "feature_schema.pkl",
        db_path=cfg["db_path"],
        elo_min=cfg["elo_min"],
        elo_max=cfg["elo_max"],
    )
