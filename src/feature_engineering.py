"""
feature_engineering.py — Step 1.3: Feature Engineering for the FM

Feature schema (all features are per-match, describing a complete 3v3 draft):

  [0,       V)        t1 brawler indicators  (binary, V = vocab size)
  [V,      2V)        t2 brawler indicators  (binary)
  [2V,  2V+M)        map indicators          (one-hot, M = number of maps)
  [2V+M, 2V+M+Mo)    mode indicators         (one-hot, Mo = number of modes)
  [2V+M+Mo]          skill_ns                (continuous float)

  With V=95, M=26, Mo=6 → D = 190 + 26 + 6 + 1 = 223 features

NOTE on including mode alongside map:
  Each map belongs to exactly one mode, so mode is technically recoverable from
  map. We include it anyway because the FM learns pairwise feature interactions:
  ⟨v_t1_CROW, v_bounty_map⟩ is learned only from games on that specific map,
  while ⟨v_t1_CROW, v_bounty_mode⟩ is learned across ALL bounty maps. The mode
  indicator gives the FM a coarser interaction level that pools signal from the
  whole mode — a cheap form of regularization at the cost of only 6 extra
  features. If you prefer to drop mode, set include_mode=False in FeatureSchema.

NOTE on symmetric augmentation:
  The FM is trained to predict P(team1 wins). "Team 1" vs "Team 2" is assigned
  arbitrarily in the raw data. Without augmentation, any team-assignment bias
  in the data source would leak into the model. Symmetry augmentation fixes
  this: for every row we also emit the "flipped" row with teams swapped and
  the label inverted. This doubles training data and guarantees P(A beats B) +
  P(B beats A) = 1 for every composition pair.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz

# Ensure src/ is on the path so intra-package imports work whether this module
# is imported as `src.feature_engineering` (from repo root) or run directly.
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_prep import (  # noqa: E402
    build_game_dataset,
    TEAM1_BRAWLER_COLS,
    TEAM2_BRAWLER_COLS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

SCHEMA_PATH = DATA_DIR / "feature_schema.pkl"
MATRIX_PATH = DATA_DIR / "fm_features.npz"
LABELS_PATH = DATA_DIR / "fm_labels.npy"


# ---------------------------------------------------------------------------
# FeatureSchema
# ---------------------------------------------------------------------------


@dataclass
class FeatureSchema:
    """
    Holds the FM feature space definition and all column-index mappings.

    Fields
    ------
    vocab : sorted list of brawler names (length V)
    maps  : sorted list of map names     (length M)
    modes : sorted list of mode names    (length Mo)
    include_mode : whether to include the mode one-hot block
    """

    vocab: list[str]
    maps: list[str]
    modes: list[str]
    include_mode: bool = True

    # Derived layout attributes (populated by __post_init__)
    t1_offset: int = field(init=False)
    t2_offset: int = field(init=False)
    map_offset: int = field(init=False)
    mode_offset: int = field(init=False)
    skill_offset: int = field(init=False)
    n_features: int = field(init=False)

    _vocab_idx: dict[str, int] = field(init=False, repr=False)
    _map_idx: dict[str, int] = field(init=False, repr=False)
    _mode_idx: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        V = len(self.vocab)
        M = len(self.maps)
        Mo = len(self.modes) if self.include_mode else 0

        self.t1_offset = 0
        self.t2_offset = V
        self.map_offset = 2 * V
        self.mode_offset = 2 * V + M
        self.skill_offset = 2 * V + M + Mo
        self.n_features = 2 * V + M + Mo + 1

        self._vocab_idx = {b: i for i, b in enumerate(self.vocab)}
        self._map_idx = {m: i for i, m in enumerate(self.maps)}
        self._mode_idx = {m: i for i, m in enumerate(self.modes)}

    def feature_name(self, idx: int) -> str:
        """Return a human-readable label for a feature column index."""
        if idx < self.t2_offset:
            return f"t1_{self.vocab[idx]}"
        if idx < self.map_offset:
            return f"t2_{self.vocab[idx - self.t2_offset]}"
        if idx < self.mode_offset:
            return f"map_{self.maps[idx - self.map_offset]}"
        if idx < self.skill_offset:
            return f"mode_{self.modes[idx - self.mode_offset]}"
        return "skill_ns"

    def save(self, path: Path = SCHEMA_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Schema saved → {path}")

    @staticmethod
    def load(path: Path = SCHEMA_PATH) -> "FeatureSchema":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    schema: FeatureSchema,
) -> tuple[csr_matrix, np.ndarray]:
    """
    Encode a game-level DataFrame into a sparse feature matrix.

    For each of the N input rows, two rows are emitted:
      - Original (row 2i):   t1 brawlers as team1, label = team1_wins
      - Flipped  (row 2i+1): t2 brawlers as team1, label = 1 - team1_wins

    Parameters
    ----------
    df : game-level DataFrame from build_game_dataset().
         Must contain TEAM1_BRAWLER_COLS, TEAM2_BRAWLER_COLS,
         'map', 'mode', 'skill_ns', 'team1_wins'.
    schema : FeatureSchema built from the same dataset.

    Returns
    -------
    X : csr_matrix of shape (2N, D), dtype float32
    y : int8 ndarray of shape (2N,), values in {0, 1}
    """
    N = len(df)
    V = len(schema.vocab)

    # ------------------------------------------------------------------
    # 1. Resolve column indices for each feature group (vectorised).
    # ------------------------------------------------------------------

    # Brawler indices — shape (N, 3) each
    t1_cols = np.stack(
        [df[col].map(schema._vocab_idx).values for col in TEAM1_BRAWLER_COLS], axis=1
    ).astype(np.int32)  # values in [0, V)
    t2_cols = np.stack(
        [df[col].map(schema._vocab_idx).values for col in TEAM2_BRAWLER_COLS], axis=1
    ).astype(np.int32)  # values in [0, V)

    # Shift into the correct positions in the feature vector
    t1_global = t1_cols + schema.t1_offset  # stays in [0, V)
    t2_global = t2_cols + schema.t2_offset  # shifts into [V, 2V)

    # Map and mode — shape (N,)
    map_global = (df["map"].map(schema._map_idx).values.astype(np.int32)
                  + schema.map_offset)

    if schema.include_mode:
        mode_global = (df["mode"].map(schema._mode_idx).values.astype(np.int32)
                       + schema.mode_offset)

    # skill_ns — shape (N,) float32
    skill_vals = df["skill_ns"].values.astype(np.float32)

    # Labels — shape (N,)
    labels = df["team1_wins"].values.astype(np.int8)

    # ------------------------------------------------------------------
    # 2. Build COO arrays for both the original and flipped rows.
    #
    #    Each row contributes at most 9 non-zero entries:
    #      3 (t1 brawlers) + 3 (t2 brawlers) + 1 (map) + 1 (mode) + 1 (skill_ns)
    #    = 9 if include_mode else 8
    # ------------------------------------------------------------------
    nz_per_row = 9 if schema.include_mode else 8

    # Row index arrays
    orig_rows = np.arange(N, dtype=np.int32) * 2        # [0, 2, 4, ...]
    flip_rows = orig_rows + 1                           # [1, 3, 5, ...]

    # Repeated row indices for multi-feature groups
    orig_rows_x3 = np.repeat(orig_rows, 3)  # shape (3N,)
    flip_rows_x3 = np.repeat(flip_rows, 3)

    # --- Original rows ---
    all_rows_orig = [orig_rows_x3, orig_rows_x3, orig_rows, orig_rows]
    all_cols_orig = [t1_global.ravel(), t2_global.ravel(), map_global,
                     np.full(N, schema.skill_offset, dtype=np.int32)]
    all_vals_orig = [np.ones(3 * N, dtype=np.float32),
                     np.ones(3 * N, dtype=np.float32),
                     np.ones(N, dtype=np.float32),
                     skill_vals]

    # --- Flipped rows: t1 ↔ t2 roles are swapped ---
    #   t1_global values are in [t1_offset, t2_offset) = [0, V)
    #   In the flipped row these brawlers become team2 → add V to shift into [V, 2V)
    #   t2_global values are in [t2_offset, map_offset) = [V, 2V)
    #   In the flipped row these brawlers become team1 → subtract V to shift into [0, V)
    flip_t1_global = t2_global - V   # was t2, becomes t1
    flip_t2_global = t1_global + V   # was t1, becomes t2

    all_rows_flip = [flip_rows_x3, flip_rows_x3, flip_rows, flip_rows]
    all_cols_flip = [flip_t1_global.ravel(), flip_t2_global.ravel(), map_global,
                     np.full(N, schema.skill_offset, dtype=np.int32)]
    all_vals_flip = [np.ones(3 * N, dtype=np.float32),
                     np.ones(3 * N, dtype=np.float32),
                     np.ones(N, dtype=np.float32),
                     skill_vals]

    if schema.include_mode:
        all_rows_orig.insert(3, orig_rows)
        all_cols_orig.insert(3, mode_global)
        all_vals_orig.insert(3, np.ones(N, dtype=np.float32))

        all_rows_flip.insert(3, flip_rows)
        all_cols_flip.insert(3, mode_global)
        all_vals_flip.insert(3, np.ones(N, dtype=np.float32))

    # ------------------------------------------------------------------
    # 3. Concatenate and build CSR matrix.
    # ------------------------------------------------------------------
    row_arr = np.concatenate(all_rows_orig + all_rows_flip)
    col_arr = np.concatenate(all_cols_orig + all_cols_flip)
    val_arr = np.concatenate(all_vals_orig + all_vals_flip)

    X = csr_matrix(
        (val_arr, (row_arr, col_arr)),
        shape=(2 * N, schema.n_features),
    )

    # Labels: original rows keep team1_wins, flipped rows invert it
    y = np.empty(2 * N, dtype=np.int8)
    y[0::2] = labels
    y[1::2] = 1 - labels

    return X, y


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------


def build_schema(df: pd.DataFrame, include_mode: bool = True) -> FeatureSchema:
    """Build a FeatureSchema from the vocabulary present in df."""
    from data_prep import build_brawler_vocab

    vocab = build_brawler_vocab(df)
    maps = sorted(df["map"].unique().tolist())
    modes = sorted(df["mode"].unique().tolist())
    return FeatureSchema(vocab=vocab, maps=maps, modes=modes, include_mode=include_mode)


# ---------------------------------------------------------------------------
# Chronological train/val split
# ---------------------------------------------------------------------------


def chronological_split(
    df: pd.DataFrame,
    val_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train and val sets by time order.

    Uses the earliest (1 - val_fraction) of rows as train and the last
    val_fraction as val, determined by 'battle_time' (ISO-8601 strings sort
    lexicographically). A chronological split prevents future-data leakage —
    the model is evaluated on matches that happened after all training matches.

    Returns
    -------
    train_df, val_df
    """
    df_sorted = df.sort_values("battle_time").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1.0 - val_fraction))
    train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
    val_df = df_sorted.iloc[split_idx:].reset_index(drop=True)

    split_time = df_sorted["battle_time"].iloc[split_idx]
    print(
        f"Chronological split at {split_time}: "
        f"{len(train_df):,} train / {len(val_df):,} val rows"
    )
    return train_df, val_df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def build_and_save(
    matrix_path: Path = MATRIX_PATH,
    labels_path: Path = LABELS_PATH,
    schema_path: Path = SCHEMA_PATH,
    db_path: Path | None = None,
    elo_min: int | None = None,
    elo_max: int | None = None,
) -> tuple[csr_matrix, np.ndarray, FeatureSchema]:
    """
    Full 1.3 pipeline: load game dataset → build schema → encode → save.

    Saves three artefacts to the target directory:
      fm_features.npz    — sparse feature matrix (2N × D)
      fm_labels.npy      — label vector (2N,)
      feature_schema.pkl — FeatureSchema for column-name lookups

    Returns (X, y, schema).
    """
    from data_prep import DB_PATH as _DEFAULT_DB, ELO_MIN as _EMIN, ELO_MAX as _EMAX  # noqa: E402

    print("=== Step 1.3: Feature Engineering ===\n")

    df, _ = build_game_dataset(
        db_path=db_path if db_path is not None else _DEFAULT_DB,
        elo_min=elo_min if elo_min is not None else _EMIN,
        elo_max=elo_max if elo_max is not None else _EMAX,
    )
    N = len(df)

    print("\nBuilding feature schema...")
    schema = build_schema(df)
    print(
        f"  D = {schema.n_features} features: "
        f"{len(schema.vocab)} t1 + {len(schema.vocab)} t2 brawler dims, "
        f"{len(schema.maps)} map dims, {len(schema.modes)} mode dims, "
        f"1 skill_ns"
    )

    print(f"\nEncoding {N:,} games → {2 * N:,} rows (with symmetric augmentation)...")
    X, y = build_feature_matrix(df, schema)

    density = X.nnz / (X.shape[0] * X.shape[1])
    print(f"  Shape: {X.shape}")
    print(f"  nnz:   {X.nnz:,}")
    print(f"  Density: {density:.5%}  (expected ~{9 / schema.n_features:.5%})")
    print(f"  Label balance: {y.mean():.4f}  (expected exactly 0.50 after augmentation)")

    assert abs(y.mean() - 0.5) < 1e-6, (
        f"Label balance broken: {y.mean():.6f} — symmetry augmentation has a bug."
    )

    print("\nSaving artefacts...")
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(str(matrix_path), X)
    print(f"  {matrix_path}")
    np.save(str(labels_path), y)
    print(f"  {labels_path}")
    schema.save(schema_path)

    print("\n=== Step 1.3 complete ===")
    return X, y, schema


# ---------------------------------------------------------------------------
# Convenience loader (used by FM training in Step 1.4)
# ---------------------------------------------------------------------------


def load_feature_matrix(
    matrix_path: Path = MATRIX_PATH,
    labels_path: Path = LABELS_PATH,
    schema_path: Path = SCHEMA_PATH,
) -> tuple[csr_matrix, np.ndarray, FeatureSchema]:
    """Load the pre-built feature matrix, labels, and schema from disk."""
    X = load_npz(str(matrix_path))
    y = np.load(str(labels_path))
    schema = FeatureSchema.load(schema_path)
    return X, y, schema


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from data_prep import SEASON_CONFIGS  # noqa: E402

    parser = argparse.ArgumentParser(description="Build FM feature matrix for a given season.")
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
    build_and_save(
        matrix_path=data_dir / "fm_features.npz",
        labels_path=data_dir / "fm_labels.npy",
        schema_path=data_dir / "feature_schema.pkl",
        db_path=cfg["db_path"],
        elo_min=cfg["elo_min"],
        elo_max=cfg["elo_max"],
    )
