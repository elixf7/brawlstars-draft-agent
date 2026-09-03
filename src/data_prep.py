"""
data_prep.py — Step 1.1: Data Extraction & Preprocessing

Loads the season42 SQLite database, applies quality filters, expands set-level
records into individual game rows, and builds the brawler vocabulary.

Key decisions baked in:
  - Quality filter: skill_ns_ok = 1, avg_elo IN [13, 23]
  - Incomplete teams (any NULL brawler): dropped
  - Record expansion: one row per game within the set; draws excluded
  - skill_ns: kept as continuous (not binned) — see note at bottom of file
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "season42" / "season42_combined_skill_ns.db"

ELO_MIN = 13
ELO_MAX = 23

# Per-season configuration: db path, ELO filter bounds, and output data directory.
# Add new entries here as new seasons are collected.
SEASON_CONFIGS: dict[str, dict] = {
    "s42": {
        "db_path":  REPO_ROOT / "season42" / "season42_combined_skill_ns.db",
        "elo_min":  13,
        "elo_max":  23,
        "data_dir": REPO_ROOT / "data" / "s42",
    },
    "s48": {
        "db_path":  REPO_ROOT / "season48" / "v1_clean.db",
        "elo_min":  10,   # lower bound relaxed — v1_clean is pre-filtered, ELO scale shifted
        "elo_max":  23,
        "data_dir": REPO_ROOT / "data" / "s48",
    },
    "s49": {
        "db_path":  REPO_ROOT / "season49" / "v1_clean.db",
        "elo_min":  10,
        "elo_max":  23,
        "data_dir": REPO_ROOT / "data" / "s49",
    },
}

TEAM1_BRAWLER_COLS = ["t1_b0_name", "t1_b1_name", "t1_b2_name"]
TEAM2_BRAWLER_COLS = ["t2_b0_name", "t2_b1_name", "t2_b2_name"]
ALL_BRAWLER_COLS = TEAM1_BRAWLER_COLS + TEAM2_BRAWLER_COLS

# Columns we actually need for modelling — drop the rest to save memory
KEEP_COLS = [
    "id",
    "battle_time",
    "mode",
    "map",
    "record",
    "avg_elo",
    "skill_ns",
    *TEAM1_BRAWLER_COLS,
    *TEAM2_BRAWLER_COLS,
]


# ---------------------------------------------------------------------------
# 1. Load & filter
# ---------------------------------------------------------------------------


def load_filtered_matches(
    db_path: Path = DB_PATH,
    elo_min: int = ELO_MIN,
    elo_max: int = ELO_MAX,
) -> pd.DataFrame:
    """
    Pull matches from SQLite with quality filters applied at the SQL level
    (faster than filtering in pandas for large tables).

    Filters:
      - skill_ns_ok = 1  (ECDF score computed from a well-populated time bin)
      - avg_elo BETWEEN elo_min AND elo_max
    """
    query = f"""
        SELECT {", ".join(KEEP_COLS)}
        FROM matches
        WHERE skill_ns_ok = 1
          AND avg_elo BETWEEN {elo_min} AND {elo_max}
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    print(f"Loaded {len(df):,} rows after quality filters.")
    return df


def drop_incomplete_teams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where any brawler slot is NULL.

    In practice, the quality-filtered data has zero NULLs in brawler columns,
    but this guard is here for correctness and future-proofing.
    """
    before = len(df)
    df = df.dropna(subset=ALL_BRAWLER_COLS).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped:,} rows with incomplete teams.")
    else:
        print("No incomplete team rows found (as expected).")
    return df


# ---------------------------------------------------------------------------
# 2. Expand sets → individual games
# ---------------------------------------------------------------------------


def expand_to_games(df: pd.DataFrame, draw_value: float = None) -> pd.DataFrame:
    """
    Expand each set row into one row per individual game.

    Each game in the `record` column (e.g. "T1-T2-T1") becomes a separate row
    with a `team1_wins` label. The team compositions are the same for all games
    in a set — the API records only the final composition per set.

    Parameters
    ----------
    draw_value : float or None
        How to handle 'D' (draw) games.
        - None  → exclude draw games entirely (recommended; draws are ~1.3% of games)
        - 0.5   → keep draws with a soft label of 0.5 (use with BCELoss, not hard labels)

    Returns
    -------
    DataFrame with one row per non-excluded game, plus column `team1_wins`.
    """
    # Explode the record string into one game token per row
    df = df.copy()
    df["game_result"] = df["record"].str.split("-")
    df = df.explode("game_result").reset_index(drop=True)

    if draw_value is None:
        before = len(df)
        df = df[df["game_result"] != "D"].reset_index(drop=True)
        dropped = before - len(df)
        print(f"Excluded {dropped:,} draw games ({dropped / before * 100:.2f}% of all games).")
        df["team1_wins"] = (df["game_result"] == "T1").astype(np.int8)
    else:
        df["team1_wins"] = df["game_result"].map({"T1": 1.0, "T2": 0.0, "D": draw_value})
        draws = (df["game_result"] == "D").sum()
        print(f"Kept {draws:,} draw games with soft label {draw_value}.")

    df = df.drop(columns=["record", "game_result"])
    print(f"Expanded to {len(df):,} individual game rows.")
    return df


# ---------------------------------------------------------------------------
# 3. Brawler vocabulary
# ---------------------------------------------------------------------------


def build_brawler_vocab(df: pd.DataFrame, min_picks: int = 0) -> list[str]:
    """
    Build a sorted vocabulary of all brawlers appearing in the dataset.

    Parameters
    ----------
    min_picks : int
        Minimum total appearances (across all 6 slots and all rows) to be
        included. Use 0 to include all brawlers. LOLA has ~8,640 picks in
        season42, which is already well above any reasonable threshold.

    Returns
    -------
    Sorted list of brawler name strings.
    """
    all_picks = pd.concat(
        [df[col] for col in ALL_BRAWLER_COLS],
        ignore_index=True,
    )
    counts = all_picks.value_counts()

    if min_picks > 0:
        excluded = counts[counts < min_picks].index.tolist()
        if excluded:
            print(f"Excluding {len(excluded)} brawlers with < {min_picks} picks: {excluded}")
        counts = counts[counts >= min_picks]

    vocab = sorted(counts.index.tolist())
    print(f"Brawler vocabulary: {len(vocab)} brawlers.")
    return vocab


# ---------------------------------------------------------------------------
# 4. Summary / validation helpers
# ---------------------------------------------------------------------------


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the preprocessed dataset."""
    print("\n=== Dataset Summary ===")
    print(f"  Rows:           {len(df):,}")
    print(f"  Modes:          {sorted(df['mode'].unique())}")
    print(f"  Maps:           {df['map'].nunique()} unique")
    print(f"  avg_elo range:  [{df['avg_elo'].min():.1f}, {df['avg_elo'].max():.1f}]")
    print(f"  skill_ns range: [{df['skill_ns'].min():.3f}, {df['skill_ns'].max():.3f}]")
    print(f"  skill_ns mean:  {df['skill_ns'].mean():.3f}")
    skill_qs = df['skill_ns'].quantile([0.25, 0.5, 0.75])
    print(f"  skill_ns Q25/Q50/Q75: {skill_qs[0.25]:.3f} / {skill_qs[0.5]:.3f} / {skill_qs[0.75]:.3f}")
    if "team1_wins" in df.columns:
        win_rate = df["team1_wins"].mean()
        print(f"  Team1 win rate: {win_rate:.4f} (should be ~0.50 before symmetry augmentation)")
    print("=======================\n")


def check_label_balance(df: pd.DataFrame) -> None:
    """
    Warn if the raw team1 win rate deviates significantly from 50%.

    The 'team1' label is arbitrary (who was recorded as team 1 in the API),
    so we expect near 50/50. A systematic imbalance suggests a data issue.
    """
    if "team1_wins" not in df.columns:
        print("No team1_wins column to check.")
        return
    win_rate = df["team1_wins"].mean()
    if abs(win_rate - 0.5) > 0.02:
        print(
            f"WARNING: team1 win rate is {win_rate:.4f}, which deviates more than 2% from 0.5. "
            "Investigate possible labelling bias before training."
        )
    else:
        print(f"Label balance OK: team1 win rate = {win_rate:.4f}")


# ---------------------------------------------------------------------------
# 5. Full pipeline
# ---------------------------------------------------------------------------


def build_game_dataset(
    db_path: Path = DB_PATH,
    elo_min: int = ELO_MIN,
    elo_max: int = ELO_MAX,
    draw_value: float = None,
    min_brawler_picks: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Run the full 1.1 preprocessing pipeline.

    Returns
    -------
    df : pd.DataFrame
        One row per individual game with quality filters applied.
        Contains team brawler columns, mode, map, skill_ns, team1_wins.
    vocab : list[str]
        Sorted brawler vocabulary.
    """
    df = load_filtered_matches(db_path, elo_min=elo_min, elo_max=elo_max)
    df = drop_incomplete_teams(df)
    df = expand_to_games(df, draw_value=draw_value)
    vocab = build_brawler_vocab(df, min_picks=min_brawler_picks)
    print_dataset_summary(df)
    check_label_balance(df)
    return df, vocab


# ---------------------------------------------------------------------------
# NOTE on skill_ns — why we keep it continuous
# ---------------------------------------------------------------------------
# skill_ns is a logit-transformed ECDF percentile computed within 3-day time
# bins (to account for Elo drift across the season). It is already normalized
# to a comparable scale across all time periods.
#
# The 10 time bins used to compute skill_ns are temporal normalization windows
# and have NO relationship to how many skill feature categories the FM should use.
#
# We keep skill_ns as a single continuous feature for the FM. The FM interaction
# term becomes ⟨v_skill, v_brawler⟩ × skill_ns_value, scaling the brawler-skill
# interaction by the actual normalized score. This avoids arbitrary bin boundary
# decisions and preserves all within-bin variation.
#
# If the chosen FM library requires binary/sparse features (e.g. libFFM format),
# revisit at Step 1.3 and use quantile bins at that point.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data_prep pipeline for a given season.")
    parser.add_argument(
        "--season",
        choices=list(SEASON_CONFIGS),
        default="s42",
        help="Which season config to use (default: s42)",
    )
    args = parser.parse_args()
    cfg = SEASON_CONFIGS[args.season]

    print(f"Season: {args.season}  |  DB: {cfg['db_path']}  |  ELO [{cfg['elo_min']}, {cfg['elo_max']}]")
    df, vocab = build_game_dataset(
        db_path=cfg["db_path"],
        elo_min=cfg["elo_min"],
        elo_max=cfg["elo_max"],
    )
    print(f"Sample rows:\n{df[['mode', 'map', 'skill_ns', 'team1_wins', *TEAM1_BRAWLER_COLS]].head()}")
    print(f"\nFirst 10 brawlers in vocab: {vocab[:10]}")
