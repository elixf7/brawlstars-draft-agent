"""
matchup_db.py — Steps 1.2.1–1.2.4: Empirical Matchup Database

Builds and serializes three O(1) lookup tables for MCTS rollout and scoring:

  1. Brawler stats     (1.2.1): pick_rate + win_rate per brawler/context
  2. Counter matrix    (1.2.2): P(win | my_brawler vs opp_brawler) per context
  3. Synergy matrix    (1.2.3): win_rate delta above per-brawler baselines

All tables use a fallback chain from most- to least-specific context so
lookups never return None for any brawler in the vocabulary.

Serialized as a single MatchupDB pickle for fast loading at MCTS startup.
All dict keys are plain Python types (str / int) for reliable hashing.

Usage:
  db = MatchupDB.build()          # one-time build (~minutes)
  db.save()                       # writes data/matchup_db.pkl
  db = MatchupDB.load()           # fast load at runtime
  db.brawler_lookup("CROW", "gemGrab", "Double Swoosh", tier=2)
  db.counter_lookup("CROW", "POCO", "gemGrab", "Double Swoosh", tier=2)
  db.synergy_lookup("CROW", "POCO", "gemGrab", tier=2)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from bsdraft.data.sources import DatasetRef
from bsdraft.data.prep import DB_PATH, TEAM1_BRAWLER_COLS, TEAM2_BRAWLER_COLS, build_game_dataset

# src/bsdraft/<subpackage>/<module>.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAVE_PATH = REPO_ROOT / "data" / "matchup_db.pkl"

N_SKILL_TIERS = 4  # quartile tiers: 0 (casual) … 3 (elite)


# ---------------------------------------------------------------------------
# Skill tier assignment (shared by all three tables)
# ---------------------------------------------------------------------------

def assign_skill_tiers(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, float, float]]:
    """
    Add a `skill_tier` int8 column (0–3) via quartile bins of `skill_ns`.

    Returns
    -------
    (df_with_tier, boundaries) where boundaries = (Q25, Q50, Q75) of skill_ns.
    Boundaries are stored in MatchupDB so MCTS can map skill_ns → tier at
    runtime via skill_ns_to_tier() without reloading the full dataset.
    """
    df = df.copy()
    _, bin_edges = pd.qcut(df["skill_ns"], q=N_SKILL_TIERS, labels=False, retbins=True)
    df["skill_tier"] = pd.cut(
        df["skill_ns"], bins=bin_edges, labels=False, include_lowest=True
    ).astype(np.int8)
    boundaries = (float(bin_edges[1]), float(bin_edges[2]), float(bin_edges[3]))
    return df, boundaries


def skill_ns_to_tier(skill_ns: float, boundaries: tuple[float, float, float]) -> int:
    """
    Map a continuous skill_ns value to a skill tier (0–3) using stored boundaries.

    Parameters
    ----------
    skill_ns   : logit-ECDF percentile from the current session
    boundaries : (Q25, Q50, Q75) from MatchupDB.skill_tier_boundaries

    Returns
    -------
    int in {0, 1, 2, 3}  (0 = casual, 3 = elite)
    """
    q25, q50, q75 = boundaries
    if skill_ns < q25:
        return 0
    if skill_ns < q50:
        return 1
    if skill_ns < q75:
        return 2
    return 3


# ---------------------------------------------------------------------------
# 1.2.1 — Brawler stats: pick_rate + win_rate
# ---------------------------------------------------------------------------

def _melt_brawler_games(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape from one row per game → one row per (game, brawler slot)."""
    frames = []
    for col in TEAM1_BRAWLER_COLS:
        sub = df[["mode", "map", "skill_tier", col, "team1_wins"]].copy()
        sub = sub.rename(columns={col: "brawler", "team1_wins": "won"})
        frames.append(sub)
    for col in TEAM2_BRAWLER_COLS:
        sub = df[["mode", "map", "skill_tier", col, "team1_wins"]].copy()
        sub["won"] = 1 - sub["team1_wins"]
        sub = sub.drop(columns=["team1_wins"]).rename(columns={col: "brawler"})
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def build_brawler_stats(df: pd.DataFrame) -> dict[str, dict]:
    """
    Compute per-brawler pick_rate and win_rate at four granularity levels.

    Fallback chain (most → least specific):
      'full'   : (brawler, map, skill_tier)   — map uniquely determines mode
      'no_map' : (brawler, mode, skill_tier)  — pool maps within mode
      'no_tier': (brawler, mode)              — pool tiers
      'global' : (brawler,)                   — pool everything

    pick_rate = brawler appearances / total brawler slots in that context cell.
    """
    long = _melt_brawler_games(df)
    total_slots = len(long)

    def _agg(group_cols: list[str], context_cols: list[str]) -> dict:
        agg = long.groupby(group_cols)["won"].agg(n="count", wins="sum").reset_index()
        agg["win_rate"] = agg["wins"] / agg["n"]
        if context_cols:
            ctx = long.groupby(context_cols).size().reset_index(name="ctx_total")
            agg = agg.merge(ctx, on=context_cols, how="left")
            agg["pick_rate"] = agg["n"] / agg["ctx_total"]
        else:
            agg["pick_rate"] = agg["n"] / total_slots
        result = {}
        for row in agg.itertuples(index=False):
            key = tuple(
                int(getattr(row, c)) if c == "skill_tier" else getattr(row, c)
                for c in group_cols
            )
            result[key] = {
                "n": int(row.n),
                "wins": int(row.wins),
                "win_rate": float(row.win_rate),
                "pick_rate": float(row.pick_rate),
            }
        return result

    stats = {
        "full":    _agg(["brawler", "map", "skill_tier"],  ["map", "skill_tier"]),
        "no_map":  _agg(["brawler", "mode", "skill_tier"], ["mode", "skill_tier"]),
        "no_tier": _agg(["brawler", "mode"],               ["mode"]),
        "global":  _agg(["brawler"],                       []),
    }
    n_brawlers = long["brawler"].nunique()
    print(
        f"Brawler stats: {n_brawlers} brawlers | "
        f"{len(stats['full']):,} / {len(stats['no_map']):,} / "
        f"{len(stats['no_tier']):,} / {len(stats['global']):,} cells "
        f"(full/no_map/no_tier/global)"
    )
    return stats


# ---------------------------------------------------------------------------
# 1.2.2 — Counter matrix: P(win | my_brawler vs opp_brawler)
# ---------------------------------------------------------------------------

def build_counter_matrix(df: pd.DataFrame) -> dict[str, dict]:
    """
    Pairwise counter matrix with four granularity levels.

    For ordered pair (x, y): x on my team, y on opponent team.
      win_rate = P(my team wins | x and y both in match)

    Symmetry guarantee: P(x beats y) + P(y beats x) = 1.
    Enforced by including both directions (x→y, won) and (y→x, 1-won)
    from each of the 9 cross-team slot combinations per game.

    Primary: (x, y, map, skill_tier). Maps are specific enough to capture
    geometry effects (open vs enclosed). Fallbacks: (mode, tier), mode, global.

    Memory strategy: process one slot pair at a time; accumulate aggregates.
    Peak memory = one ~9M-row sub-DataFrame, not the full 79M-row product.
    """
    # Running accumulators: key → [n, wins]
    levels: dict[str, dict[tuple, list[int]]] = {
        "full":    {},   # (x, y, map, skill_tier)
        "no_map":  {},   # (x, y, mode, skill_tier)
        "no_tier": {},   # (x, y, mode)
        "global":  {},   # (x, y)
    }

    def _add(d: dict, key: tuple, n: int, wins: int) -> None:
        if key in d:
            d[key][0] += n
            d[key][1] += wins
        else:
            d[key] = [n, wins]

    for t1_col in TEAM1_BRAWLER_COLS:
        for t2_col in TEAM2_BRAWLER_COLS:
            sub = df[[t1_col, t2_col, "mode", "map", "skill_tier", "team1_wins"]].copy()
            sub.columns = ["x", "y", "mode", "map", "skill_tier", "won"]

            # Both directions from the same games — enforces P(x→y) + P(y→x) = 1
            rev = sub.assign(x=sub["y"], y=sub["x"], won=1 - sub["won"])
            both = pd.concat([sub, rev], ignore_index=True)

            for lvl, grp in [
                ("full",    ["x", "y", "map", "skill_tier"]),
                ("no_map",  ["x", "y", "mode", "skill_tier"]),
                ("no_tier", ["x", "y", "mode"]),
                ("global",  ["x", "y"]),
            ]:
                agg = both.groupby(grp)["won"].agg(n="count", wins="sum").reset_index()
                for row in agg.itertuples(index=False):
                    key = tuple(
                        int(getattr(row, c)) if c == "skill_tier" else getattr(row, c)
                        for c in grp
                    )
                    _add(levels[lvl], key, int(row.n), int(row.wins))

    result = {
        lvl: {k: {"n": v[0], "wins": v[1], "win_rate": v[1] / v[0]}
              for k, v in data.items()}
        for lvl, data in levels.items()
    }
    print(
        f"Counter matrix: {len(result['global']):,} ordered pairs | "
        f"{len(result['full']):,} / {len(result['no_map']):,} / "
        f"{len(result['no_tier']):,} / {len(result['global']):,} cells "
        f"(full/no_map/no_tier/global)"
    )
    return result


# ---------------------------------------------------------------------------
# 1.2.3 — Synergy matrix: win_rate delta above individual baselines
# ---------------------------------------------------------------------------

def build_synergy_matrix(
    df: pd.DataFrame,
    brawler_stats: dict[str, dict],
) -> dict[str, dict]:
    """
    Pairwise synergy matrix for same-team brawler pairs.

    synergy_delta = win_rate(a+b together) - (win_rate(a alone) + win_rate(b alone)) / 2

    Pairs stored in canonical alphabetical order (a ≤ b) — synergy is symmetric.
    Primary: (a, b, mode, skill_tier). Fallbacks: (a, b, mode), (a, b).
    Granularity is mode (not map) — per-map synergy cells are too sparse.

    Individual baselines use brawler_stats['no_map'] (mode+tier level),
    falling back to global win rate if unavailable.
    """
    no_map_stats = brawler_stats["no_map"]   # (brawler, mode, tier) → stats
    global_stats = brawler_stats["global"]   # (brawler,) → stats

    def _baseline_wr(brawler: str, mode: str, tier: int) -> float:
        """Individual win rate for one brawler at (mode, tier), with global fallback."""
        v = no_map_stats.get((brawler, mode, tier))
        if v:
            return v["win_rate"]
        return global_stats.get((brawler,), {}).get("win_rate", 0.5)

    # Build same-team pair rows (6 pairs per game: 3C2 × 2 teams)
    frames = []
    for team_cols, is_team1 in [(TEAM1_BRAWLER_COLS, True), (TEAM2_BRAWLER_COLS, False)]:
        for i, j in combinations(range(3), 2):
            ca, cb = team_cols[i], team_cols[j]
            sub = df[["mode", "skill_tier", ca, cb, "team1_wins"]].copy()
            sub["won"] = sub["team1_wins"] if is_team1 else 1 - sub["team1_wins"]
            sub = sub.drop(columns=["team1_wins"]).rename(columns={ca: "a", cb: "b"})
            frames.append(sub)

    pair_df = pd.concat(frames, ignore_index=True)

    # Canonical order: a ≤ b alphabetically
    swap = pair_df["a"] > pair_df["b"]
    pair_df.loc[swap, ["a", "b"]] = pair_df.loc[swap, ["b", "a"]].values

    def _build_level(group_cols: list[str]) -> dict:
        agg = pair_df.groupby(group_cols)["won"].agg(n="count", wins="sum").reset_index()
        agg["win_rate"] = agg["wins"] / agg["n"]
        result = {}
        for row in agg.itertuples(index=False):
            a, b = row.a, row.b
            mode_val = row.mode if "mode" in group_cols else None
            tier_val = int(row.skill_tier) if "skill_tier" in group_cols else None

            if mode_val is not None and tier_val is not None:
                baseline = (_baseline_wr(a, mode_val, tier_val) + _baseline_wr(b, mode_val, tier_val)) / 2
            elif mode_val is not None:
                # Average individual rates across all tiers for this mode
                wa = np.mean([_baseline_wr(a, mode_val, t) for t in range(N_SKILL_TIERS)])
                wb = np.mean([_baseline_wr(b, mode_val, t) for t in range(N_SKILL_TIERS)])
                baseline = (wa + wb) / 2
            else:
                wa = global_stats.get((a,), {}).get("win_rate", 0.5)
                wb = global_stats.get((b,), {}).get("win_rate", 0.5)
                baseline = (wa + wb) / 2

            key = tuple(
                int(getattr(row, c)) if c == "skill_tier" else getattr(row, c)
                for c in group_cols
            )
            result[key] = {
                "n": int(row.n),
                "wins": int(row.wins),
                "win_rate": float(row.win_rate),
                "baseline": float(baseline),
                "synergy_delta": float(row.win_rate) - baseline,
            }
        return result

    stats = {
        "mode_tier": _build_level(["a", "b", "mode", "skill_tier"]),
        "mode":      _build_level(["a", "b", "mode"]),
        "global":    _build_level(["a", "b"]),
    }
    print(
        f"Synergy matrix: {len(stats['global']):,} canonical pairs | "
        f"{len(stats['mode_tier']):,} / {len(stats['mode']):,} / {len(stats['global']):,} cells "
        f"(mode+tier / mode / global)"
    )
    return stats


# ---------------------------------------------------------------------------
# 1.2.4 — MatchupDB: unified in-memory lookup + pickle serialization
# ---------------------------------------------------------------------------

@dataclass
class MatchupDB:
    """
    Unified in-memory matchup database for fast MCTS rollout and scoring.

    All three lookup methods are O(1) flat-dict lookups.
    The 'level' field in each result indicates which fallback was used:
      brawler/counter: 0=full(map+tier), 1=no_map(mode+tier), 2=no_tier(mode), 3=global
      synergy:         0=mode+tier, 1=mode, 2=global

    Build once:  db = MatchupDB.build(); db.save()
    Load fast:   db = MatchupDB.load()

    skill_tier_boundaries stores (Q25, Q50, Q75) of skill_ns from the training
    data, used by skill_ns_to_tier() to map a live session's skill_ns to a tier
    for matchup DB lookups without reloading the dataset.
    """
    brawler: dict[str, dict]
    counter: dict[str, dict]
    synergy: dict[str, dict]
    skill_tier_boundaries: tuple[float, float, float] | None = None

    # ---- Brawler lookup ----

    def brawler_lookup(
        self, brawler: str, mode: str, map_name: str, tier: int
    ) -> dict | None:
        """Returns {'n', 'wins', 'win_rate', 'pick_rate', 'level'} or None."""
        if (k := (brawler, map_name, tier)) in self.brawler["full"]:
            return {**self.brawler["full"][k], "level": 0}
        if (k := (brawler, mode, tier)) in self.brawler["no_map"]:
            return {**self.brawler["no_map"][k], "level": 1}
        if (k := (brawler, mode)) in self.brawler["no_tier"]:
            return {**self.brawler["no_tier"][k], "level": 2}
        if (k := (brawler,)) in self.brawler["global"]:
            return {**self.brawler["global"][k], "level": 3}
        return None

    # ---- Counter lookup ----

    def counter_lookup(
        self, my_brawler: str, opp_brawler: str, mode: str, map_name: str, tier: int
    ) -> dict | None:
        """
        Returns {'n', 'wins', 'win_rate', 'level'} or None.
        win_rate = P(my team wins | my_brawler present vs opp_brawler present).
        Complements: counter_lookup(x, y) + counter_lookup(y, x) win_rates sum to 1.
        """
        if (k := (my_brawler, opp_brawler, map_name, tier)) in self.counter["full"]:
            return {**self.counter["full"][k], "level": 0}
        if (k := (my_brawler, opp_brawler, mode, tier)) in self.counter["no_map"]:
            return {**self.counter["no_map"][k], "level": 1}
        if (k := (my_brawler, opp_brawler, mode)) in self.counter["no_tier"]:
            return {**self.counter["no_tier"][k], "level": 2}
        if (k := (my_brawler, opp_brawler)) in self.counter["global"]:
            return {**self.counter["global"][k], "level": 3}
        return None

    # ---- Synergy lookup ----

    def synergy_lookup(
        self, brawler_a: str, brawler_b: str, mode: str, tier: int
    ) -> dict | None:
        """
        Returns {'n', 'wins', 'win_rate', 'baseline', 'synergy_delta', 'level'} or None.
        Pair order is canonicalized internally — synergy_lookup(A, B) == synergy_lookup(B, A).
        synergy_delta > 0 means the pair wins more often than individual rates predict.
        """
        a, b = min(brawler_a, brawler_b), max(brawler_a, brawler_b)
        if (k := (a, b, mode, tier)) in self.synergy["mode_tier"]:
            return {**self.synergy["mode_tier"][k], "level": 0}
        if (k := (a, b, mode)) in self.synergy["mode"]:
            return {**self.synergy["mode"][k], "level": 1}
        if (k := (a, b)) in self.synergy["global"]:
            return {**self.synergy["global"][k], "level": 2}
        return None

    # ---- Serialization ----

    def save(self, path: Path = DEFAULT_SAVE_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = path.stat().st_size / 1e6
        print(f"MatchupDB saved → {path}  ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Path = DEFAULT_SAVE_PATH) -> "MatchupDB":
        import importlib

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if module == "__main__":
                    for mod_name in ("matchup_db", "src.matchup_db"):
                        try:
                            return getattr(importlib.import_module(mod_name), name)
                        except (ModuleNotFoundError, AttributeError):
                            pass
                return super().find_class(module, name)

        with open(path, "rb") as f:
            db = _Unpickler(f).load()
        print(f"MatchupDB loaded ← {path}")
        return db

    @classmethod
    def build(
        cls,
        source: "DatasetRef | str | Path | None" = None,
        db_path: Path = DB_PATH,
        elo_min: int | None = None,
        elo_max: int | None = None,
    ) -> "MatchupDB":
        """Build all three tables from scratch. Runs the full 1.2 pipeline."""
        from bsdraft.data.prep import ELO_MIN as _EMIN, ELO_MAX as _EMAX

        df, _ = build_game_dataset(
            source if source is not None else db_path,
            elo_min=elo_min if elo_min is not None else _EMIN,
            elo_max=elo_max if elo_max is not None else _EMAX,
        )
        df, boundaries = assign_skill_tiers(df)
        brawler = build_brawler_stats(df)
        counter = build_counter_matrix(df)
        synergy = build_synergy_matrix(df, brawler)
        return cls(
            brawler=brawler,
            counter=counter,
            synergy=synergy,
            skill_tier_boundaries=boundaries,
        )


# ---------------------------------------------------------------------------
# Entry point — also serves as Step 1.2.5 sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    from bsdraft.data.prep import SEASON_CONFIGS

    parser = argparse.ArgumentParser(description="Build MatchupDB for a given season.")
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
    save_path = data_dir / "matchup_db.pkl"

    print(f"Season: {args.season}  |  DB: {cfg['db_path']}  |  ELO [{cfg['elo_min']}, {cfg['elo_max']}]")
    print(f"Output: {save_path}\n")

    # Build and save
    db = MatchupDB.build(
        db_path=cfg["db_path"],
        elo_min=cfg["elo_min"],
        elo_max=cfg["elo_max"],
    )
    db.save(save_path)

    # --- 1.2.5 check 1: per-brawler win rate summary for one mode ---
    mode_sample = "gemGrab"
    rows = [
        {"brawler": k[0], **v}
        for k, v in db.brawler["no_tier"].items()
        if k[1] == mode_sample
    ]
    if rows:
        gdf = pd.DataFrame(rows).sort_values("win_rate", ascending=False)
        print(f"\nTop 5 brawlers in {mode_sample} by win rate:")
        print(gdf.head()[["brawler", "n", "win_rate", "pick_rate"]].to_string(index=False))
        print(f"\nBottom 5 brawlers in {mode_sample}:")
        print(gdf.tail()[["brawler", "n", "win_rate", "pick_rate"]].to_string(index=False))
    else:
        print(f"No mode '{mode_sample}' found — check mode names in your dataset.", file=sys.stderr)

    # --- 1.2.5 check 2: counter matrix — CROW vs opponents, verify directionality ---
    sample_brawler = "CROW"
    # Use a real map name from the dataset; fall back gracefully if not found
    sample_maps = sorted({k[2] for k in db.counter["full"] if k[0] == sample_brawler})
    map_sample = sample_maps[0] if sample_maps else None
    tier_sample = 2
    if map_sample:
        counter_rows = [
            {"opp": k[1], **v}
            for k, v in db.counter["full"].items()
            if k[0] == sample_brawler and k[2] == map_sample and k[3] == tier_sample
        ]
        cdf = pd.DataFrame(counter_rows).sort_values("win_rate", ascending=False)
        print(f"\nCounter: {sample_brawler} (my team) vs opponents on '{map_sample}' / tier {tier_sample}")
        print(cdf.head(5)[["opp", "n", "win_rate"]].to_string(index=False))

        # Directionality check: P(CROW→X) + P(X→CROW) ≈ 1 for first opponent
        opp = cdf.iloc[0]["opp"]
        fwd = db.counter_lookup(sample_brawler, opp, mode_sample, map_sample, tier_sample)
        rev = db.counter_lookup(opp, sample_brawler, mode_sample, map_sample, tier_sample)
        if fwd and rev:
            total = fwd["win_rate"] + rev["win_rate"]
            print(f"\nSymmetry check {sample_brawler}→{opp}: {fwd['win_rate']:.4f} + {rev['win_rate']:.4f} = {total:.4f} (should be 1.0)")

    # --- 1.2.5 check 3: fallback lookup for sparse / nonexistent map ---
    r = db.brawler_lookup("LOLA", "gemGrab", "NONEXISTENT_MAP", 3)
    print(f"\nFallback: LOLA / gemGrab / NONEXISTENT_MAP / tier 3  →  level {r['level'] if r else 'None'}")

    r = db.counter_lookup("CROW", "POCO", "gemGrab", "NONEXISTENT_MAP", 3)
    print(f"Fallback: CROW vs POCO / gemGrab / NONEXISTENT_MAP / tier 3  →  level {r['level'] if r else 'None'}")

    # --- Synergy spot-check ---
    synergy_rows = sorted(
        [{"a": k[0], "b": k[1], **v} for k, v in db.synergy["global"].items()],
        key=lambda x: x["synergy_delta"],
        reverse=True,
    )
    print("\nTop 5 global synergy pairs:")
    sdf = pd.DataFrame(synergy_rows[:5])
    print(sdf[["a", "b", "n", "win_rate", "baseline", "synergy_delta"]].to_string(index=False))
