"""Everything the dashboard needs, as one JSON payload.

The model is 29,354 parameters — about 230 KB of JSON — so it ships to the
browser whole and the page runs inference itself. No server, no API, no latency:
a visitor changes a pick and the probability updates as they watch.

That is only possible because the model is small, which is a consequence of the
factorized design rather than an accident.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bsdraft.data.sources import TEAM1_BRAWLER_COLS, TEAM2_BRAWLER_COLS
from bsdraft.fm.ffm import FFMInference

#: Four decimals costs nothing in accuracy and roughly halves the payload.
PRECISION = 4


def _round(a: np.ndarray) -> list:
    return np.round(np.asarray(a, dtype=np.float64), PRECISION).tolist()


def serialise_model(model: FFMInference) -> dict[str, Any]:
    """The weights, in the layout the page's inference code expects."""
    return {
        "vocab": list(model.vocab),
        "maps": list(model.maps),
        "modes": list(model.modes),
        "k": int(model.e_syn.shape[1]),
        "w": _round(model.w),
        "e_syn": _round(model.e_syn),
        "e_att": _round(model.e_att),
        "e_def": _round(model.e_def),
        "e_ctx": _round(model.e_ctx),
        "m_map": _round(model.m_map),
        "m_mode": _round(model.m_mode),
        "v_skill": _round(model.v_skill),
        "metrics": {
            "logloss": round(float(model.val_logloss), 4),
            "auc": round(float(model.val_auc), 4),
            "brier": round(float(model.val_brier), 4),
            "n_train": int(model.n_train),
            "n_val": int(model.n_val),
        },
    }


def _map_modes(df: pd.DataFrame) -> dict[str, str]:
    """Each map is played in exactly one mode, so the page can infer it."""
    pairs = df[["map", "mode"]].drop_duplicates()
    return {str(r.map): str(r.mode) for r in pairs.itertuples()}


def character_stats(df: pd.DataFrame, min_games: int = 200) -> list[dict]:
    """Pick rate and win rate per character, from the season's games."""
    frames = []
    for cols, won in ((TEAM1_BRAWLER_COLS, df["team1_wins"]),
                      (TEAM2_BRAWLER_COLS, 1 - df["team1_wins"])):
        for c in cols:
            frames.append(pd.DataFrame({"name": df[c], "won": won}))
    stacked = pd.concat(frames, ignore_index=True).dropna(subset=["name"])
    grouped = stacked.groupby("name")["won"].agg(["sum", "count"])
    grouped = grouped[grouped["count"] >= min_games]

    total_slots = len(df) * 6
    out = [
        {
            "name": name,
            "games": int(row["count"]),
            "pick_rate": round(float(row["count"]) / total_slots, 5),
            "win_rate": round(float(row["sum"]) / float(row["count"]), 4),
        }
        for name, row in grouped.iterrows()
    ]
    return sorted(out, key=lambda r: -r["games"])


def season_stats(df: pd.DataFrame, season: str, dataset: str) -> dict[str, Any]:
    """Headline numbers and per-day volume for the overview."""
    day = df["battle_time"].str[:8]
    daily = day.value_counts().sort_index()
    return {
        "season": season,
        "dataset": dataset,
        "games": int(len(df)),
        "modes": sorted(df["mode"].dropna().unique().tolist()),
        "maps": sorted(df["map"].dropna().unique().tolist()),
        "first_day": str(daily.index[0]) if len(daily) else None,
        "last_day": str(daily.index[-1]) if len(daily) else None,
        "team1_win_rate": round(float(df["team1_wins"].mean()), 4),
        "daily": [{"day": d, "games": int(n)} for d, n in daily.items()],
        "map_modes": _map_modes(df),
    }


#: Measured on the same held-out games as the model. Shown so a visitor can see
#: what the model is worth relative to counting, not just an unanchored number.
BASELINES = [
    {"name": "Coin flip", "knows": "Nothing", "logloss": 0.6931, "auc": 0.500, "ece": 0.0012},
    {"name": "Character win rates", "knows": "Which characters win",
     "logloss": 0.6850, "auc": 0.597, "ece": 0.0455},
    {"name": "Character × map", "knows": "…and where they win",
     "logloss": 0.6794, "auc": 0.624, "ece": 0.0572},
    {"name": "Head-to-head rates", "knows": "Which beat which",
     "logloss": 0.6834, "auc": 0.612, "ece": 0.0560},
]


def build_payload(
    model: FFMInference, df: pd.DataFrame, *, season: str, dataset: str,
    generated_utc: str,
) -> dict[str, Any]:
    return {
        "generated_utc": generated_utc,
        "model": serialise_model(model),
        "season": season_stats(df, season, dataset),
        "characters": character_stats(df),
        "baselines": BASELINES,
    }


def write_payload(payload: dict, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, separators=(",", ":")))
    return p
