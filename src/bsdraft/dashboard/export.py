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

from bsdraft.data.sources import (
    ALL_ELO_COLS,
    TEAM1_BRAWLER_COLS,
    TEAM2_BRAWLER_COLS,
)
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


def _unit(a: np.ndarray) -> np.ndarray:
    """Scale a field so no one of them dominates the concatenation."""
    return a / (float(np.linalg.norm(a, axis=1).mean()) + 1e-9)


def interaction_space(model: FFMInference) -> np.ndarray:
    """How each character interacts: beside, against, and defending against.

    Deliberately excludes the context field and the linear weight — those say
    how *strong* a character is, and mixing strength into a similarity map would
    put every strong character together regardless of how they play.
    """
    return np.hstack([_unit(model.e_syn), _unit(model.e_att), _unit(model.e_def)])


def embed_characters(model: FFMInference, *, seed: int = 0) -> dict[str, Any]:
    """Two-dimensional layouts of the interaction space, plus neighbours.

    PCA keeps distances meaningful but explains only about a fifth of the
    variance in two dimensions. t-SNE separates groups far more legibly at the
    cost of global geometry. Both ship; the page lets a reader switch, because
    they answer different questions.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X = interaction_space(model)
    pca = PCA(n_components=2, random_state=seed)
    xy_pca = pca.fit_transform(X)
    xy_tsne = TSNE(n_components=2, perplexity=18, init="pca",
                   random_state=seed, max_iter=1200).fit_transform(X)

    unit = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    neighbours = {
        model.vocab[i]: [model.vocab[j] for j in np.argsort(-sim[i])[:4]]
        for i in range(len(model.vocab))
    }

    def scaled(a):
        a = np.asarray(a, dtype=np.float64)
        span = a.max(axis=0) - a.min(axis=0)
        span[span == 0] = 1.0
        return np.round((a - a.min(axis=0)) / span, 4).tolist()

    return {
        "pca": scaled(xy_pca),
        "tsne": scaled(xy_tsne),
        "pca_variance": round(float(pca.explained_variance_ratio_[:2].sum()), 4),
        "neighbours": neighbours,
    }


def _map_modes(df: pd.DataFrame) -> dict[str, str]:
    """Each map is played in exactly one mode, so the page can infer it."""
    pairs = df[["map", "mode"]].drop_duplicates()
    return {str(r.map): str(r.mode) for r in pairs.itertuples()}


#: Skill bands, as equal slices of the season's lobbies. Five rather than three
#: so the page's skill slider crosses a boundary four times instead of twice —
#: with three, most of the slider's travel changed nothing a reader could see.
SKILL_BAND_LABELS = ("bottom 20%", "20-40%", "40-60%", "60-80%", "top 20%")
N_SKILL_BANDS = len(SKILL_BAND_LABELS)


def _skill_band(df: pd.DataFrame, n_bands: int = N_SKILL_BANDS) -> pd.Series:
    """Which equal slice of the skill distribution each game sits in, 0-indexed."""
    ranks = df["skill_ns"].rank(pct=True, method="average")
    return np.minimum((ranks * n_bands).astype(int), n_bands - 1)


def season_maps(df: pd.DataFrame) -> list[str]:
    """The map order everything else indexes against.

    The per-character grid is positional, so the page can only read it if it
    agrees with this list exactly. One definition, used by both.
    """
    return sorted(df["map"].dropna().unique().tolist())


def character_stats(df: pd.DataFrame, min_games: int = 200) -> list[dict]:
    """Pick rate and win rate per character, sliced by map and lobby skill.

    Each character carries a positional `grid`: for every (map, skill band) pair
    in `season_maps(df)` x `SKILL_BAND_LABELS` order, the appearances and wins
    in that slice. The page sums whatever cells the current filters select and
    recomputes the rate, so a filtered view never quietly shows a season-wide
    number — and coarser views (a whole mode, or every band) are sums of the
    same cells rather than a second copy of the data.

    Storing it positionally rather than under `"<map>|<band>"` keys is what
    makes the resolution affordable: the keys were four fifths of the bytes, so
    26 maps x 5 bands costs about what 6 modes x 3 bands did.
    """
    frames = []
    for cols, won in ((TEAM1_BRAWLER_COLS, df["team1_wins"]),
                      (TEAM2_BRAWLER_COLS, 1 - df["team1_wins"])):
        for c in cols:
            frames.append(pd.DataFrame({"name": df[c], "won": won}))
    stacked = pd.concat(frames, ignore_index=True).dropna(subset=["name"])
    grouped = stacked.groupby("name")["won"].agg(["sum", "count"])
    grouped = grouped[grouped["count"] >= min_games]

    maps = season_maps(df)
    map_ix = {m: i for i, m in enumerate(maps)}
    band = _skill_band(df)
    width = len(maps) * N_SKILL_BANDS * 2

    grids: dict[str, list[int]] = {}
    for cols, won in ((TEAM1_BRAWLER_COLS, df["team1_wins"]),
                      (TEAM2_BRAWLER_COLS, 1 - df["team1_wins"])):
        for c in cols:
            sub = pd.DataFrame({"name": df[c], "map": df["map"],
                                "band": band, "won": won}).dropna(subset=["name"])
            agg = sub.groupby(["name", "map", "band"])["won"].agg(["sum", "count"])
            for (name, mp, bnd), row in agg.iterrows():
                mi = map_ix.get(str(mp))
                if mi is None:
                    continue
                g = grids.setdefault(name, [0] * width)
                at = (mi * N_SKILL_BANDS + int(bnd)) * 2
                g[at] += int(row["count"])
                g[at + 1] += int(row["sum"])

    total_slots = len(df) * 6
    out = [
        {
            "name": name,
            "games": int(row["count"]),
            "pick_rate": round(float(row["count"]) / total_slots, 5),
            "win_rate": round(float(row["sum"]) / float(row["count"]), 4),
            # (map, band) -> [appearances, wins], flattened in that order.
            "grid": grids.get(name, [0] * width),
        }
        for name, row in grouped.iterrows()
    ]
    return sorted(out, key=lambda r: -r["games"])


#: Histogram resolution. Ratings are integers, so a bin per rating point is
#: exact.
#:
#: skill_ns only looks continuous. It is built from `avg_elo`, which is the mean
#: of six integers and therefore takes 58 distinct values in a season; a single
#: skill_ns value can hold eight per cent of the matches. Bin it finely and
#: those atoms land unevenly across the grid, and the chart reads as noise when
#: it is really showing a spiky discrete variable. A quarter of a standard
#: deviation measured as the least ragged of the widths tried, so it is what
#: the chart uses.
ELO_BIN = 1.0
SKILL_LO, SKILL_HI, SKILL_BIN = -3.0, 3.0, 0.25
SKILL_COL = "skill_ns"


def rating_distribution(df: pd.DataFrame) -> dict[str, Any]:
    """How the season's ratings are spread, day by day.

    Counted per drafted slot rather than per match. A match carries six players
    and `avg_elo` is their mean, so a histogram of the average is visibly
    narrower than the spread of the players themselves -- it would understate
    exactly the tails a reader is looking for. Both views here count the same
    six slots per match, so switching between them changes the scale being
    measured and nothing else.

    Buckets are kept per day so the page can sum any date range itself, which
    is cheaper and far more responsive than shipping a histogram per range.
    """
    if "battle_time" not in df or df.empty:
        return {}
    # One row per set: the modelling frame has a row per *game*, so counting it
    # directly would weight a three-game set three times.
    sets = df.drop_duplicates(subset="id") if "id" in df else df
    day = sets["battle_time"].str[:8]
    days = sorted(day.dropna().unique().tolist())
    if not days:
        return {}
    day_ix = {d: i for i, d in enumerate(days)}
    rows = day.map(day_ix).to_numpy()

    elo_cols = [c for c in ALL_ELO_COLS if c in sets.columns]
    out: dict[str, Any] = {"days": days, "slots_per_match": 6}

    def histogram(values: np.ndarray, at: np.ndarray, lo: float, hi: float,
                  width: float) -> dict[str, Any]:
        n_bins = max(1, int(round((hi - lo) / width)))
        ix = np.clip(((values - lo) / width).astype(int), 0, n_bins - 1)
        flat = np.bincount(at * n_bins + ix, minlength=len(days) * n_bins)
        return {"lo": round(lo, 3), "width": round(width, 3),
                "counts": flat.reshape(len(days), n_bins).tolist()}

    if elo_cols:
        vals = sets[elo_cols].to_numpy(dtype="float64")
        at = np.repeat(rows, len(elo_cols))
        flat = vals.reshape(-1)
        ok = ~np.isnan(flat)
        flat, at = flat[ok], at[ok]
        if flat.size:
            lo = float(np.floor(flat.min()))
            hi = float(np.ceil(flat.max())) + ELO_BIN
            out["elo"] = histogram(flat, at, lo, hi, ELO_BIN)

    if SKILL_COL in sets.columns:
        skill = sets[SKILL_COL].to_numpy(dtype="float64")
        ok = ~np.isnan(skill)
        # Each slot inherits its match's lobby skill, so the two views count the
        # same population and their totals agree.
        vals = np.repeat(skill[ok], len(elo_cols) or 6)
        at = np.repeat(rows[ok], len(elo_cols) or 6)
        if vals.size:
            out["skill"] = histogram(vals, at, SKILL_LO, SKILL_HI, SKILL_BIN)
    return out


def season_stats(df: pd.DataFrame, season: str, dataset: str) -> dict[str, Any]:
    """Headline numbers and per-day volume for the overview."""
    day = df["battle_time"].str[:8]
    daily = day.value_counts().sort_index()
    return {
        "season": season,
        "dataset": dataset,
        "games": int(len(df)),
        "modes": sorted(df["mode"].dropna().unique().tolist()),
        "maps": season_maps(df),
        "first_day": str(daily.index[0]) if len(daily) else None,
        "last_day": str(daily.index[-1]) if len(daily) else None,
        "team1_win_rate": round(float(df["team1_wins"].mean()), 4),
        "daily": [{"day": d, "games": int(n)} for d, n in daily.items()],
        "map_modes": _map_modes(df),
        "skill_bands": list(SKILL_BAND_LABELS),
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


def training_history(store_path: str | Path, limit: int = 10) -> list[dict]:
    """Recent training runs, so the page can answer what happened this week.

    Reads whatever the registry holds; an absent or unreadable registry yields
    an empty list rather than breaking the build.
    """
    try:
        from bsdraft.tracking import RunStore

        if not Path(store_path).exists():
            return []
        store = RunStore(store_path)
    except Exception:
        return []

    out = []
    for run in store.list_runs(limit=limit * 3):
        full = store.get_run(run["run_id"])
        if full is None:
            continue
        m = full.get("metrics", {})
        row = {
            "run_id": run["run_id"],
            "stage": run["stage"],
            "status": run["status"],
            "started": (run["started_utc"] or "")[:16].replace("T", " "),
            "seed": run["seed"],
            "elapsed": run["elapsed_seconds"],
        }
        if run["stage"] == "fm":
            row |= {"logloss": m.get("val_logloss"), "auc": m.get("val_auc")}
        elif run["stage"] == "selfplay":
            wp = m.get("final_eval_win_prob") or m.get("eval_win_prob")
            row |= {"win_prob": wp,
                    "promoted": bool(m.get("final_promoted") or m.get("promoted"))}
        elif run["stage"] == "eval":
            best = min(
                ((k.rsplit(".", 1)[0], v) for k, v in m.items() if k.endswith(".logloss")),
                key=lambda kv: kv[1], default=None,
            )
            row |= {"best_predictor": best[0] if best else None,
                    "logloss": best[1] if best else None}
        out.append(row)
        if len(out) >= limit:
            break
    return out


def build_payload(
    model: FFMInference, df: pd.DataFrame, *, season: str, dataset: str,
    generated_utc: str, registry_path: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "generated_utc": generated_utc,
        "runs": training_history(registry_path) if registry_path else [],
        "model": serialise_model(model),
        "embedding": embed_characters(model),
        "season": season_stats(df, season, dataset),
        "characters": character_stats(df),
        "ratings": rating_distribution(df),
        "baselines": BASELINES,
    }


def write_payload(payload: dict, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, separators=(",", ":")))
    return p
