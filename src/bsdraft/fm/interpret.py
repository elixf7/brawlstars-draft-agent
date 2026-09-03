"""
fm_interpret.py — Step 1.6: FM Interpretability Analysis

Utility functions that extract and analyze the learned embeddings from a
trained FMInference object.

Key ideas:
  - The FM learns SEPARATE embeddings for t1_BRAWLER (on my team) and
    t2_BRAWLER (on opponent's team). Use the right one for each analysis.
  - Synergy  = ⟨v_t1_A, v_t1_B⟩  — both on my team
  - Counter  = ⟨v_t1_A, v_t2_B⟩  — A on mine, B on opponent
  - Skill    = ⟨v_skill_ns, v_t1_B⟩  — how much brawler B scales with skill
  - net_power = w_linear[t1_i] − w_linear[t2_i]  — first-order brawler strength

All functions are stateless and return DataFrames ready for display or plotting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from bsdraft.fm.model import FMInference


# ---------------------------------------------------------------------------
# Raw embedding extractors
# ---------------------------------------------------------------------------

def get_brawler_embeddings(inf: FMInference) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    t1_emb : (V, k) — embeddings for brawlers on MY team
    t2_emb : (V, k) — embeddings for brawlers on OPPONENT'S team
    """
    sc = inf.schema
    return (
        inf.V[sc.t1_offset: sc.t2_offset],   # (V, k)
        inf.V[sc.t2_offset: sc.map_offset],   # (V, k)
    )


def get_map_embeddings(inf: FMInference) -> np.ndarray:
    """Map embedding matrix: (M, k)."""
    sc = inf.schema
    return inf.V[sc.map_offset: sc.mode_offset]


def get_mode_embeddings(inf: FMInference) -> np.ndarray:
    """Mode embedding matrix: (Mo, k)."""
    sc = inf.schema
    return inf.V[sc.mode_offset: sc.skill_offset]


def get_skill_embedding(inf: FMInference) -> np.ndarray:
    """Skill_ns embedding vector: (k,)."""
    return inf.V[inf.schema.skill_offset]


# ---------------------------------------------------------------------------
# Brawler synergy & counter (second-order interactions)
# ---------------------------------------------------------------------------

def top_counter_pairs(inf: FMInference, n: int = 10) -> pd.DataFrame:
    """
    Top counter pairs by FM embedding dot product.

    Counter = ⟨v_t1_A, v_t2_B⟩ — A on MY team, B on OPPONENT.
    High positive value → A strongly counters B.

    Columns: my_brawler, opp_brawler, fm_dot
    """
    t1_emb, t2_emb = get_brawler_embeddings(inf)
    vocab = inf.schema.vocab
    V = len(vocab)

    dot = t1_emb @ t2_emb.T  # (V, V)

    rows = [
        (vocab[i], vocab[j], float(dot[i, j]))
        for i in range(V)
        for j in range(V)
        if i != j
    ]
    df = pd.DataFrame(rows, columns=["my_brawler", "opp_brawler", "fm_dot"])
    return df.nlargest(n, "fm_dot").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Brawler importance (first-order linear weights)
# ---------------------------------------------------------------------------

def brawler_importance(inf: FMInference) -> pd.DataFrame:
    """
    Brawler importance from FM linear (first-order) weights.

    w_t1[i] = contribution of having brawler i on MY team (all else equal).
    w_t2[i] = contribution of having brawler i on OPPONENT'S team.
    net_power = w_t1[i] − w_t2[i]
      Positive → strong brawler: helps you when you have it, hurts you when opponent has it.
      Negative → weak brawler.

    Columns: brawler, w_t1, w_t2, net_power
    """
    sc = inf.schema
    w_t1 = inf.w_linear[sc.t1_offset: sc.t2_offset].astype(float)
    w_t2 = inf.w_linear[sc.t2_offset: sc.map_offset].astype(float)
    net  = w_t1 - w_t2

    df = pd.DataFrame({
        "brawler":   sc.vocab,
        "w_t1":      w_t1,
        "w_t2":      w_t2,
        "net_power": net,
    })
    return df.sort_values("net_power", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Skill scaling (skill_ns interaction)
# ---------------------------------------------------------------------------

def skill_scaling_ranking(inf: FMInference) -> pd.DataFrame:
    """
    Rank brawlers by their interaction with skill_ns.

    The FM interaction term ⟨v_skill_ns, v_t1_B⟩ × skill_ns_value scales
    brawler B's win contribution by the player's skill level.
    High dot product → high-skill-ceiling brawler.

    Columns: brawler, skill_dot
    """
    t1_emb, _ = get_brawler_embeddings(inf)
    skill_emb = get_skill_embedding(inf)
    dots = (t1_emb @ skill_emb).astype(float)

    df = pd.DataFrame({"brawler": inf.schema.vocab, "skill_dot": dots})
    return df.sort_values("skill_dot", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Map analysis
# ---------------------------------------------------------------------------

def map_skill_affinity(inf: FMInference) -> pd.DataFrame:
    """
    Rank maps by their interaction with skill_ns.

    ⟨v_skill_ns, v_map_j⟩ captures how much the map amplifies skill differences.
    High value → high-skill players gain more edge on this map.

    Columns: map, skill_dot
    """
    map_emb   = get_map_embeddings(inf)
    skill_emb = get_skill_embedding(inf)
    dots = (map_emb @ skill_emb).astype(float)

    df = pd.DataFrame({"map": inf.schema.maps, "skill_dot": dots})
    return df.sort_values("skill_dot", ascending=False).reset_index(drop=True)


