"""
draft_state.py — Step 2.1: Draft State Representation

Defines DraftState (frozen, hashable) and two pure functions:
  available_actions(state, vocab) → list of legal next picks
  apply_pick(state, brawler)      → new DraftState after one pick

Pick order (confirmed, game client verified):
  P1 (first-pick team):  [mine, opp, opp, mine, mine, opp]
  P2 (second-pick team): [opp, mine, mine, opp, opp, mine]

Who is P1 vs P2 is determined by a coin flip — callers must pass
is_first_pick=True/False to reflect the actual match assignment.

whose_turn is always derived from pick_number = len(my_team) + len(opp_team),
never stored explicitly, keeping the state minimal and derivation consistent.

Note on transposition table (Step 2.1.1):
  Transpositions do exist in this pick format: P1 picks at slots 0 and 3, so
  (A then B) and (B then A) both produce my_team={A, B} — same frozenset state,
  different paths. Skipping the transposition table because:
    (a) The FM evaluation cache (Step 2.3.1) eliminates redundant terminal calls,
        which is where all the expensive work happens.
    (b) Non-terminal transpositions only save cheap Python tree traversal.
    (c) Merging values across shared subtrees adds correctness complexity
        disproportionate to the gain at 6-pick depth and 10–20k simulations.
  Revisit only if profiling (Step 2.8.1) shows this is a significant bottleneck.

Note on bans (Step 2.1.2):
  bans filters the available pick pool — it has no effect on FM evaluation
  or matchup DB lookups. This models live ranked: the user enters known bans
  at call time to restrict what can be recommended.
  Training data note: the API does not expose ban data, so the training data
  reflects post-ban compositions from an unobservable pool. Frequently banned
  high-tier brawlers appear less often than their true win rate would suggest;
  the FM may slightly underestimate them. Accepted limitation for now.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Pick sequences indexed by pick_number (0–5).
# Coin flip at match start determines P1/P2 assignment — caller specifies
# is_first_pick when constructing the initial DraftState.
_P1_ORDER: tuple[str, ...] = ("mine", "opp", "opp", "mine", "mine", "opp")
_P2_ORDER: tuple[str, ...] = ("opp", "mine", "mine", "opp", "opp", "mine")


@dataclass(frozen=True)
class DraftState:
    """
    Immutable snapshot of one point in the ranked 3v3 draft.

    Frozen so it is hashable — safe to use as a dict key if needed.

    Parameters
    ----------
    my_team       : brawlers already picked by my team (0–3 members)
    opp_team      : brawlers already picked by the opponent (0–3 members)
    mode          : game mode string (e.g. "gemGrab")
    map_name      : map name string (e.g. "Double Swoosh")
    skill_ns      : continuous skill score for the session (logit-ECDF percentile).
                    Constant for the entire draft — set once at session start.
    is_first_pick : True if my team is P1 (picks first); False if P2
    bans          : brawlers excluded from the pick pool (default: empty)
    """

    my_team: frozenset[str]
    opp_team: frozenset[str]
    mode: str
    map_name: str
    skill_ns: float
    is_first_pick: bool
    bans: frozenset[str] = field(default_factory=frozenset)

    @property
    def pick_number(self) -> int:
        """0-based index of the current pick (0 = draft start, 5 = last pick)."""
        return len(self.my_team) + len(self.opp_team)

    @property
    def whose_turn(self) -> str:
        """'mine' or 'opp' — who makes the next pick."""
        n = self.pick_number
        if n >= 6:
            raise ValueError(f"Draft is already complete (pick_number={n}).")
        return (_P1_ORDER if self.is_first_pick else _P2_ORDER)[n]

    @property
    def is_terminal(self) -> bool:
        """True when both teams have exactly 3 brawlers."""
        return len(self.my_team) == 3 and len(self.opp_team) == 3


def available_actions(state: DraftState, vocab: tuple[str, ...]) -> list[str]:
    """
    Return all brawlers that can legally be picked next.

    A brawler is available if it is in the FM vocabulary, not yet picked
    by either team, and not banned.

    Parameters
    ----------
    state : current draft state
    vocab : full brawler vocabulary from FeatureSchema.vocab (95 brawlers)

    Returns
    -------
    List of available brawler names in vocabulary order (deterministic).
    Vocabulary order is used (not sorted) to match FeatureSchema.vocab exactly.
    """
    used = state.my_team | state.opp_team | state.bans
    return [b for b in vocab if b not in used]


def apply_pick(state: DraftState, brawler: str) -> DraftState:
    """
    Return a new DraftState after `brawler` is picked by whoever's turn it is.

    Does not validate legality — callers must only pass brawlers from
    available_actions(state, vocab).
    """
    if state.whose_turn == "mine":
        return DraftState(
            my_team=state.my_team | {brawler},
            opp_team=state.opp_team,
            mode=state.mode,
            map_name=state.map_name,
            skill_ns=state.skill_ns,
            is_first_pick=state.is_first_pick,
            bans=state.bans,
        )
    else:
        return DraftState(
            my_team=state.my_team,
            opp_team=state.opp_team | {brawler},
            mode=state.mode,
            map_name=state.map_name,
            skill_ns=state.skill_ns,
            is_first_pick=state.is_first_pick,
            bans=state.bans,
        )


if __name__ == "__main__":
    # Step 2.1 sanity checks

    _STUB_VOCAB = (
        "BRAWLER_A", "BRAWLER_B", "BRAWLER_C", "BRAWLER_D",
        "BRAWLER_E", "BRAWLER_F", "BRAWLER_G",
    )

    print("=== Step 2.1 Sanity Checks ===\n")

    # ── Check 1: whose_turn matches both pick sequences end-to-end ────────────
    for label, is_p1, order in [
        ("P1", True, _P1_ORDER),
        ("P2", False, _P2_ORDER),
    ]:
        state = DraftState(
            my_team=frozenset(), opp_team=frozenset(),
            mode="gemGrab", map_name="Double Swoosh",
            skill_ns=1.0, is_first_pick=is_p1,
        )
        print(f"Pick order ({label}):")
        for i, expected in enumerate(order):
            assert state.whose_turn == expected, (
                f"pick {i}: expected {expected!r}, got {state.whose_turn!r}"
            )
            brawler = _STUB_VOCAB[i]
            state = apply_pick(state, brawler)
            print(f"  pick {i}: {expected:4s}  → picks {brawler}")
        assert state.is_terminal, "Expected terminal after 6 picks"
        print("  is_terminal = True  ✓\n")

    # ── Check 2: available_actions respects picks and bans ────────────────────
    state = DraftState(
        my_team=frozenset({"BRAWLER_A"}),
        opp_team=frozenset({"BRAWLER_B"}),
        mode="gemGrab", map_name="Double Swoosh",
        skill_ns=1.0, is_first_pick=True,
        bans=frozenset({"BRAWLER_C"}),
    )
    actions = available_actions(state, _STUB_VOCAB)
    expected_count = len(_STUB_VOCAB) - 3  # 2 picked + 1 banned
    assert len(actions) == expected_count, f"Expected {expected_count}, got {len(actions)}"
    assert not ({"BRAWLER_A", "BRAWLER_B", "BRAWLER_C"} & set(actions)), (
        "Picked/banned brawler appeared in actions"
    )
    print(f"available_actions (2 picked, 1 banned): {actions}")
    print(f"  count = {len(actions)}  (expected {expected_count} from {len(_STUB_VOCAB)})  ✓\n")

    # ── Check 3: apply_pick on terminal raises on whose_turn ─────────────────
    terminal = DraftState(
        my_team=frozenset({"A", "B", "C"}),
        opp_team=frozenset({"D", "E", "F"}),
        mode="gemGrab", map_name="Double Swoosh",
        skill_ns=1.0, is_first_pick=True,
    )
    assert terminal.is_terminal
    try:
        _ = terminal.whose_turn
        raise AssertionError("Expected ValueError")
    except ValueError:
        print("whose_turn on terminal raises ValueError  ✓\n")

    print("=== All checks passed ===")
