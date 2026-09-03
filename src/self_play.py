"""
self_play.py — Part 4.1: Self-Play Data Generation

Two symmetric MCTS agents (sharing the same FM + matchup DB) play complete
6-pick drafts against each other. Every pick generates a SelfPlayRecord that
becomes training data for the policy network (Part 4.2).

Architecture
------------
Each game is played from a single canonical perspective (P1 or P2, alternating
by game_idx for a 50/50 training balance). At each pick, the acting team runs
_run_mcts from ITS OWN perspective:

  Primary team picks : _run_mcts runs on the canonical game_state directly.
  Opposing team picks : a mirrored DraftState is created (swap my_team/opp_team,
                        flip is_first_pick) so that whose_turn == "mine" from
                        the opposing agent's point of view.

Every stored SelfPlayRecord has state.whose_turn == "mine" — it always
represents "I am about to pick; here is the board." This is exactly the input
the policy network needs. The is_first_pick flag and pick_number embedded in
DraftState let the network distinguish P1 vs P2 situations.

Speed optimisations
-------------------
  - Tree vocab filtered once per game (min_pick_rate filter on map/mode/tier),
    reducing branching factor from ~90 to ~40-50. This alone cuts MCTS node
    expansion cost roughly in half.

  - RolloutWeightCache built once per game and passed to all 6 _run_mcts calls.
    The O(V²) counter-matrix construction (~10-50 ms) is paid only once instead
    of 6 times per game, saving ~60-300 ms/game.

  - Default 2000 sims/pick: the FM value function is accurate enough that 2000
    sims produces a trainable visit distribution. The top brawler accumulates
    ~200-400 visits out of 2000, giving a clear signal without spending minutes
    per pick.

  - Multiprocessing: games are independent. Pool(n_workers) with a per-process
    initializer loads FM + DB once per worker, giving near-linear speedup.

Replay buffer
-------------
ReplayBuffer is a bounded in-memory deque over complete games (each a
list[SelfPlayRecord]). When full, the oldest game is evicted automatically.
Batches are persisted to disk as pickle files and tracked via manifest.json.
load() reconstructs the buffer from the most recent batches up to max_games.
Discarding old games avoids training on stale data from weak early policies.
"""

from __future__ import annotations

import json
import pickle
from collections import deque
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from src.draft_state import DraftState, apply_pick
    from src.fm_integration import FMEvaluator
    from src.fm_model import FMInference
    from src.matchup_db import MatchupDB, skill_ns_to_tier
    from src.mcts_node import UCB1_C
    from src.policy_net import PolicyInference
    from src.joint_net import JointNetInference
    from src.recommend import _run_mcts
    from src.rollout import (
        DEFAULT_MIN_COUNTER_GAMES,
        DEFAULT_MY_COUNTER_WEIGHT,
        DEFAULT_OPP_COUNTER_WEIGHT,
        RolloutWeightCache,
    )
except ModuleNotFoundError:
    from draft_state import DraftState, apply_pick
    from fm_integration import FMEvaluator
    from fm_model import FMInference
    from matchup_db import MatchupDB, skill_ns_to_tier
    from mcts_node import UCB1_C
    from policy_net import PolicyInference
    from joint_net import JointNetInference
    from recommend import _run_mcts
    from rollout import (
        DEFAULT_MIN_COUNTER_GAMES,
        DEFAULT_MY_COUNTER_WEIGHT,
        DEFAULT_OPP_COUNTER_WEIGHT,
        RolloutWeightCache,
    )


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_N_SIMS_PER_PICK: int = 2_000
"""
Simulation budget per pick slot.

With ~90 vocab brawlers filtered to ~45 via min_pick_rate, 2000 sims allocate
roughly 44 initial visits (one per brawler) then concentrate the remaining
~1955 on promising children. The best brawler accumulates ~200-400 visits —
enough for a stable, trainable visit distribution. At ~50k FM evals/sec, one
pick takes ~0.2 s; a full 6-pick game ~1.2 s per worker process.
"""

DEFAULT_SKILL_NS_CHOICES: tuple[float, ...] = (1.0, 2.0, 3.0)
"""Casual / mid / elite skill tiers — sampled uniformly per game."""

DEFAULT_VISIT_DIST_TOP_N: int = 20
"""
Store visit fractions for the top-20 most-visited children.

With 2000 sims across ~45 tree-vocab brawlers, roughly 20-35 receive any
visits. The top-20 capture >90% of the probability mass while keeping
each SelfPlayRecord small.
"""

DEFAULT_PICK_TEMPERATURE: float = 1.0
"""
Temperature for sampling the actual pick from the visit distribution.

T=1.0 — proportional to visit counts: diverse games, good state-space coverage.
T→0  — always pick the argmax: strongest play but low training diversity.

T=1.0 is the AlphaZero default for data generation. The stored visit_dist in
SelfPlayRecord always uses raw visit fractions (unaffected by temperature);
temperature only controls which brawler is actually picked in the game.
"""

DEFAULT_MAX_GAMES: int = 10_000
"""
Maximum complete games retained in the replay buffer (≈60k records at 6/game).

When full, the oldest game is evicted before the newest is added. This prevents
the policy from training on stale games generated by a weaker earlier policy.
"""

# Self-play-specific MCTS defaults (separate from interactive recommend defaults)
_SP_OPP_ROLLOUT_MODE: str = "top_k"
_SP_OPP_TOP_K: int = 3
_SP_MIN_PICK_RATE: float = 0.005   # keeps ~40-50 brawlers per map/mode/tier
_SP_MIN_COUNTER_GAMES: int = 10

# Ban simulation defaults
_SP_BAN_TOP_N: int = 15
"""
Number of top-pick-rate brawlers eligible for ban sampling.

Using top-15 captures roughly the S/A-tier meta brawlers — the ones that
dominate draft if never banned.  Brawlers outside the top-15 are never
banned by the sampler, preserving them as normal picks.
"""

_SP_BAN_P: float = 0.35
"""
Per-brawler probability that each top-N candidate is banned in a given game.

At 0.35, each S-tier brawler is absent from ~35% of games — enough to force
the policy to learn non-dominant picks without erasing dominant brawlers from
training.  Expected bans per game ≈ 0.35 × (top-N brawlers drawn) before the
max_bans cap.
"""

_SP_MAX_BANS: int = 6
"""
Hard cap on bans per game.

Prevents pathological games where too many strong brawlers are absent at once.
With p_ban=0.35 and top_n=15, the expected ban count is ~5.3 before capping,
so the cap only activates in the tail of the distribution.
"""


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SelfPlayRecord:
    """
    One pick event from a self-play game, used as a training example.

    Fields
    ------
    state       : DraftState *before* this pick, from the picker's perspective.
                  state.whose_turn is always "mine" — picks are always stored
                  from the acting agent's point of view.
    pick_made   : brawler chosen at this pick slot.
    visit_dist  : {brawler: fraction} for the top-N most-visited MCTS children.
                  Fractions sum to ≤1.0 (top-N, not all brawlers). This is the
                  soft policy training target (Part 4.2).
    terminal_win_prob : FM win probability at the final terminal state, from the
                  acting team's perspective. Value network training target (Part 5).
    game_id     : unique sequential game identifier for grouping records.
    pick_number : 0-5, the pick's position within the 6-pick draft.
    """

    state: DraftState
    pick_made: str
    visit_dist: dict[str, float]
    terminal_win_prob: float
    game_id: int
    pick_number: int
    is_primary_team_pick: bool = True  # True = primary team's pick; False = opponent's mirrored pick


class ReplayBuffer:
    """
    Bounded ring buffer of complete self-play games.

    Stores up to `max_games` complete games, each a list[SelfPlayRecord].
    When the buffer is full, the oldest game is automatically evicted (deque
    semantics). Games are persisted as batch pickle files indexed by
    manifest.json in save_dir.

    Usage
    -----
        buf = ReplayBuffer(save_dir, max_games=10_000)
        new_games = run_self_play_batch(...)
        buf.add_games(new_games)
        buf.save_batch(new_games)         # persist to disk
        records = buf.all_records()       # flat list for policy training
    """

    def __init__(self, save_dir: Path, max_games: int = DEFAULT_MAX_GAMES) -> None:
        self.save_dir = save_dir
        self.max_games = max_games
        self._games: deque[list[SelfPlayRecord]] = deque(maxlen=max_games)
        self._batch_idx: int = 0  # monotonically increasing batch counter

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_games(self) -> int:
        """Number of complete games currently in the buffer."""
        return len(self._games)

    @property
    def n_records(self) -> int:
        """Total SelfPlayRecord count across all buffered games."""
        return sum(len(g) for g in self._games)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_games(self, games: list[list[SelfPlayRecord]]) -> None:
        """Append complete games to the buffer; oldest evicted when full."""
        for game in games:
            self._games.append(game)

    # ── Access ────────────────────────────────────────────────────────────────

    def all_records(self) -> list[SelfPlayRecord]:
        """Flat list of every SelfPlayRecord across all buffered games."""
        return [r for game in self._games for r in game]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_batch(self, games: list[list[SelfPlayRecord]]) -> Path:
        """
        Pickle a batch of games to save_dir/batch_NNNNN.pkl and update
        the manifest. Returns the saved file path.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"batch_{self._batch_idx:05d}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(games, fh, protocol=pickle.HIGHEST_PROTOCOL)
        _append_manifest(self.save_dir, path, sum(len(g) for g in games))
        self._batch_idx += 1
        return path

    @classmethod
    def load(cls, save_dir: Path, max_games: int = DEFAULT_MAX_GAMES) -> "ReplayBuffer":
        """
        Reconstruct a ReplayBuffer from the most recent batch files on disk.

        Reads batches from newest to oldest until max_games games are loaded,
        then adds them chronologically so the deque is in the correct eviction
        order (oldest first).
        """
        buf = cls(save_dir, max_games)
        manifest_path = save_dir / "manifest.json"
        if not manifest_path.exists():
            return buf

        with open(manifest_path) as fh:
            manifest = json.load(fh)

        batches = manifest.get("batches", [])
        buf._batch_idx = len(batches)

        # Collect games newest-first until we have enough to fill the buffer.
        collected: list[list[SelfPlayRecord]] = []
        for entry in reversed(batches):
            bp = Path(entry["path"])
            if not bp.exists():
                continue
            with open(bp, "rb") as fh:
                collected[:0] = pickle.load(fh)   # prepend keeps chronological order
            if len(collected) >= max_games:
                break

        for game in collected[-max_games:]:
            buf._games.append(game)

        return buf


# ── Manifest helper ───────────────────────────────────────────────────────────

def _append_manifest(save_dir: Path, batch_path: Path, n_records: int) -> None:
    """Append one batch entry to save_dir/manifest.json (creates if absent)."""
    manifest_path = save_dir / "manifest.json"
    manifest: dict = {"batches": []}
    if manifest_path.exists():
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    manifest["batches"].append({"path": str(batch_path), "n_records": n_records})
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)


# ── Per-game helpers ──────────────────────────────────────────────────────────

# ── Ban sampling ──────────────────────────────────────────────────────────────

def sample_bans(
    vocab: tuple[str, ...],
    db: MatchupDB,
    mode: str,
    map_name: str,
    tier: int,
    rng: np.random.Generator,
    *,
    ban_top_n: int = _SP_BAN_TOP_N,
    ban_p: float = _SP_BAN_P,
    max_bans: int = _SP_MAX_BANS,
) -> frozenset[str]:
    """
    Stochastically sample a ban set for one self-play game.

    Selects ban candidates from the top-N brawlers by pick_rate on the current
    map/mode/tier.  Each candidate is banned independently with probability
    ban_p.  The result is capped at max_bans to prevent games with too few
    available brawlers.

    Dominant brawlers (Sirius, Bull, Crow, etc.) naturally rise to the top of
    the pick-rate ranking and are therefore most likely to be banned, while
    fringe brawlers are never banned by this sampler.  This prevents self-play
    from collapsing into "whoever picks the S-tier brawler wins" dynamics while
    keeping dominant brawlers present in the majority of games.

    Parameters
    ----------
    vocab       : full brawler vocabulary from the FM schema.
    db          : MatchupDB for pick_rate lookups.
    mode        : game mode string.
    map_name    : map name string.
    tier        : skill tier integer.
    rng         : NumPy random generator (caller's; mutated in place).
    ban_top_n   : how many top-pick-rate brawlers are eligible for banning
                  (default 15).
    ban_p       : per-brawler ban probability for each eligible candidate
                  (default 0.35).
    max_bans    : hard cap on total bans returned (default 6).

    Returns
    -------
    frozenset of banned brawler names (may be empty).
    """
    if ban_p <= 0.0 or ban_top_n <= 0 or max_bans <= 0:
        return frozenset()

    # Rank vocab by pick_rate descending; brawlers with no data get rate=0.
    rates: list[tuple[float, str]] = []
    for b in vocab:
        entry = db.brawler_lookup(b, mode, map_name, tier)
        rate = entry["pick_rate"] if entry is not None else 0.0
        rates.append((rate, b))
    rates.sort(reverse=True)

    candidates = [b for _, b in rates[:ban_top_n]]
    if not candidates:
        return frozenset()

    # Each candidate is banned independently with probability ban_p.
    banned = [b for b in candidates if rng.random() < ban_p]

    # Cap and return.
    if len(banned) > max_bans:
        banned = list(rng.choice(banned, size=max_bans, replace=False))
    return frozenset(banned)


def _build_tree_vocab(
    vocab: tuple[str, ...],
    db: MatchupDB,
    mode: str,
    map_name: str,
    tier: int,
    min_pick_rate: float,
) -> tuple[str, ...]:
    """
    Return the subset of vocab whose pick_rate on (map_name, mode, tier) meets
    min_pick_rate. Falls back to the full vocab if all brawlers would be filtered.

    Built once per game and reused across all 6 _run_mcts calls.
    """
    if min_pick_rate <= 0.0 or db.skill_tier_boundaries is None:
        return vocab
    filtered = [
        b for b in vocab
        if (entry := db.brawler_lookup(b, mode, map_name, tier)) is None
        or entry["pick_rate"] >= min_pick_rate
    ]
    return tuple(filtered) if filtered else vocab


def _extract_visit_dist(root, top_n: int) -> dict[str, float]:
    """
    Return normalised visit fractions for the top-N most-visited root children.

    Only includes children with at least one visit. Fractions sum to ≤1.0
    (top-N subset, not all available brawlers).
    """
    if root._all_actions is None:
        return {}
    n = len(root._all_actions)
    visits = root._child_visits[:n]
    total = float(visits.sum())
    if total <= 0.0:
        return {}
    top_idx = np.argsort(-visits)[:top_n]
    return {
        root._all_actions[i]: float(visits[i] / total)
        for i in top_idx
        if visits[i] > 0
    }


def _sample_pick(root, temperature: float, rng: np.random.Generator) -> str:
    """
    Sample the next brawler pick from the MCTS root's visit distribution.

    temperature=1.0 : proportional to visit counts (standard for training).
    temperature→0   : always choose the argmax (greedy / deterministic).

    Note: temperature controls the pick selection only. The visit_dist stored
    in SelfPlayRecord always uses raw visit fractions regardless of temperature.
    """
    if root._all_actions is None:
        raise RuntimeError("Root node has not been expanded")
    n = len(root._all_actions)
    visits = root._child_visits[:n]
    visited_mask = visits > 0

    if not visited_mask.any():
        # Pathological: no sims reached any child. Uniform fallback.
        return root._all_actions[int(rng.integers(n))]

    v = visits[visited_mask].astype(np.float64)
    brawlers = [root._all_actions[i] for i in range(n) if visits[i] > 0]

    if temperature <= 0.0:
        return brawlers[int(np.argmax(v))]

    if temperature == 1.0:
        probs = v / v.sum()
    else:
        log_p = np.log(v) / temperature
        log_p -= log_p.max()        # numerical stability
        probs = np.exp(log_p)
        probs /= probs.sum()

    return brawlers[int(rng.choice(len(brawlers), p=probs))]


# ── Single-game runner ────────────────────────────────────────────────────────

def run_self_play_game(
    evaluator: FMEvaluator,
    db: MatchupDB,
    game_idx: int,
    map_mode_pairs: list[tuple[str, str]],
    *,
    n_sims_per_pick: int = DEFAULT_N_SIMS_PER_PICK,
    skill_ns_choices: tuple[float, ...] = DEFAULT_SKILL_NS_CHOICES,
    opp_rollout_mode: str = _SP_OPP_ROLLOUT_MODE,
    opp_top_k: int = _SP_OPP_TOP_K,
    min_pick_rate: float = _SP_MIN_PICK_RATE,
    min_counter_games: int = _SP_MIN_COUNTER_GAMES,
    visit_dist_top_n: int = DEFAULT_VISIT_DIST_TOP_N,
    pick_temperature: float = DEFAULT_PICK_TEMPERATURE,
    rng: Optional[np.random.Generator] = None,
    policy: Optional["PolicyInference"] = None,
    value_net: Optional["JointNetInference"] = None,
    ban_top_n: int = _SP_BAN_TOP_N,
    ban_p: float = _SP_BAN_P,
    max_bans: int = _SP_MAX_BANS,
) -> list[SelfPlayRecord]:
    """
    Run one complete 6-pick self-play draft and return 6 SelfPlayRecords.

    Two symmetric MCTS agents (same FM + DB) alternate picks. At each slot,
    the acting agent runs _run_mcts from its own perspective:
      - Primary team picks  : use canonical game_state directly.
      - Opposing team picks : mirror state (swap teams, flip is_first_pick)
                              so whose_turn == "mine" for the acting agent.

    P1/P2 assignment alternates by game_idx (even = primary is P1) to ensure
    a 50/50 training split across the draft-order perspectives.

    Parameters
    ----------
    evaluator        : pre-loaded FMEvaluator.
    db               : pre-loaded MatchupDB.
    game_idx         : sequential game index. Determines P1/P2 assignment and
                       is embedded in every SelfPlayRecord for grouping.
    map_mode_pairs   : list of valid (map_name, mode) tuples. Obtain via
                       load_map_mode_pairs(sqlite_path).
    n_sims_per_pick  : MCTS simulation budget per pick (default 2000).
    skill_ns_choices : pool of skill_ns values sampled uniformly (default
                       (1.0, 2.0, 3.0) = casual / mid / elite).
    opp_rollout_mode : rollout policy for opponent picks during simulation
                       (default "top_k" = adversarial but not deterministic).
    opp_top_k        : k for top_k opponent rollout (default 3).
    min_pick_rate    : brawlers below this pick rate on the map/mode/tier are
                       excluded from the MCTS tree, reducing branching factor
                       from ~90 to ~40-50. Full vocab is still used by rollout.
    min_counter_games: minimum game count required to trust counter_rate.
    visit_dist_top_n : how many top children to store in visit_dist.
    pick_temperature : sampling temperature. 1.0 = proportional, 0 = greedy.
    rng              : NumPy generator (created fresh if None).
    policy           : optional trained PolicyInference (or JointNetInference) used
                       as PUCT prior in _run_mcts. When None, falls back to counter-rate prior.
    value_net        : optional JointNetInference for rollout-free MCTS. When provided,
                       skips rollout() and calls value_net.evaluate() at every leaf.
                       Pass the same JointNetInference as both policy and value_net.
    ban_top_n        : number of top-pick-rate brawlers eligible for ban sampling
                       (default 15). Set to 0 to disable ban simulation.
    ban_p            : per-brawler ban probability for each ban candidate
                       (default 0.35).
    max_bans         : hard cap on total bans per game (default 6).

    Returns
    -------
    list of 6 SelfPlayRecord, one per pick slot.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not map_mode_pairs:
        raise ValueError("map_mode_pairs must not be empty")

    vocab = evaluator._fm.schema.vocab

    # ── Sample game parameters ─────────────────────────────────────────────
    map_name, mode = map_mode_pairs[int(rng.integers(len(map_mode_pairs)))]
    skill_ns = float(skill_ns_choices[int(rng.integers(len(skill_ns_choices)))])
    # Alternating P1/P2 guarantees exact 50/50 balance across the batch.
    is_first_pick = (game_idx % 2 == 0)

    tier = (
        skill_ns_to_tier(skill_ns, db.skill_tier_boundaries)
        if db.skill_tier_boundaries is not None
        else 1
    )

    # ── Sample bans for this game ──────────────────────────────────────────
    # Dominant brawlers (high pick_rate) are banned with probability ban_p,
    # capped at max_bans.  This prevents training from collapsing into
    # "first to pick S-tier wins" dynamics while keeping them in most games.
    bans = sample_bans(
        vocab, db, mode, map_name, tier, rng,
        ban_top_n=ban_top_n, ban_p=ban_p, max_bans=max_bans,
    )

    # ── Per-game resources built once and reused across all 6 picks ────────
    # tree_vocab: prunes low pick-rate brawlers from the MCTS tree.
    # weight_cache: pre-builds the V×V counter matrix (O(V²), ~10-50 ms).
    tree_vocab = _build_tree_vocab(vocab, db, mode, map_name, tier, min_pick_rate)
    weight_cache = RolloutWeightCache(db, vocab, min_counter_games=min_counter_games)
    weight_cache._get(mode, map_name, tier)    # pay the build cost now, once

    # ── Canonical game state from the primary team's perspective ───────────
    game_state = DraftState(
        my_team=frozenset(),
        opp_team=frozenset(),
        mode=mode,
        map_name=map_name,
        skill_ns=skill_ns,
        is_first_pick=is_first_pick,
        bans=bans,
    )

    # Collect (mcts_state, pick, visit_dist, is_primary_team_pick) per slot.
    # terminal_win_prob is filled in after the game ends.
    _picks: list[tuple[DraftState, str, dict[str, float], bool]] = []

    for _ in range(6):
        current_turn = game_state.whose_turn   # "mine" or "opp"

        if current_turn == "mine":
            mcts_state = game_state
        else:
            # Mirror: the opposing agent evaluates from their own perspective.
            # Swapping teams and flipping is_first_pick makes whose_turn=="mine"
            # at the same pick_number, so _run_mcts optimises for their team.
            mcts_state = DraftState(
                my_team=game_state.opp_team,
                opp_team=game_state.my_team,
                mode=mode,
                map_name=map_name,
                skill_ns=skill_ns,
                is_first_pick=not is_first_pick,
                bans=bans,
            )

        root = _run_mcts(
            state=mcts_state,
            evaluator=evaluator,
            db=db,
            vocab=vocab,
            n_simulations=n_sims_per_pick,
            my_counter_weight=DEFAULT_MY_COUNTER_WEIGHT,
            opp_counter_weight=DEFAULT_OPP_COUNTER_WEIGHT,
            min_pick_rate=min_pick_rate,
            opp_rollout_mode=opp_rollout_mode,
            opp_top_k=opp_top_k,
            my_rollout_mode="weighted",
            my_top_k=3,
            min_counter_games=min_counter_games,
            c=UCB1_C,
            rng=rng,
            puct_alpha=0.7,
            tree_vocab=tree_vocab,
            weight_cache=weight_cache,
            policy=policy,
            value_net=value_net,
        )

        visit_dist = _extract_visit_dist(root, visit_dist_top_n)
        pick = _sample_pick(root, pick_temperature, rng)
        _picks.append((mcts_state, pick, visit_dist, current_turn == "mine"))

        game_state = apply_pick(game_state, pick)

    # ── Terminal evaluation ─────────────────────────────────────────────────
    # game_state is now terminal from the primary team's perspective.
    p1_win_prob = evaluator.evaluate(game_state)

    records: list[SelfPlayRecord] = []
    for pick_number, (mcts_state, pick, visit_dist, is_primary) in enumerate(_picks):
        # Each team's records use win prob from their own perspective so that
        # the value target is always "my team's probability of winning".
        win_prob = p1_win_prob if is_primary else (1.0 - p1_win_prob)
        records.append(SelfPlayRecord(
            state=mcts_state,
            pick_made=pick,
            visit_dist=visit_dist,
            terminal_win_prob=win_prob,
            game_id=game_idx,
            pick_number=pick_number,
            is_primary_team_pick=is_primary,
        ))
    return records


# ── Map/mode helper ───────────────────────────────────────────────────────────

def load_map_mode_pairs(
    sqlite_path: Path,
    known_maps: set[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Query the season's SQLite database for all unique (map_name, mode) pairs.

    The MatchupDB does not store the map→mode mapping (it's implicit in the
    data), so this function reads it directly from the source database.

    Parameters
    ----------
    sqlite_path : path to the season's SQLite file, e.g.
                  REPO_ROOT / "season48" / "v1_clean.db"
    known_maps  : if provided, only return pairs whose map name appears in
                  this set. Pass ``set(schema.maps)`` to restrict self-play
                  to maps the FM was actually trained on (avoids KeyError for
                  maps with too few ELO-filtered games to enter the schema).

    Returns
    -------
    Sorted list of (map_name, mode) tuples. Pass directly to
    run_self_play_game or run_self_play_batch.

    Example
    -------
        from data_prep import SEASON_CONFIGS
        pairs = load_map_mode_pairs(SEASON_CONFIGS["s48"]["db_path"])
        games = run_self_play_batch(100, data_dir, pairs)
    """
    import sqlite3
    with sqlite3.connect(sqlite_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT map, mode FROM matches ORDER BY map"
        ).fetchall()
    if not rows:
        raise RuntimeError(f"No (map, mode) pairs found in {sqlite_path}")
    pairs = [(str(r[0]), str(r[1])) for r in rows]
    if known_maps is not None:
        dropped = {m for m, _ in pairs if m not in known_maps}
        if dropped:
            print(f"load_map_mode_pairs: skipping {len(dropped)} map(s) "
                  f"not in FM schema: {sorted(dropped)}")
        pairs = [(m, mode) for m, mode in pairs if m in known_maps]
    if not pairs:
        raise RuntimeError(
            f"No (map, mode) pairs remain after filtering by known_maps in {sqlite_path}"
        )
    return pairs


# ── Multiprocessing support ───────────────────────────────────────────────────

# Process-local globals set by _worker_init. Using globals avoids re-pickling
# large objects (FM weights, matchup matrices, policy tensors) for every task.
_g_evaluator: Optional[FMEvaluator] = None
_g_db: Optional[MatchupDB] = None
_g_policy: Optional["PolicyInference"] = None
_g_value_net: Optional["JointNetInference"] = None


def _worker_init(data_dir: Path) -> None:
    """Load FM, DB, and optional joint net / policy into process-local globals.

    Prefers joint_net.pkl (policy + value heads) over policy_best.pkl
    (policy only). When joint_net.pkl exists, the same JointNetInference
    object serves as both the PUCT prior (policy) and the rollout-free
    value estimator (value_net).
    """
    global _g_evaluator, _g_db, _g_policy, _g_value_net
    _g_evaluator = FMEvaluator(FMInference.load(data_dir / "fm_model.pkl"))
    _g_db = MatchupDB.load(data_dir / "matchup_db.pkl")

    joint_path = data_dir / "joint_net.pkl"
    policy_path = data_dir / "policy" / "policy_best.pkl"

    if joint_path.exists():
        joint = JointNetInference.load(joint_path)
        _g_policy = joint       # implements predict_prior interface
        _g_value_net = joint    # implements evaluate interface
    elif policy_path.exists():
        _g_policy = PolicyInference.load(policy_path)
        _g_value_net = None
    else:
        _g_policy = None
        _g_value_net = None


def _worker_run_game(args: tuple) -> list[SelfPlayRecord]:
    """Worker entry point: run one game using process-local FM + DB + policy."""
    game_idx, map_mode_pairs, seed, kwargs = args
    return run_self_play_game(
        _g_evaluator, _g_db, game_idx, map_mode_pairs,
        rng=np.random.default_rng(seed),
        policy=_g_policy,
        value_net=_g_value_net,
        **kwargs,
    )


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_self_play_batch(
    n_games: int,
    data_dir: Path,
    map_mode_pairs: list[tuple[str, str]],
    replay_buffer: Optional[ReplayBuffer] = None,
    *,
    evaluator: Optional[FMEvaluator] = None,
    db: Optional[MatchupDB] = None,
    n_workers: int = 1,
    start_game_idx: int = 0,
    seed: Optional[int] = None,
    n_sims_per_pick: int = DEFAULT_N_SIMS_PER_PICK,
    skill_ns_choices: tuple[float, ...] = DEFAULT_SKILL_NS_CHOICES,
    opp_rollout_mode: str = _SP_OPP_ROLLOUT_MODE,
    opp_top_k: int = _SP_OPP_TOP_K,
    min_pick_rate: float = _SP_MIN_PICK_RATE,
    min_counter_games: int = _SP_MIN_COUNTER_GAMES,
    visit_dist_top_n: int = DEFAULT_VISIT_DIST_TOP_N,
    pick_temperature: float = DEFAULT_PICK_TEMPERATURE,
    policy: Optional["PolicyInference"] = None,
    value_net: Optional["JointNetInference"] = None,
    ban_top_n: int = _SP_BAN_TOP_N,
    ban_p: float = _SP_BAN_P,
    max_bans: int = _SP_MAX_BANS,
) -> list[list[SelfPlayRecord]]:
    """
    Generate `n_games` self-play games, optionally in parallel.

    When replay_buffer is provided, games are added to the in-memory buffer
    and saved to disk as a batch pickle (buffer.save_dir / batch_NNNNN.pkl).

    Parameters
    ----------
    n_games        : number of complete games to generate.
    data_dir       : directory containing fm_model.pkl and matchup_db.pkl.
    map_mode_pairs : valid (map_name, mode) pairs from load_map_mode_pairs().
    replay_buffer  : optional ReplayBuffer; games are added and saved if given.
    evaluator      : pre-loaded FMEvaluator (serial path only; ignored when
                     n_workers > 1, where workers load from data_dir).
    db             : pre-loaded MatchupDB (serial path only).
    n_workers      : number of parallel worker processes.
                     1 = serial (easier to debug, uses evaluator/db if given).
                     >1 = multiprocessing; each worker loads FM+DB from data_dir.
    start_game_idx : game index offset for continuing from a previous batch.
                     Ensures game_id is globally unique and P1/P2 alternation
                     continues correctly across batch calls.
    seed           : root RNG seed for reproducible batches (None = random).
    n_sims_per_pick, skill_ns_choices, opp_rollout_mode, opp_top_k,
    min_pick_rate, min_counter_games, visit_dist_top_n, pick_temperature :
                     forwarded to run_self_play_game.
    ban_top_n      : top-pick-rate brawlers eligible for ban sampling (default 15).
                     Set to 0 to disable ban simulation entirely.
    ban_p          : per-brawler ban probability for ban candidates (default 0.35).
    max_bans       : hard cap on total bans per game (default 6).

    Returns
    -------
    list of list[SelfPlayRecord], one inner list per game (6 records each).

    Example
    -------
        from data_prep import SEASON_CONFIGS
        from pathlib import Path

        data_dir = Path("data/s48")
        pairs    = load_map_mode_pairs(SEASON_CONFIGS["s48"]["db_path"])
        buf      = ReplayBuffer(data_dir / "self_play")

        games = run_self_play_batch(
            n_games=200, data_dir=data_dir, map_mode_pairs=pairs,
            replay_buffer=buf, n_workers=4, seed=42,
        )
        print(f"{buf.n_games} games  {buf.n_records} records in buffer")
    """
    if not map_mode_pairs:
        raise ValueError("map_mode_pairs must not be empty")

    game_kwargs: dict = dict(
        n_sims_per_pick=n_sims_per_pick,
        skill_ns_choices=skill_ns_choices,
        opp_rollout_mode=opp_rollout_mode,
        opp_top_k=opp_top_k,
        min_pick_rate=min_pick_rate,
        min_counter_games=min_counter_games,
        visit_dist_top_n=visit_dist_top_n,
        pick_temperature=pick_temperature,
        ban_top_n=ban_top_n,
        ban_p=ban_p,
        max_bans=max_bans,
        policy=policy,
        value_net=value_net,
    )

    # Independent per-game seeds derived from a root seed for reproducibility.
    ss = np.random.SeedSequence(seed)
    child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_games)]

    if n_workers <= 1:
        # ── Serial path ───────────────────────────────────────────────────
        # Reuse pre-loaded resources if provided; otherwise load from disk.
        if evaluator is None:
            evaluator = FMEvaluator(FMInference.load(data_dir / "fm_model.pkl"))
        if db is None:
            db = MatchupDB.load(data_dir / "matchup_db.pkl")
        games = [
            run_self_play_game(
                evaluator, db,
                game_idx=start_game_idx + i,
                map_mode_pairs=map_mode_pairs,
                rng=np.random.default_rng(child_seeds[i]),
                **game_kwargs,
            )
            for i in range(n_games)
        ]
    else:
        # ── Parallel path ─────────────────────────────────────────────────
        # Each worker loads FM + DB + policy independently via _worker_init.
        # Policy and value_net are NOT sent through the task queue (PyTorch tensors
        # can't be pickled via ForkingPickler on macOS spawn). Workers load
        # joint_net.pkl or policy_best.pkl from data_dir in _worker_init instead.
        parallel_kwargs = {k: v for k, v in game_kwargs.items()
                          if k not in ("policy", "value_net")}
        task_args = [
            (start_game_idx + i, map_mode_pairs, child_seeds[i], parallel_kwargs)
            for i in range(n_games)
        ]
        with Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(data_dir,),
        ) as pool:
            games = pool.map(_worker_run_game, task_args)

    if replay_buffer is not None:
        replay_buffer.add_games(games)
        saved_path = replay_buffer.save_batch(games)
        n_records = sum(len(g) for g in games)
        print(
            f"Batch saved: {n_games} games  ({n_records} records)"
            f"  →  {saved_path.name}"
            f"  |  buffer: {replay_buffer.n_games} games total"
        )

    return games


# ── Part 5.1.1 — Buffer coverage verification ──────────────────────────────────

def verify_buffer_coverage(records: list[SelfPlayRecord]) -> dict:
    """
    Verify replay buffer coverage across all pick depths for value network training.

    Checks that:
    - Every pick depth 0–5 has at least 100 records (minimum for stable training).
    - All ``terminal_win_prob`` values are valid (not NaN, in [0, 1]).

    Returns a summary dict; the ``'pass'`` key is True when coverage is adequate.
    Call this before training the joint policy+value network (Part 5.2).

    Parameters
    ----------
    records : flat list of SelfPlayRecord from ``ReplayBuffer.all_records()``.
    """
    import math
    from collections import Counter

    by_depth: Counter = Counter(r.pick_number for r in records)

    win_probs    = [r.terminal_win_prob for r in records]
    nan_count    = sum(1 for p in win_probs if math.isnan(p))
    out_of_range = sum(1 for p in win_probs if not math.isnan(p) and not (0.0 <= p <= 1.0))

    min_depth_count    = min(by_depth.get(d, 0) for d in range(6))
    all_depths_covered = all(by_depth.get(d, 0) >= 100 for d in range(6))

    result = {
        "n_records":            len(records),
        "by_pick_depth":        {d: by_depth.get(d, 0) for d in range(6)},
        "min_depth_count":      min_depth_count,
        "nan_win_probs":        nan_count,
        "out_of_range_win_probs": out_of_range,
        "pass": all_depths_covered and nan_count == 0 and out_of_range == 0,
    }

    status = "PASS" if result["pass"] else "FAIL"
    print(f"Buffer coverage [{status}]  ({result['n_records']:,} records):")
    for d in range(6):
        print(f"  pick depth {d}: {by_depth.get(d, 0):,}")
    if nan_count or out_of_range:
        print(f"  WARNING: {nan_count} NaN win_probs, {out_of_range} out-of-range win_probs")

    return result


# ── Part 5.1.2 — Team-swap augmentation ───────────────────────────────────────

def augment_with_team_swap(records: list[SelfPlayRecord]) -> list[SelfPlayRecord]:
    """
    Generate team-swapped augmented copies of self-play records for value head
    training in the joint policy+value network (Part 5).

    For each record, produces a copy where ``my_team`` and ``opp_team`` are
    swapped, ``is_first_pick`` is flipped (so ``pick_number`` encodes correctly
    for the swapped team's pick-order perspective), and
    ``terminal_win_prob = 1 - original`` (the opponent's win probability).

    The ``visit_dist`` is cleared to ``{}`` because it was computed for the
    original picker and is invalid for the swapped team. Augmented records
    therefore contribute **only** to the value head loss. The joint network
    trainer must zero-weight the policy head loss for any record with an empty
    ``visit_dist``.

    Note: augmented records may have ``state.whose_turn == "opp"`` — the pick
    slot that belonged to "mine" for the original team may belong to "opp" after
    the swap. This is correct; the value head must learn from both turn values.

    Returns only the augmented records (not the originals). Combine as::

        all_value_records = records + augment_with_team_swap(records)

    Parameters
    ----------
    records : original self-play records (from ``ReplayBuffer.all_records()``).
    """
    augmented: list[SelfPlayRecord] = []
    for rec in records:
        swapped_state = DraftState(
            my_team=rec.state.opp_team,
            opp_team=rec.state.my_team,
            mode=rec.state.mode,
            map_name=rec.state.map_name,
            skill_ns=rec.state.skill_ns,
            is_first_pick=not rec.state.is_first_pick,
            bans=rec.state.bans,
        )
        augmented.append(SelfPlayRecord(
            state=swapped_state,
            pick_made="",   # not meaningful for augmented records
            visit_dist={},  # cleared — invalid from the swapped team's perspective
            terminal_win_prob=1.0 - rec.terminal_win_prob,
            game_id=rec.game_id,
            pick_number=rec.pick_number,
            is_primary_team_pick=not rec.is_primary_team_pick,
        ))
    return augmented
