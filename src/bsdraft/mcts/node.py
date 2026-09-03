"""
mcts_node.py — Steps 2.2, 2.5, & 3.2: MCTS Tree Structure + Backpropagation + PUCT

Provides:
  MCTSNode        — one node in the MCTS tree
  UCB1_C          — default exploration constant (see tuning note below)
  ucb1_score      — UCB1 or PUCT formula for one child (supports exploitation flip)
  select_child    — select the best child of a node (turn-aware, UCB1 or PUCT)
  select          — traverse root → leaf via greedy UCB1/PUCT
  backpropagate   — walk leaf → root, updating visit counts and value sums

Value convention (root perspective):
  Every node stores value_sum from MY perspective — the root's perspective.
  value_sum always accumulates win_prob directly (never 1 − win_prob).
  Q(node) = value_sum / visit_count = my expected win rate through this node.

  Adversarial selection is handled in select_child:
    • my-turn node selecting a child: maximise Q + exploration (I want high win rate)
    • opp-turn node selecting a child: maximise (1−Q) + exploration
      (opponent wants high opp win rate = low Q from my perspective)

  Why not "current-player perspective + maximise everywhere"?
  In the pick order [mine, opp, opp, mine, mine, opp] a my-turn parent selects
  among opp-turn children.  If those children stored Q from the opponent's
  perspective, maximising their Q would pick the state where the opponent wins
  most — which is wrong for my-turn selection.  Root perspective avoids this
  by keeping Q uniformly interpretable as my win probability.

PUCT at opponent-turn nodes (Step 3.2):
  When expand() is called with prior_weights, the node stores _child_priors and
  select_child switches to the PUCT formula for that node:

    PUCT(child) = exploit + C * prior(child) * sqrt(N_parent) / (1 + N_child)

  where exploit = (1−Q) for opp-turn nodes (adversarial) or Q for my-turn nodes.
  This concentrates early exploration on the opponent's most likely counter-picks.

  For my-turn nodes, prior_weights is None → UCB1 is used unchanged.

UCB1 constant C (see also Step 2.8.1):
  The default UCB1_C = 0.5 is lower than the theoretical sqrt(2) ≈ 1.41
  because this game has only 6 picks (shallow tree) and Q values are
  win probabilities bounded in (0, 1).  A smaller C lets visit counts
  concentrate on the best pick more quickly.  Tune C in Step 2.8.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bsdraft.mcts.state import DraftState, apply_pick, available_actions

# ── Exploration constant ───────────────────────────────────────────────────────

UCB1_C: float = 0.5
"""
Default UCB1 exploration constant.

Chosen lower than sqrt(2) ≈ 1.41 because:
  - Only 6 picks deep → shallow tree, exploitation bias is appropriate.
  - Q values are win probabilities in (0, 1) → tighter reward range than the
    unbounded rewards UCB1 was originally analysed for.

To use a different value, pass `c=...` to select() or select_child().

See Step 2.8.1 in IMPLEMENTATION_CHECKLIST.md for the planned C-tuning sweep.
"""


# ── Node ──────────────────────────────────────────────────────────────────────

@dataclass
class MCTSNode:
    """
    One node in the MCTS tree.

    Parameters
    ----------
    state       : DraftState snapshot this node represents.
    parent      : Parent node; None for the root.
    children    : Mapping from brawler name (action) to child MCTSNode.
                  Only contains children that have been lazily created so far.
                  Empty until the first child is selected; grows one entry at a
                  time as select_child() is called.
    visit_count : Number of times this node has been visited (N).
    value_sum   : Cumulative value from the picker's perspective (W).
                  Q = value_sum / visit_count.
    prior       : Optional heuristic prior probability for this node's action,
                  set by the expansion policy.  Currently unused (uniform = 0.0);
                  reserved for a future PUCT extension.

    Lazy expansion (Step 2.8.1 optimization):
      expand() stores all available action strings in _all_actions and
      pre-allocates numpy arrays, but does NOT create any MCTSNode children.
      select_child() creates each child DraftState + MCTSNode on first selection,
      one at a time.  This drops apply_pick() calls from ~96k to ~2k per 1,000
      simulations (48× reduction in the dominant bottleneck) because most
      children of a node are never visited within a fixed simulation budget.
    """

    state: DraftState
    parent: MCTSNode | None = field(default=None, repr=False)
    children: dict[str, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    # Lazy-expansion + vectorized UCB1/PUCT arrays.
    # _all_actions  : all legal actions at this node (set by expand(), None=leaf).
    # _child_list   : MCTSNode objects created so far; grows lazily in select_child().
    # _child_visits / _child_values : pre-allocated arrays of len(_all_actions);
    #   only [:len(_child_list)] are valid — the rest remain zero-filled.
    # _child_node_idx : id(child) → index in the arrays for O(1) backprop update.
    # _child_priors : normalized prior probability for each action (PUCT, Step 3.2).
    #   Set only at opponent-turn nodes; None at my-turn nodes → UCB1 used instead.
    #   Sorted descending during expand() so lazy creation visits high-prior children first.
    _all_actions: list | None = field(default=None, repr=False, compare=False)
    _child_list: list | None = field(default=None, repr=False, compare=False)
    _child_visits: np.ndarray | None = field(default=None, repr=False, compare=False)
    _child_values: np.ndarray | None = field(default=None, repr=False, compare=False)
    _child_node_idx: dict | None = field(default=None, repr=False, compare=False)
    _child_priors: np.ndarray | None = field(default=None, repr=False, compare=False)

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def q_value(self) -> float:
        """
        Mean value from the *picker's* perspective.

        Returns 0.0 for unvisited nodes — callers should check visit_count
        before interpreting Q as meaningful.  In UCB1, unvisited nodes receive
        +inf exploration bonus so they are selected before any Q comparison.
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_leaf(self) -> bool:
        """True if expand() has not been called yet (no available actions stored).

        After expand(), this is False even if no children have been created yet
        (lazy expansion: children are created one at a time in select_child()).
        Terminal nodes stay as leaves because expand() is a no-op for them.
        """
        return self._all_actions is None

    @property
    def n_legal_actions(self) -> int:
        """Total number of legal actions at this node.

        With lazy expansion, len(children) only counts created children; this
        gives the full count including uncreated ones.  Used by layer2_confidence
        to compute the correct random-baseline visit fraction.
        """
        if self._all_actions is not None:
            return len(self._all_actions)
        return len(self.children)  # fallback for manually constructed test nodes

    # ── Expansion ─────────────────────────────────────────────────────────────

    def expand(
        self,
        vocab: tuple[str, ...],
        *,
        prior_weights: np.ndarray | None = None,
    ) -> None:
        """
        Record all legal actions and pre-allocate UCB1/PUCT arrays, but do NOT
        create any child nodes yet.  Children are created lazily in select_child().

        Safe to call only once — calling again on an already-expanded node
        is a no-op (_all_actions is already set).

        Parameters
        ----------
        vocab         : full brawler vocabulary, e.g. fm.schema.vocab
        prior_weights : optional normalized prior probability vector (PUCT, Step 3.2).
                        When provided, actions are sorted in descending prior order so
                        lazy child creation naturally visits high-prior picks first.
                        Must have length == number of available actions.
                        Pass only for opponent-turn nodes; leave None for my-turn nodes
                        to preserve UCB1 behaviour unchanged.
        """
        if self._all_actions is not None:
            return  # already expanded
        if self.state.is_terminal:
            return  # terminal nodes stay as leaves

        actions = list(available_actions(self.state, vocab))
        if not actions:
            return  # guard: no legal actions (should not occur in a valid draft)

        n = len(actions)

        # If prior weights are supplied, sort actions by descending prior so that
        # lazy creation naturally explores high-prior (high counter-rate) children first.
        if prior_weights is not None and len(prior_weights) == n:
            order = np.argsort(-prior_weights)
            actions = [actions[i] for i in order]
            self._child_priors = prior_weights[order]

        self._all_actions = actions
        self._child_list = []
        self._child_visits = np.zeros(n, dtype=np.float64)
        self._child_values = np.zeros(n, dtype=np.float64)
        self._child_node_idx = {}


# ── UCB1 ──────────────────────────────────────────────────────────────────────

def ucb1_score(
    node: MCTSNode,
    parent_visits: int,
    c: float,
    *,
    flip: bool = False,
    prior: float | None = None,
) -> float:
    """
    UCB1 or PUCT score for `node` given its parent has been visited `parent_visits` times.

    UCB1 (prior=None):
        score = exploit + C * sqrt(log(N_parent) / N_node)
        Unvisited nodes return +inf so they are always selected before visited ones.

    PUCT (prior provided):
        score = exploit + C * prior * sqrt(N_parent) / (1 + N_node)
        Unvisited nodes get a finite score; high-prior nodes are explored before
        low-prior ones (natural ordering from expand()'s prior-sorted action list).

    where exploit = (1 − Q) if flip else Q.

    flip=True is used at opponent-turn parents: the exploitation term inverts
    Q so that maximising the score achieves adversarial selection.

    Parameters
    ----------
    node          : the child node being scored
    parent_visits : visit count of the parent (must be > 0 for UCB1)
    c             : exploration constant (default UCB1_C = 0.5)
    flip          : if True, use (1 − Q) as the exploitation term
    prior         : PUCT prior probability for this child (0–1).  When provided,
                    PUCT formula is used instead of UCB1.
    """
    exploit = (1.0 - node.q_value) if flip else node.q_value
    if prior is not None:
        # PUCT: finite score for all nodes, prior governs exploration order.
        return exploit + c * prior * math.sqrt(parent_visits) / (1.0 + node.visit_count)
    # UCB1: unvisited nodes have infinite priority.
    if node.visit_count == 0:
        return float("inf")
    return exploit + c * math.sqrt(math.log(parent_visits) / node.visit_count)


def select_child(node: MCTSNode, c: float = UCB1_C) -> tuple[str, MCTSNode]:
    """
    Select the child of `node` with the highest UCB1 score (from node's perspective).

    All Q values are stored from my (root's) perspective.  The exploitation
    term is flipped at opponent-turn nodes so that maximising UCB1 still
    achieves adversarial selection:

      my-turn   : maximise Q + C·√(logN/n)  — I want high win probability
      opp-turn  : maximise (1−Q) + C·√(logN/n)  — opponent wants high opp win rate

    Lazy child creation: if node was expanded via expand() and still has
    uncreated children, the next child is created here (one apply_pick call)
    rather than during expand() (which would eagerly create all ~90).  Unvisited
    (uncreated) children always have priority over visited ones (equivalent to
    UCB1 = +inf).

    Falls back to a Python loop for nodes whose children were set manually
    (used in unit tests that bypass expand()).

    Parameters
    ----------
    node : a non-leaf, non-terminal node (expand() must have been called)
    c    : exploration constant

    Returns
    -------
    (action, child_node) — the selected action string and its child node.

    Raises
    ------
    ValueError  if node is still a leaf (call expand() first).
    """
    flip = node.state.whose_turn == "opp"

    # ── Fast path: lazy creation + vectorized UCB1 (nodes expanded via expand()) ─
    if node._all_actions is not None:
        n_created = len(node._child_list)
        n_total = len(node._all_actions)

        # Uncreated children always have priority (equivalent to visit_count=0 = +inf).
        if n_created < n_total:
            action = node._all_actions[n_created]
            child_state = apply_pick(node.state, action)
            child = MCTSNode(state=child_state, parent=node)
            node.children[action] = child
            node._child_list.append(child)
            node._child_node_idx[id(child)] = n_created
            return action, child

        # All children created — vectorized UCB1 or PUCT over the created slice.
        visits = node._child_visits[:n_created]
        values = node._child_values[:n_created]

        if node._child_priors is not None:
            # PUCT path: Q + C * prior * sqrt(N_parent) / (1 + N_child)
            # Handles unvisited nodes naturally (Q=0, denominator=1); high-prior
            # children are already first in _all_actions due to expand() sorting.
            priors = node._child_priors[:n_created]
            q = np.where(visits > 0, values / visits, 0.0)
            exploit = (1.0 - q) if flip else q
            ucb = exploit + c * priors * math.sqrt(node.visit_count) / (1.0 + visits)
        else:
            # UCB1 path: unvisited children always have priority (+inf).
            unvisited = visits == 0.0
            if unvisited.any():
                idx = int(np.argmax(unvisited))
                return node._all_actions[idx], node._child_list[idx]
            log_n = math.log(node.visit_count)
            exploit = (1.0 - values / visits) if flip else (values / visits)
            ucb = exploit + c * np.sqrt(log_n / visits)

        idx = int(np.argmax(ucb))
        return node._all_actions[idx], node._child_list[idx]

    # ── Fallback: Python loop for manually constructed test nodes ──────────────
    if not node.children:
        raise ValueError("select_child called on a leaf node — call expand() first.")

    log_n: float | None = None
    best_score = -math.inf
    best_action: str | None = None
    best_child: MCTSNode | None = None

    for action, child in node.children.items():
        vc = child.visit_count
        if vc == 0:
            return action, child
        if log_n is None:
            log_n = math.log(node.visit_count)
        exploit = (1.0 - child.value_sum / vc) if flip else (child.value_sum / vc)
        score = exploit + c * math.sqrt(log_n / vc)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child  # type: ignore[return-value]


def select(root: MCTSNode, c: float = UCB1_C) -> list[MCTSNode]:
    """
    Traverse from `root` to a leaf node via greedy UCB1, returning the path.

    Stops at:
      (a) a leaf node (not yet expanded), or
      (b) a terminal state (no further picks possible).

    The returned path includes both the root and the leaf.

    Parameters
    ----------
    root : the root of the MCTS tree
    c    : exploration constant

    Returns
    -------
    List of MCTSNode from root to leaf (inclusive), length ≥ 1.
    """
    path: list[MCTSNode] = [root]
    node = root

    while not node.is_leaf and not node.state.is_terminal:
        _, node = select_child(node, c)
        path.append(node)

    return path


# ── Backpropagation ───────────────────────────────────────────────────────────

def backpropagate(path: list[MCTSNode], win_prob: float) -> None:
    """
    Walk the path from root to leaf (or leaf to root — order irrelevant),
    incrementing visit_count and adding win_prob to value_sum for every node.

    win_prob is always P(my_team wins) — from my perspective.  Because all
    nodes store Q from the root's (my) perspective, no sign flip is applied
    here.  The adversarial effect is handled entirely in select_child() via
    the flip parameter.

    Also mirrors each update into the parent's _child_visits / _child_values
    numpy arrays so that select_child()'s vectorized UCB1 path stays in sync
    with the scalar visit_count / value_sum on each node.

    Parameters
    ----------
    path     : list of MCTSNode returned by select() — root at [0], leaf at [-1]
    win_prob : P(my_team wins) ∈ (0, 1) returned by rollout() or evaluate()
    """
    for i, node in enumerate(path):
        node.visit_count += 1
        node.value_sum += win_prob
        # Mirror the update into the parent's child arrays for vectorized UCB1.
        if i > 0:
            parent = path[i - 1]
            if parent._child_node_idx is not None:
                idx = parent._child_node_idx.get(id(node))
                if idx is not None:
                    parent._child_visits[idx] += 1.0
                    parent._child_values[idx] += win_prob


# ── Sanity checks ─────────────────────────────────────────────────────────────

if __name__ == "__main__":


    print("=== Steps 2.2 & 2.5 Sanity Checks ===\n")

    # ── Tiny toy vocab (avoids loading FM weights) ────────────────────────────
    _VOCAB: tuple[str, ...] = tuple(f"B{i}" for i in range(10))

    # P1 order: mine=0, opp=1, opp=2, mine=3, mine=4, opp=5
    # Root is always my-turn (pick 0).
    _ROOT_STATE = DraftState(
        my_team=frozenset(),
        opp_team=frozenset(),
        mode="gemGrab",
        map_name="Double Swoosh",
        skill_ns=1.0,
        is_first_pick=True,
    )
    # Opp-turn state (pick 1 in P1 order: my_team has 1 brawler, opp has 0)
    _OPP_STATE = DraftState(
        my_team=frozenset({"B0"}),
        opp_team=frozenset(),
        mode="gemGrab",
        map_name="Double Swoosh",
        skill_ns=1.0,
        is_first_pick=True,
    )
    assert _ROOT_STATE.whose_turn == "mine"
    assert _OPP_STATE.whose_turn == "opp"

    # ── Check 1: UCB1 selects highest-Q child at low C (my-turn node) ─────────
    root = MCTSNode(state=_ROOT_STATE, visit_count=100)
    child_a = MCTSNode(state=_OPP_STATE, parent=root, visit_count=40, value_sum=32.0)  # Q=0.800
    child_b = MCTSNode(state=_OPP_STATE, parent=root, visit_count=10, value_sum=5.0)   # Q=0.500
    child_c = MCTSNode(state=_OPP_STATE, parent=root, visit_count=50, value_sum=27.5)  # Q=0.550
    root.children = {"A": child_a, "B": child_b, "C": child_c}

    # My-turn root: flip=False → maximize Q → A has Q=0.800, wins at low C
    scores_low_c = {a: ucb1_score(child, 100, c=0.1, flip=False) for a, child in root.children.items()}
    best_low_c, _ = select_child(root, c=0.1)
    print("Check 1: my-turn node, low C → selects highest-Q child (A, Q=0.800)")
    for a, s in sorted(scores_low_c.items(), key=lambda x: -x[1]):
        print(f"  {a}: Q={root.children[a].q_value:.3f}  UCB1={s:.4f}")
    print(f"  → selected: {best_low_c}  (expected A)")
    assert best_low_c == "A", f"Expected A, got {best_low_c}"
    print("  ✓\n")

    scores_high_c = {a: ucb1_score(child, 100, c=2.0, flip=False) for a, child in root.children.items()}
    best_high_c, _ = select_child(root, c=2.0)
    print("Check 1b: my-turn node, high C → exploration pushes toward least-visited (B)")
    for a, s in sorted(scores_high_c.items(), key=lambda x: -x[1]):
        print(f"  {a}: Q={root.children[a].q_value:.3f}  UCB1={s:.4f}")
    print(f"  → selected: {best_high_c}  (expected B at C=2.0)")
    assert best_high_c == "B", f"Expected B at high C, got {best_high_c}"
    print("  ✓\n")

    # ── Check 2: Adversarial flip — opp-turn node selects lowest-Q child ──────
    # Three children with Q = my win probability (root perspective).
    # Opponent wants the child with LOWEST Q (minimises my win rate).
    # With flip=True: maximize (1-Q) = minimize Q.
    opp_root = MCTSNode(state=_OPP_STATE, visit_count=100)
    # child_X: Q=0.700 → 1-Q=0.300 (worst for me, opponent likes this most)
    child_x = MCTSNode(state=_ROOT_STATE, parent=opp_root, visit_count=40, value_sum=28.0)
    # child_Y: Q=0.400 → 1-Q=0.600 (best for opponent, worst for me)
    child_y = MCTSNode(state=_ROOT_STATE, parent=opp_root, visit_count=40, value_sum=16.0)
    # child_Z: Q=0.550 → 1-Q=0.450 (middle)
    child_z = MCTSNode(state=_ROOT_STATE, parent=opp_root, visit_count=20, value_sum=11.0)
    opp_root.children = {"X": child_x, "Y": child_y, "Z": child_z}

    scores_flip = {a: ucb1_score(child, 100, c=0.1, flip=True) for a, child in opp_root.children.items()}
    best_opp, _ = select_child(opp_root, c=0.1)
    print("Check 2: opp-turn node (flip=True) → selects lowest-Q child (Y, Q=0.400 → worst for me)")
    for a, s in sorted(scores_flip.items(), key=lambda x: -x[1]):
        print(f"  {a}: Q={opp_root.children[a].q_value:.3f}  flip_UCB1={s:.4f}")
    print(f"  → selected: {best_opp}  (expected Y)")
    assert best_opp == "Y", f"Expected Y (lowest Q), got {best_opp}"
    print("  ✓\n")

    # ── Check 3: Unvisited child always selected first ─────────────────────────
    root2 = MCTSNode(state=_ROOT_STATE, visit_count=50)
    visited_child = MCTSNode(state=_OPP_STATE, parent=root2, visit_count=50, value_sum=49.0)
    unvisited_child = MCTSNode(state=_OPP_STATE, parent=root2, visit_count=0)
    root2.children = {"visited": visited_child, "unvisited": unvisited_child}

    best_unvisited, _ = select_child(root2)
    print("Check 3: unvisited child always selected before any visited child")
    print(f"  → selected: {best_unvisited}  (expected: unvisited)")
    assert best_unvisited == "unvisited"
    print("  ✓\n")

    # ── Check 4: expand() stores actions lazily (no children created yet) ────────
    # After lazy expand: is_leaf=False (has _all_actions), but children is empty.
    # select_child() creates the first child on demand (one apply_pick call).
    leaf = MCTSNode(state=_ROOT_STATE)
    assert leaf.is_leaf
    leaf.expand(_VOCAB)
    assert not leaf.is_leaf, "After expand(), is_leaf should be False"
    assert len(leaf._all_actions) == len(_VOCAB), f"Expected {len(_VOCAB)} actions, got {len(leaf._all_actions)}"
    assert len(leaf.children) == 0, "Lazy expand: children dict must be empty after expand()"

    # select_child() creates the first child lazily.
    first_action, first_child = select_child(leaf)
    assert first_child.state.pick_number == 1, f"Expected pick_number=1, got {first_child.state.pick_number}"
    assert first_child.parent is leaf
    assert len(leaf.children) == 1, "One child created by select_child()"
    assert leaf.children[first_action] is first_child
    print(f"Check 4: expand() stores {len(leaf._all_actions)} actions lazily; "
          f"select_child() creates first child on demand (pick_number={first_child.state.pick_number})  ✓\n")

    # ── Check 5: select() traverses to a leaf ────────────────────────────────
    # With lazy expansion, select() on the root creates one new child per call
    # (since all children start uncreated = +inf priority).
    root3 = MCTSNode(state=_ROOT_STATE, visit_count=10)
    root3.expand(_VOCAB)
    # select() calls select_child(root3), which lazily creates the first child.
    path = select(root3)
    assert path[0] is root3
    assert path[-1].is_leaf, "Newly created child must itself be a leaf (unexpanded)"
    assert len(path) == 2, f"Expected depth 2, got {len(path)}"
    assert len(root3.children) == 1, "Exactly one child created during select()"
    print(f"Check 5: select() returns path length {len(path)}, ends at a fresh leaf  ✓\n")

    # ── Check 6: backpropagate() increments counts and sums ───────────────────
    # Build a 3-node chain: root → child → grandchild (all fresh).
    bp_root = MCTSNode(state=_ROOT_STATE)
    bp_child = MCTSNode(state=_OPP_STATE, parent=bp_root)
    # grandchild state: pick 2 (opp), after 1 opp pick
    _PICK2_STATE = DraftState(
        my_team=frozenset({"B0"}),
        opp_team=frozenset({"B1"}),
        mode="gemGrab",
        map_name="Double Swoosh",
        skill_ns=1.0,
        is_first_pick=True,
    )
    bp_grandchild = MCTSNode(state=_PICK2_STATE, parent=bp_child)
    bp_path = [bp_root, bp_child, bp_grandchild]

    WIN_PROB = 0.72
    backpropagate(bp_path, WIN_PROB)

    print("Check 6: backpropagate() — all nodes updated with win_prob from my perspective")
    for label, node in [("root", bp_root), ("child", bp_child), ("grandchild", bp_grandchild)]:
        assert node.visit_count == 1, f"{label}: expected visit_count=1, got {node.visit_count}"
        assert abs(node.value_sum - WIN_PROB) < 1e-9, f"{label}: expected value_sum={WIN_PROB}"
        assert abs(node.q_value - WIN_PROB) < 1e-9, f"{label}: expected Q={WIN_PROB}"
        print(f"  {label:12s}: visit_count={node.visit_count}, value_sum={node.value_sum:.4f}, Q={node.q_value:.4f}")
    print("  ✓\n")

    # Run backprop a second time (different win_prob) to check accumulation.
    WIN_PROB_2 = 0.30
    backpropagate(bp_path, WIN_PROB_2)
    expected_q = (WIN_PROB + WIN_PROB_2) / 2
    assert abs(bp_root.q_value - expected_q) < 1e-9, f"Expected Q={expected_q:.4f}"
    print(f"Check 6b: after 2 backprops Q = ({WIN_PROB} + {WIN_PROB_2}) / 2 = {bp_root.q_value:.4f}  ✓\n")

    # ── Check 7: adversarial consistency — opp-turn child preferred correctly ──
    # After backprop, simulate that one child leads to high win_prob and another
    # to low win_prob.  My-turn root should prefer high; opp-turn root should
    # prefer low (i.e., pick the worst for me).
    adv_root_mine = MCTSNode(state=_ROOT_STATE, visit_count=200)
    adv_root_opp  = MCTSNode(state=_OPP_STATE,  visit_count=200)

    for parent in [adv_root_mine, adv_root_opp]:
        # good child: 100 sims, all win → Q = 0.9
        good = MCTSNode(state=_OPP_STATE, parent=parent, visit_count=100, value_sum=90.0)
        # bad  child: 100 sims, mostly loss → Q = 0.2
        bad  = MCTSNode(state=_OPP_STATE, parent=parent, visit_count=100, value_sum=20.0)
        parent.children = {"good": good, "bad": bad}

    # My-turn root (C=0.1 to suppress exploration) → should prefer "good" (Q=0.9)
    best_mine, _ = select_child(adv_root_mine, c=0.1)
    # Opp-turn root (C=0.1) → should prefer "bad" (Q=0.2 → 1-Q=0.8 is highest)
    best_opp2, _ = select_child(adv_root_opp, c=0.1)

    print(f"Check 7: my-turn  root selects '{best_mine}'  (expected 'good', Q=0.9)")
    print(f"         opp-turn root selects '{best_opp2}'   (expected 'bad',  Q=0.2 → worst for me)")
    assert best_mine == "good",  f"Expected 'good', got {best_mine}"
    assert best_opp2 == "bad",   f"Expected 'bad',  got {best_opp2}"
    print("  ✓\n")

    print("=== All checks passed ===")
