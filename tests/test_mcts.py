"""Search invariants.

Tree search fails quietly: a wrong Q sign or a lost visit count still returns a
brawler name, and it looks exactly like a right answer."""
import math

import pytest

from bsdraft.mcts.node import MCTSNode, backpropagate, ucb1_score
from bsdraft.mcts.state import DraftState


def state(mine=(), opp=(), first=True):
    return DraftState(
        my_team=frozenset(mine), opp_team=frozenset(opp),
        mode="brawlBall", map_name="Hot Potato", skill_ns=0.0,
        is_first_pick=first, bans=frozenset(),
    )


def node(visits=0, value=0.0, **kw):
    n = MCTSNode(state=state(), **kw)
    n.visit_count, n.value_sum = visits, value
    return n


# ------------------------------------------------------------------ q_value
def test_q_is_the_mean_of_what_was_backed_up():
    assert node(visits=4, value=3.0).q_value == pytest.approx(0.75)


def test_an_unvisited_node_has_no_estimate():
    """Not 0.5 — callers must check visit_count rather than trust a number."""
    assert node().q_value == 0.0
    assert node().visit_count == 0


# --------------------------------------------------------------------- UCB1
def test_unvisited_children_are_tried_before_visited_ones():
    """Otherwise a single unlucky rollout can bury an action permanently."""
    assert ucb1_score(node(), parent_visits=10, c=0.5) == float("inf")


def test_more_visits_lowers_the_exploration_bonus():
    little, lots = node(visits=1, value=0.5), node(visits=100, value=50.0)
    assert little.q_value == lots.q_value          # same mean
    assert ucb1_score(little, 200, 0.5) > ucb1_score(lots, 200, 0.5)


def test_the_exploration_constant_scales_the_bonus():
    n = node(visits=4, value=2.0)
    greedy = ucb1_score(n, 100, 0.0)
    curious = ucb1_score(n, 100, 1.0)
    assert greedy == pytest.approx(n.q_value)      # c=0 is pure exploitation
    assert curious > greedy


def test_ucb1_matches_the_formula():
    n = node(visits=4, value=2.0)
    expected = 0.5 + 0.7 * math.sqrt(math.log(100) / 4)
    assert ucb1_score(n, 100, 0.7) == pytest.approx(expected)


def test_the_opponent_maximises_our_loss():
    """Q is stored from our perspective throughout, so opponent nodes flip the
    exploitation term. Getting this wrong makes the opponent cooperate."""
    good_for_us = node(visits=10, value=9.0)       # Q = 0.9
    assert ucb1_score(good_for_us, 100, 0.0, flip=False) == pytest.approx(0.9)
    assert ucb1_score(good_for_us, 100, 0.0, flip=True) == pytest.approx(0.1)


def test_puct_gives_unvisited_children_a_finite_score():
    """PUCT orders by prior instead of trying everything once."""
    s = ucb1_score(node(), parent_visits=25, c=1.0, prior=0.3)
    assert math.isfinite(s)
    assert s == pytest.approx(0.0 + 1.0 * 0.3 * 5 / 1)


def test_puct_prefers_the_higher_prior_when_all_else_is_equal():
    likely = ucb1_score(node(), 25, 1.0, prior=0.6)
    unlikely = ucb1_score(node(), 25, 1.0, prior=0.05)
    assert likely > unlikely


# ------------------------------------------------------------ backpropagate
def test_backpropagation_updates_every_node_on_the_path():
    root = node()
    child = MCTSNode(state=state(mine=("SHELLY",)), parent=root)
    leaf = MCTSNode(state=state(mine=("SHELLY",), opp=("COLT",)), parent=child)

    backpropagate([root, child, leaf], 0.8)
    for n in (root, child, leaf):
        assert n.visit_count == 1
        assert n.value_sum == pytest.approx(0.8)


def test_repeated_backpropagation_accumulates():
    root = node()
    for p in (1.0, 0.0, 1.0, 0.5):
        backpropagate([root], p)
    assert root.visit_count == 4
    assert root.q_value == pytest.approx(0.625)


def test_values_are_stored_from_our_perspective_at_every_depth():
    """No sign flip in backprop — the adversarial effect lives in selection.
    Flipping in both places silently cancels out."""
    root = node()
    opp_turn = MCTSNode(state=state(mine=("SHELLY",)), parent=root)
    backpropagate([root, opp_turn], 0.9)
    assert opp_turn.q_value == pytest.approx(0.9)   # not 0.1
