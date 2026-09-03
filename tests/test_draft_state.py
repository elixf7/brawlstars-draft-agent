"""Draft legality. Search explores states by the thousand, so an illegal one
does not raise — it just quietly makes the recommendation wrong."""
import pytest

from bsdraft.mcts.state import DraftState, apply_pick, available_actions

VOCAB = ("SHELLY", "COLT", "BULL", "RICO", "MORTIS", "SPIKE", "CROW")


def state(mine=(), opp=(), first=True, bans=()):
    return DraftState(
        my_team=frozenset(mine), opp_team=frozenset(opp),
        mode="brawlBall", map_name="Hot Potato", skill_ns=0.5,
        is_first_pick=first, bans=frozenset(bans),
    )


# ------------------------------------------------------------- snake order
def test_first_pick_follows_the_snake_order():
    """Blue 1, Red 1, Red 2, Blue 2, Blue 3, Red 3."""
    s, seen = state(first=True), []
    for brawler in VOCAB[:6]:
        seen.append(s.whose_turn)
        s = apply_pick(s, brawler)
    assert seen == ["mine", "opp", "opp", "mine", "mine", "opp"]


def test_second_pick_is_the_mirror():
    s, seen = state(first=False), []
    for brawler in VOCAB[:6]:
        seen.append(s.whose_turn)
        s = apply_pick(s, brawler)
    assert seen == ["opp", "mine", "mine", "opp", "opp", "mine"]


def test_both_sides_end_with_three():
    for first in (True, False):
        s = state(first=first)
        for brawler in VOCAB[:6]:
            s = apply_pick(s, brawler)
        assert len(s.my_team) == 3 and len(s.opp_team) == 3


def test_asking_whose_turn_after_the_draft_is_an_error():
    s = state(mine=VOCAB[:3], opp=VOCAB[3:6])
    with pytest.raises(ValueError, match="complete"):
        _ = s.whose_turn


# ---------------------------------------------------------------- legality
def test_a_picked_brawler_leaves_the_pool():
    """The same brawler may not appear twice in one draft, on either side."""
    s = state(mine=("SHELLY",), opp=("COLT",))
    actions = available_actions(s, VOCAB)
    assert "SHELLY" not in actions and "COLT" not in actions
    assert set(actions) == set(VOCAB) - {"SHELLY", "COLT"}


def test_bans_are_excluded():
    s = state(bans=("CROW", "SPIKE"))
    assert {"CROW", "SPIKE"}.isdisjoint(available_actions(s, VOCAB))


def test_actions_follow_vocabulary_order():
    """Order must match FeatureSchema.vocab, or feature indices misalign."""
    assert available_actions(state(), VOCAB) == list(VOCAB)


def test_a_completed_draft_offers_nothing_useful():
    s = state(mine=VOCAB[:3], opp=VOCAB[3:6])
    assert s.is_terminal
    assert set(available_actions(s, VOCAB)) == {"CROW"}   # unpicked, but unusable


# ------------------------------------------------------------ immutability
def test_applying_a_pick_leaves_the_original_untouched():
    """States are used as dictionary keys during search; mutation would corrupt
    the tree."""
    s = state()
    after = apply_pick(s, "SHELLY")
    assert s.my_team == frozenset()
    assert after.my_team == frozenset({"SHELLY"})
    assert after is not s


def test_context_survives_a_pick():
    s = state()
    after = apply_pick(s, "SHELLY")
    assert (after.mode, after.map_name, after.skill_ns, after.is_first_pick) == \
           (s.mode, s.map_name, s.skill_ns, s.is_first_pick)


def test_equal_states_are_interchangeable_as_keys():
    a = state(mine=("SHELLY", "COLT"))
    b = state(mine=("COLT", "SHELLY"))
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1


def test_pick_number_counts_both_teams():
    assert state().pick_number == 0
    assert state(mine=("SHELLY",), opp=("COLT", "BULL")).pick_number == 3
    assert state(mine=VOCAB[:3], opp=VOCAB[3:6]).pick_number == 6


def test_terminal_only_when_both_teams_are_full():
    assert not state(mine=VOCAB[:3]).is_terminal
    assert not state(mine=VOCAB[:3], opp=VOCAB[3:5]).is_terminal
    assert state(mine=VOCAB[:3], opp=VOCAB[3:6]).is_terminal
