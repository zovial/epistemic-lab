from epistemic_lab.leduc import Action, LeducState, enumerate_terminal_states


def test_ordered_private_deals_and_terminal_enumeration():
    assert len(LeducState.initial().chance_outcomes()) == 30
    assert sum(1 for _ in enumerate_terminal_states()) == 5520


def test_showdown_utility_is_net_of_contributions():
    state = (
        LeducState.initial()
        ._with_private(4, 0)
        .apply_action(Action.CHECK)
        .apply_action(Action.CHECK)
        ._with_public(2)
        .apply_action(Action.CHECK)
        .apply_action(Action.CHECK)
    )
    assert state.terminal
    assert state.utility(0) == 1.0
    assert state.utility(1) == -1.0


def test_fold_utility_awards_pot_to_other_player():
    state = (
        LeducState.initial()
        ._with_private(0, 4)
        .apply_action(Action.BET)
        .apply_action(Action.FOLD)
    )
    assert state.terminal
    assert state.utility(0) == 1.0
    assert state.utility(1) == -1.0
