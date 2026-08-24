from epistemic_lab.leduc import LeducState


def test_initial_state_has_chance_deals():
    state = LeducState.initial()
    deals = state.chance_outcomes()
    assert len(deals) == 30
