import numpy as np

from epistemic_lab.leduc import Action, LeducState
from epistemic_lab.models import (
    BayesianOpponentModeler,
    Hypothesis,
    ScriptedOpponent,
    brute_force_hypothesis_posterior,
    range_posterior,
)


def test_posterior_matches_brute_force_enumeration_to_1e_9():
    hypotheses = [
        Hypothesis("tight", ScriptedOpponent("tight", aggression=0.2, looseness=0.25)),
        Hypothesis("loose", ScriptedOpponent("loose", aggression=0.45, looseness=0.9)),
        Hypothesis("aggro", ScriptedOpponent("aggro", aggression=0.9, looseness=0.8, raise_bias=0.8)),
    ]
    prior = np.array([0.2, 0.5, 0.3])
    observations = [
        (LeducState.initial()._with_private(0, 4).apply_action(Action.CHECK), 1, Action.BET),
        (
            LeducState.initial()
            ._with_private(2, 5)
            .apply_action(Action.CHECK)
            .apply_action(Action.BET)
            .apply_action(Action.RAISE),
            1,
            Action.CALL,
        ),
    ]
    modeler = BayesianOpponentModeler(hypotheses, prior)
    for state, player, action in observations:
        modeler.update(state, player, action)
    brute = brute_force_hypothesis_posterior(hypotheses, prior, observations)
    assert np.max(np.abs(modeler.posterior - brute)) < 1e-9


def test_range_posterior_is_normalized_over_unblocked_cards():
    policy = ScriptedOpponent("balanced")
    obs_state = LeducState.initial()._with_private(0, 4).apply_action(Action.CHECK)
    posterior = range_posterior(
        hero_player=0,
        hero_card=0,
        public_card=None,
        observations=[(obs_state, Action.BET)],
        policy=policy,
    )
    assert set(posterior) == {1, 2, 3, 4, 5}
    assert abs(sum(posterior.values()) - 1.0) < 1e-12
