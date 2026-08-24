from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .leduc import Action, LeducState
from .models import Policy


def collect_decision_states(
    player: int | None = None,
    predicate: Callable[[LeducState], bool] | None = None,
) -> list[LeducState]:
    states: list[LeducState] = []

    def visit(state: LeducState) -> None:
        if state.terminal:
            return
        if state.is_chance_node():
            for _, child in state.chance_outcomes():
                visit(child)
            return
        if (player is None or state.to_act == player) and (predicate is None or predicate(state)):
            states.append(state)
        for action in state.legal_actions():
            visit(state.apply_action(action))

    visit(LeducState.initial())
    return states


def sample_action(probs: dict[Action, float], rng: np.random.Generator) -> Action:
    actions = list(probs)
    weights = np.array([probs[action] for action in actions], dtype=float)
    weights /= weights.sum()
    return actions[int(rng.choice(len(actions), p=weights))]


def sample_observations(
    policy: Policy,
    n: int,
    player: int = 1,
    seed: int = 7,
    predicate: Callable[[LeducState], bool] | None = None,
) -> list[tuple[LeducState, int, Action]]:
    rng = np.random.default_rng(seed)
    states = collect_decision_states(player=player, predicate=predicate)
    observations: list[tuple[LeducState, int, Action]] = []
    for _ in range(n):
        state = states[int(rng.integers(0, len(states)))]
        action = sample_action(policy.action_probs(state, player), rng)
        observations.append((state, player, action))
    return observations


def predictive_log_likelihood(policy: Policy, observations: list[tuple[LeducState, int, Action]]) -> float:
    total = 0.0
    for state, player, action in observations:
        prob = max(policy.action_probs(state, player).get(action, 0.0), 1e-15)
        total += float(np.log(prob))
    return total
