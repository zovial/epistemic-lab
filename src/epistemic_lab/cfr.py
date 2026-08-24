from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Mapping

import numpy as np

from .leduc import Action, LeducState


@dataclass
class InfoSetNode:
    actions: tuple[Action, ...]
    regret_sum: np.ndarray = field(init=False)
    strategy_sum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.regret_sum = np.zeros(len(self.actions), dtype=float)
        self.strategy_sum = np.zeros(len(self.actions), dtype=float)

    def strategy(self) -> np.ndarray:
        positive = np.maximum(self.regret_sum, 0.0)
        total = positive.sum()
        if total <= 0.0:
            return np.full(len(self.actions), 1.0 / len(self.actions))
        return positive / total

    def average_strategy(self) -> dict[Action, float]:
        total = self.strategy_sum.sum()
        if total <= 0.0:
            probs = np.full(len(self.actions), 1.0 / len(self.actions))
        else:
            probs = self.strategy_sum / total
        return {action: float(prob) for action, prob in zip(self.actions, probs)}


class CFRSolver:
    def __init__(self) -> None:
        self.nodes: dict[str, InfoSetNode] = {}
        self.utility_trace: list[float] = []

    def train(self, iterations: int) -> "CFRSolver":
        for _ in range(iterations):
            self.utility_trace.append(self._cfr(LeducState.initial(), reach=(1.0, 1.0), chance_reach=1.0))
        return self

    def average_strategy(self) -> dict[str, dict[Action, float]]:
        return {key: node.average_strategy() for key, node in self.nodes.items()}

    def _node_for(self, state: LeducState) -> InfoSetNode:
        key = state.info_set_key(state.to_act)
        actions = state.legal_actions()
        node = self.nodes.get(key)
        if node is None:
            node = InfoSetNode(actions)
            self.nodes[key] = node
        elif node.actions != actions:
            raise RuntimeError(f"infoset action mismatch for {key}")
        return node

    def _cfr(self, state: LeducState, reach: tuple[float, float], chance_reach: float) -> float:
        if state.terminal:
            return state.utility(0)
        if state.is_chance_node():
            return sum(
                prob * self._cfr(child, reach, chance_reach * prob)
                for prob, child in state.chance_outcomes()
            )

        actor = state.to_act
        node = self._node_for(state)
        strategy = node.strategy()
        action_utils = np.zeros(len(node.actions), dtype=float)
        node_util = 0.0

        for index, action in enumerate(node.actions):
            next_reach = list(reach)
            next_reach[actor] *= strategy[index]
            action_utils[index] = self._cfr(
                state.apply_action(action),
                (next_reach[0], next_reach[1]),
                chance_reach,
            )
            node_util += strategy[index] * action_utils[index]

        node.strategy_sum += chance_reach * reach[actor] * strategy
        if actor == 0:
            node.regret_sum += chance_reach * reach[1] * (action_utils - node_util)
        else:
            node.regret_sum += chance_reach * reach[0] * (node_util - action_utils)
        return float(node_util)


def expected_utility(
    strategy_profile: Mapping[str, Mapping[Action, float]],
    state: LeducState | None = None,
) -> float:
    state = state or LeducState.initial()
    if state.terminal:
        return state.utility(0)
    if state.is_chance_node():
        return sum(prob * expected_utility(strategy_profile, child) for prob, child in state.chance_outcomes())

    key = state.info_set_key(state.to_act)
    actions = state.legal_actions()
    strategy = strategy_profile.get(key)
    if strategy is None:
        prob = 1.0 / len(actions)
        return sum(prob * expected_utility(strategy_profile, state.apply_action(action)) for action in actions)
    return sum(
        strategy.get(action, 0.0) * expected_utility(strategy_profile, state.apply_action(action))
        for action in actions
    )


def best_response_value(
    strategy_profile: Mapping[str, Mapping[Action, float]],
    br_player: int,
    state: LeducState | None = None,
) -> float:
    if state is not None:
        return _perfect_recall_best_response_value(strategy_profile, br_player, state)
    return _infoset_best_response_value(strategy_profile, br_player)


def _perfect_recall_best_response_value(
    strategy_profile: Mapping[str, Mapping[Action, float]],
    br_player: int,
    state: LeducState,
) -> float:
    if state.terminal:
        return state.utility(0)
    if state.is_chance_node():
        return sum(
            prob * _perfect_recall_best_response_value(strategy_profile, br_player, child)
            for prob, child in state.chance_outcomes()
        )

    actions = state.legal_actions()
    values = [
        _perfect_recall_best_response_value(strategy_profile, br_player, state.apply_action(action))
        for action in actions
    ]
    if state.to_act == br_player:
        return max(values) if br_player == 0 else min(values)

    key = state.info_set_key(state.to_act)
    strategy = strategy_profile.get(key)
    if strategy is None:
        return float(np.mean(values))
    return sum(strategy.get(action, 0.0) * value for action, value in zip(actions, values))


def _infoset_best_response_value(
    strategy_profile: Mapping[str, Mapping[Action, float]],
    br_player: int,
) -> float:
    buckets: dict[str, list[tuple[float, LeducState]]] = defaultdict(list)

    def collect(state: LeducState, reach: float) -> None:
        if state.terminal:
            return
        if state.is_chance_node():
            for prob, child in state.chance_outcomes():
                collect(child, reach * prob)
            return
        if state.to_act == br_player:
            buckets[state.info_set_key(br_player)].append((reach, state))
            for action in state.legal_actions():
                collect(state.apply_action(action), reach)
            return
        key = state.info_set_key(state.to_act)
        strategy = strategy_profile.get(key)
        actions = state.legal_actions()
        if strategy is None:
            uniform = 1.0 / len(actions)
            for action in actions:
                collect(state.apply_action(action), reach * uniform)
        else:
            for action in actions:
                collect(state.apply_action(action), reach * strategy.get(action, 0.0))

    br_policy: dict[str, Action] = {}

    def value(state: LeducState) -> float:
        if state.terminal:
            return state.utility(0)
        if state.is_chance_node():
            return sum(prob * value(child) for prob, child in state.chance_outcomes())
        actions = state.legal_actions()
        if state.to_act == br_player:
            chosen = br_policy.get(state.info_set_key(br_player))
            if chosen is None:
                child_values = [value(state.apply_action(action)) for action in actions]
                return max(child_values) if br_player == 0 else min(child_values)
            return value(state.apply_action(chosen))
        key = state.info_set_key(state.to_act)
        strategy = strategy_profile.get(key)
        if strategy is None:
            return float(np.mean([value(state.apply_action(action)) for action in actions]))
        return sum(strategy.get(action, 0.0) * value(state.apply_action(action)) for action in actions)

    collect(LeducState.initial(), reach=1.0)
    ordered = sorted(
        buckets.items(),
        key=lambda item: max(len(state.history) for _, state in item[1]),
        reverse=True,
    )
    for key, states in ordered:
        actions = states[0][1].legal_actions()
        action_values = []
        for action in actions:
            numerator = sum(reach * value(state.apply_action(action)) for reach, state in states)
            denominator = sum(reach for reach, _ in states)
            action_values.append(numerator / denominator if denominator > 0.0 else 0.0)
        index = int(np.argmax(action_values) if br_player == 0 else np.argmin(action_values))
        br_policy[key] = actions[index]
    return value(LeducState.initial())


def exploitability(strategy_profile: Mapping[str, Mapping[Action, float]]) -> float:
    br0 = best_response_value(strategy_profile, br_player=0)
    br1_as_p0_utility = best_response_value(strategy_profile, br_player=1)
    return 0.5 * (br0 - br1_as_p0_utility)
