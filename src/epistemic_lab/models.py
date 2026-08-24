from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .leduc import Action, LeducState, card_rank


class Policy(Protocol):
    def action_probs(self, state: LeducState, player: int) -> dict[Action, float]:
        ...


@dataclass(frozen=True)
class ScriptedOpponent:
    name: str
    aggression: float = 0.5
    looseness: float = 0.5
    raise_bias: float = 0.25
    off_path_probe_response: float = 0.5

    def action_probs(self, state: LeducState, player: int) -> dict[Action, float]:
        actions = state.legal_actions()
        if not actions:
            return {}

        strength = normalized_strength(state, player)
        facing_bet = state.round_contributions[player] < state.current_bet
        off_path = "r" in state.history and state.history.count("r") >= 1

        scores: dict[Action, float] = {}
        for action in actions:
            if action == Action.CHECK:
                scores[action] = 1.15 - self.aggression + 0.35 * (1.0 - strength)
            elif action == Action.BET:
                scores[action] = 0.15 + self.aggression * (0.35 + strength)
            elif action == Action.FOLD:
                fold_base = 1.25 - self.looseness - 0.9 * strength
                if off_path:
                    fold_base += 0.8 * (1.0 - self.off_path_probe_response)
                scores[action] = fold_base
            elif action == Action.CALL:
                call_base = 0.2 + self.looseness + 0.6 * strength
                if off_path:
                    call_base += 0.45 * self.off_path_probe_response
                scores[action] = call_base
            elif action == Action.RAISE:
                raise_base = 0.08 + self.raise_bias * (0.25 + strength)
                if off_path:
                    raise_base += 0.5 * self.off_path_probe_response * strength
                scores[action] = raise_base

        # Facing a bet should make pure aggression less important than pot-continuation.
        if facing_bet and Action.RAISE in scores:
            scores[Action.RAISE] *= 0.55 + self.aggression

        return softmax_scores(scores)


@dataclass(frozen=True)
class EpsilonContaminatedPolicy:
    base: Policy
    epsilon: float

    def action_probs(self, state: LeducState, player: int) -> dict[Action, float]:
        base_probs = self.base.action_probs(state, player)
        if not base_probs:
            return {}
        uniform = 1.0 / len(base_probs)
        return {
            action: (1.0 - self.epsilon) * prob + self.epsilon * uniform
            for action, prob in base_probs.items()
        }


@dataclass(frozen=True)
class Hypothesis:
    name: str
    policy: Policy
    parameter_count: int = 1


class BayesianOpponentModeler:
    def __init__(self, hypotheses: list[Hypothesis], prior: np.ndarray | None = None) -> None:
        if not hypotheses:
            raise ValueError("at least one hypothesis is required")
        self.hypotheses = hypotheses
        if prior is None:
            prior = np.full(len(hypotheses), 1.0 / len(hypotheses))
        prior = np.asarray(prior, dtype=float)
        if prior.shape != (len(hypotheses),):
            raise ValueError("prior shape must match hypotheses")
        if prior.sum() <= 0:
            raise ValueError("prior must have positive mass")
        self.posterior = prior / prior.sum()
        self.log_evidence = 0.0

    def update(self, state: LeducState, player: int, action: Action | str) -> np.ndarray:
        action = Action(action)
        likelihoods = np.array(
            [hyp.policy.action_probs(state, player).get(action, 0.0) for hyp in self.hypotheses],
            dtype=float,
        )
        unnorm = self.posterior * likelihoods
        evidence = unnorm.sum()
        if evidence <= 0.0:
            raise ValueError(f"all hypotheses assigned zero likelihood to {action}")
        self.posterior = unnorm / evidence
        self.log_evidence += float(np.log(evidence))
        return self.posterior.copy()


def brute_force_hypothesis_posterior(
    hypotheses: list[Hypothesis],
    prior: np.ndarray,
    observations: list[tuple[LeducState, int, Action | str]],
) -> np.ndarray:
    posterior = np.asarray(prior, dtype=float)
    posterior = posterior / posterior.sum()
    for state, player, action in observations:
        likelihoods = np.array(
            [hyp.policy.action_probs(state, player).get(Action(action), 0.0) for hyp in hypotheses],
            dtype=float,
        )
        posterior = posterior * likelihoods
        posterior = posterior / posterior.sum()
    return posterior


def range_posterior(
    hero_player: int,
    hero_card: int,
    public_card: int | None,
    observations: list[tuple[LeducState, Action | str]],
    policy: Policy,
) -> dict[int, float]:
    villain = 1 - hero_player
    blocked = {hero_card}
    if public_card is not None:
        blocked.add(public_card)
    candidates = [card for card in range(6) if card not in blocked]
    weights = np.full(len(candidates), 1.0 / len(candidates))

    for obs_state, action in observations:
        likelihoods = []
        for card in candidates:
            private = [None, None]
            private[hero_player] = hero_card
            private[villain] = card
            state = obs_state._replace(private=(private[0], private[1]), public=public_card)
            likelihoods.append(policy.action_probs(state, villain).get(Action(action), 0.0))
        weights *= np.asarray(likelihoods, dtype=float)
        total = weights.sum()
        if total <= 0.0:
            raise ValueError("observations have zero probability under the policy")
        weights /= total

    return {card: float(weight) for card, weight in zip(candidates, weights)}


def normalized_strength(state: LeducState, player: int) -> float:
    private = state.private[player]
    if private is None:
        raise ValueError("policy needs the acting player's private card")
    rank = card_rank(private)
    if state.public is not None and rank == card_rank(state.public):
        return 1.0
    return rank / 2.0


def softmax_scores(scores: dict[Action, float]) -> dict[Action, float]:
    actions = list(scores)
    raw = np.array([max(scores[action], 1e-9) for action in actions], dtype=float)
    raw = raw / raw.sum()
    return {action: float(prob) for action, prob in zip(actions, raw)}


def default_hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis(
            "nit",
            ScriptedOpponent("nit", aggression=0.18, looseness=0.2, raise_bias=0.12, off_path_probe_response=0.2),
            parameter_count=4,
        ),
        Hypothesis(
            "calling_station",
            ScriptedOpponent(
                "calling_station",
                aggression=0.22,
                looseness=0.92,
                raise_bias=0.08,
                off_path_probe_response=0.85,
            ),
            parameter_count=4,
        ),
        Hypothesis(
            "maniac",
            ScriptedOpponent("maniac", aggression=0.9, looseness=0.85, raise_bias=0.75, off_path_probe_response=0.7),
            parameter_count=4,
        ),
        Hypothesis(
            "balanced",
            ScriptedOpponent("balanced", aggression=0.52, looseness=0.55, raise_bias=0.33, off_path_probe_response=0.5),
            parameter_count=4,
        ),
    ]
