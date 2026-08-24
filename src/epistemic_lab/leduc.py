from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

RANKS = ("J", "Q", "K")
CARDS = tuple(range(6))
MAX_RAISES_PER_ROUND = 1
ANTE = 1
BET_SIZES = (2, 4)


class Action(str, Enum):
    CHECK = "k"
    BET = "b"
    CALL = "c"
    RAISE = "r"
    FOLD = "f"


def card_rank(card: int) -> int:
    return card // 2


def card_name(card: int) -> str:
    return f"{RANKS[card_rank(card)]}{card % 2 + 1}"


@dataclass(frozen=True)
class LeducState:
    private: tuple[Optional[int], Optional[int]]
    public: Optional[int]
    round_index: int
    to_act: int
    history: tuple[str, ...]
    contributions: tuple[int, int]
    round_contributions: tuple[int, int]
    current_bet: int
    raises_this_round: int
    checks_this_round: int
    terminal: bool = False
    folded_player: Optional[int] = None

    @classmethod
    def initial(cls) -> "LeducState":
        return cls(
            private=(None, None),
            public=None,
            round_index=0,
            to_act=-1,
            history=(),
            contributions=(ANTE, ANTE),
            round_contributions=(0, 0),
            current_bet=0,
            raises_this_round=0,
            checks_this_round=0,
        )

    def is_chance_node(self) -> bool:
        return self.to_act == -1 and not self.terminal

    def chance_outcomes(self) -> list[tuple[float, "LeducState"]]:
        if not self.is_chance_node():
            return []

        if self.private[0] is None:
            outcomes = []
            for p0 in CARDS:
                for p1 in CARDS:
                    if p1 == p0:
                        continue
                    outcomes.append((1.0 / 30.0, self._with_private(p0, p1)))
            return outcomes

        if self.public is None and self.round_index == 1:
            used = {self.private[0], self.private[1]}
            remaining = [card for card in CARDS if card not in used]
            return [(1.0 / len(remaining), self._with_public(card)) for card in remaining]

        raise ValueError("invalid chance state")

    def _with_private(self, p0: int, p1: int) -> "LeducState":
        return self._replace(private=(p0, p1), to_act=0)

    def _with_public(self, card: int) -> "LeducState":
        return self._replace(public=card, to_act=0, history=self.history + ("D",))

    def legal_actions(self) -> tuple[Action, ...]:
        if self.terminal or self.is_chance_node():
            return ()

        actor = self.to_act
        facing_bet = self.round_contributions[actor] < self.current_bet
        if not facing_bet:
            return (Action.CHECK, Action.BET)

        actions = [Action.FOLD, Action.CALL]
        if self.raises_this_round < MAX_RAISES_PER_ROUND:
            actions.append(Action.RAISE)
        return tuple(actions)

    def apply_action(self, action: Action | str) -> "LeducState":
        action = Action(action)
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action} for state {self}")

        if action == Action.CHECK:
            if self.checks_this_round == 1:
                return self._advance_or_terminal(self.history + (action.value,))
            return self._replace(
                to_act=1 - self.to_act,
                checks_this_round=1,
                history=self.history + (action.value,),
            )

        if action == Action.BET:
            return self._commit_to_bet(
                target=self.current_bet + BET_SIZES[self.round_index],
                action=action,
                raises_this_round=self.raises_this_round,
            )

        if action == Action.RAISE:
            return self._commit_to_bet(
                target=self.current_bet + BET_SIZES[self.round_index],
                action=action,
                raises_this_round=self.raises_this_round + 1,
            )

        if action == Action.CALL:
            actor = self.to_act
            needed = self.current_bet - self.round_contributions[actor]
            contributions = add_at(self.contributions, actor, needed)
            round_contributions = add_at(self.round_contributions, actor, needed)
            return self._replace(
                contributions=contributions,
                round_contributions=round_contributions,
            )._advance_or_terminal(self.history + (action.value,))

        if action == Action.FOLD:
            return self._replace(
                terminal=True,
                folded_player=self.to_act,
                history=self.history + (action.value,),
            )

        raise AssertionError("unreachable")

    def _commit_to_bet(
        self,
        target: int,
        action: Action,
        raises_this_round: int,
    ) -> "LeducState":
        actor = self.to_act
        needed = target - self.round_contributions[actor]
        return self._replace(
            contributions=add_at(self.contributions, actor, needed),
            round_contributions=add_at(self.round_contributions, actor, needed),
            current_bet=target,
            raises_this_round=raises_this_round,
            checks_this_round=0,
            to_act=1 - actor,
            history=self.history + (action.value,),
        )

    def _advance_or_terminal(self, history: tuple[str, ...]) -> "LeducState":
        if self.round_index == 1:
            return self._replace(terminal=True, history=history)
        return self._replace(
            round_index=1,
            to_act=-1,
            history=history + ("|",),
            round_contributions=(0, 0),
            current_bet=0,
            raises_this_round=0,
            checks_this_round=0,
        )

    def utility(self, player: int = 0) -> float:
        if not self.terminal:
            raise ValueError("utility is only defined at terminal states")
        pot = sum(self.contributions)
        if self.folded_player is not None:
            winner = 1 - self.folded_player
        else:
            winner = self.showdown_winner()
        if winner is None:
            return pot / 2.0 - self.contributions[player]
        if winner == player:
            return pot - self.contributions[player]
        return -float(self.contributions[player])

    def showdown_winner(self) -> Optional[int]:
        if self.private[0] is None or self.private[1] is None or self.public is None:
            raise ValueError("showdown requires all cards")
        p0_rank = hand_strength(self.private[0], self.public)
        p1_rank = hand_strength(self.private[1], self.public)
        if p0_rank == p1_rank:
            return None
        return 0 if p0_rank > p1_rank else 1

    def info_set_key(self, player: int) -> str:
        private = self.private[player]
        if private is None:
            raise ValueError("infoset requires private cards")
        public = "?" if self.public is None else RANKS[card_rank(self.public)]
        return (
            f"p{player}:r{self.round_index}:h{''.join(self.history)}:"
            f"priv{RANKS[card_rank(private)]}:pub{public}:"
            f"rc{self.round_contributions[player]}:{self.current_bet}:"
            f"raises{self.raises_this_round}:checks{self.checks_this_round}"
        )

    def _replace(self, **kwargs: object) -> "LeducState":
        values = {
            "private": self.private,
            "public": self.public,
            "round_index": self.round_index,
            "to_act": self.to_act,
            "history": self.history,
            "contributions": self.contributions,
            "round_contributions": self.round_contributions,
            "current_bet": self.current_bet,
            "raises_this_round": self.raises_this_round,
            "checks_this_round": self.checks_this_round,
            "terminal": self.terminal,
            "folded_player": self.folded_player,
        }
        values.update(kwargs)
        return LeducState(**values)


def add_at(values: tuple[int, int], index: int, amount: int) -> tuple[int, int]:
    items = list(values)
    items[index] += amount
    return (items[0], items[1])


def hand_strength(private: int, public: int) -> tuple[int, int]:
    rank = card_rank(private)
    pair = int(rank == card_rank(public))
    return (pair, rank)


def enumerate_terminal_states(state: Optional[LeducState] = None) -> Iterable[LeducState]:
    state = state or LeducState.initial()
    if state.terminal:
        yield state
        return
    if state.is_chance_node():
        for _, child in state.chance_outcomes():
            yield from enumerate_terminal_states(child)
        return
    for action in state.legal_actions():
        yield from enumerate_terminal_states(state.apply_action(action))
