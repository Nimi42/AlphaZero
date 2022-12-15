import math
import random
from enum import Enum
from typing import Any, Tuple, List, Dict

import numpy as np


class Node:
    __slots__ = ['P_sa', 'N_sa', 'Q_sa', 'children', 'game_state']

    def __init__(self, prior, game_state):
        self.P_sa: float = prior
        self.N_sa: int = 0
        self.Q_sa: int = 0
        self.children: Dict = {}

        self.game_state = game_state

    def is_expanded(self):
        return True if self.children else False


class SearchType(Enum):
    MCTS = 'MCTS'
    AlphaZero = 'AlphaZero'


# noinspection NonAsciiCharacters
class MCTS:

    def __init__(self, model, iteration_limit, mode: SearchType):
        self.iteration_limit = iteration_limit
        self.model = model
        self.mode = mode

        match mode:
            case SearchType.MCTS:
                self._playout = self._rollout
                self._score_func = self._ucb_score
            case SearchType.AlphaZero:
                self._playout = self.model.predict
                self._score_func = self._puct_score

        self.mode = mode

    def search(self, s, train: bool = False) -> Any:
        root = Node(0, s)
        if self.mode == SearchType.AlphaZero:
            π, _ = self.model.predict(root.game_state)
            self._expand(root, π)

        if train:
            self.add_dirichletnoise(root)

        for _ in range(self.iteration_limit):
            self._execute_round(root)

        action, child = self._select(root, self._most_visits)
        return root, action

    def _execute_round(self, node: Node) -> None:
        search_path = []
        to_play = node.game_state.current_player
        node.N_sa += 1
        while node.is_expanded():
            _, node = self._select(node, self._score_func)
            search_path.append(node)

        v = node.game_state.get_reward()
        if v is None:
            π, v = self._playout(node.game_state)
            self._expand(node, π)

        self._backpropogate(search_path, to_play, v)

    def _select(self, parent, score_func) -> Tuple[Any, Node]:
        best_score = -np.inf
        best_child = []

        for action, child in parent.children.items():
            score = score_func(parent, child)

            if score > best_score:
                best_score = score
                best_child = [(action, child)]
            elif score == best_score:
                best_child.append((action, child))

        return random.choice(best_child)

    def _expand(self, node: Node, π):
        for action, probability in π:
            new_game_state = node.game_state.take_action(action)
            new_node = Node(probability, new_game_state)
            node.children[action] = new_node

    def _backpropogate(self, search_path: List[Node], to_play, reward: Any) -> None:
        for node in search_path:
            node.Q_sa = (node.N_sa * node.Q_sa + to_play * reward) / (node.N_sa + 1)
            node.N_sa += 1
            to_play *= -1

    def _rollout(self, game_state) -> Any:
        π, _ = self.model.predict(game_state)
        reward = game_state.get_reward()
        while reward is None:
            action = self.model.get_best_action(game_state)
            game_state = game_state.take_action(action)
            reward = game_state.get_reward()
        return π, reward

    def add_dirichletnoise(self, root):
        α = 0.6
        ε = 0.75
        rng = np.random.default_rng()
        valid_moves = root.game_state.get_valid_moves()
        noise = rng.dirichlet([α] * np.count_nonzero(valid_moves))
        for child, η in zip(root.children.values(), noise):
            child.P_sa = (ε * child.P_sa + (1 - ε) * η)

    @staticmethod
    def _puct_score(parent: Node, child: Node):
        c = 2
        prior_score = c * child.P_sa * math.sqrt(parent.N_sa - 1) / (1 + child.N_sa)
        return child.Q_sa + prior_score

    @staticmethod
    def _ucb_score(parent: Node, child: Node):
        exploration_value = math.sqrt(2)
        exploitation = (1 + child.Q_sa) / 2
        exploration = exploration_value * math.sqrt(math.log(parent.N_sa) / child.N_sa if child.N_sa else 1)
        return exploitation + exploration

    @staticmethod
    def _most_visits(parent: Node, child: Node):
        return child.N_sa
