import math
import random
from enum import Enum
from typing import Any, Tuple, List

import numpy as np


class Node:
    __slots__ = ['P_sa', 'N_sa', 'Q_sa', 'children', 'game_state']

    def __init__(self, prior, game_state):
        self.P_sa = prior
        self.N_sa = 0
        self.Q_sa = 0
        self.children = {}

        self.game_state = game_state

    def is_expanded(self):
        return True if self.children else False


class Mode(Enum):
    MCTS = 'MCTS'
    AlphaZero = 'AlphaZero'


class MCTS:

    def __init__(self, model, iteration_limit, mode: Mode):
        self.iteration_limit = iteration_limit
        self.model = model

        match mode:
            case Mode.MCTS:
                self._playout = self._rollout
                self._score_func = self._ucb_score
            case Mode.AlphaZero:
                # TODO: _evaluate should be assigned to _playout. But there is no neural network models yet
                self._playout = self._rollout
                self._score_func = self._puct_score

    def search(self, root: Node) -> Any:
        if not root.is_expanded():
            self._expand(root, self.model.predict(root.game_state))

        for _ in range(self.iteration_limit):
            self._execute_round(root)

        _, child = self._select(root, self._most_visits)
        return root, child

    def _execute_round(self, root) -> None:
        search_path = []
        to_play = root.game_state.current_player
        node = root
        while node.is_expanded():
            _, node = self._select(node, self._score_func)
            search_path.append(node)

        value = node.game_state.get_reward()
        if value is None:
            probabilities, value = self._playout(node.game_state)
            self._expand(node, probabilities)

        root.N_sa += 1
        self._backpropogate(search_path, to_play, value)

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

    def _expand(self, node: Node, probabilities):
        for action, probability in probabilities:
            new_game_state = node.game_state.take_action(action)
            new_node = Node(probability, new_game_state)
            node.children[action] = new_node

    def _backpropogate(self, search_path: List[Node], to_play, reward: Any) -> None:
        for node in search_path:
            node.Q_sa = (node.N_sa * node.Q_sa + to_play * reward) / (node.N_sa + 1)
            node.N_sa += 1
            to_play *= -1

    def _rollout(self, game_state) -> Any:
        probabilities = self.model.predict(game_state)
        reward = game_state.get_reward()
        while reward is None:
            action = self.model.get_best_action(game_state)
            game_state = game_state.take_action(action)
            reward = game_state.get_reward()
        return probabilities, reward

    def _evaluate(self, game_state) -> Any:
        # TODO: A neural network will at this point evaluate the given position and return
        #   actions their probabilities and a value for the given state
        pass

    @staticmethod
    def _puct_score(parent: Node, child: Node):
        c = 4
        prior_score = c * child.P_sa * math.sqrt(parent.N_sa) / (1 + child.N_sa)
        return child.Q_sa + prior_score

    @staticmethod
    def _ucb_score(parent: Node, child: Node):
        exploration_value = 20 #math.sqrt(2)
        exploitation = (1 + child.Q_sa) / 2
        exploration = exploration_value * math.sqrt(math.log(parent.N_sa) / child.N_sa if child.N_sa else 1)
        return exploitation + exploration

    @staticmethod
    def _most_visits(parent: Node, child: Node):
        return child.N_sa
