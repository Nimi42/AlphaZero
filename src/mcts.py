import math
import random
from typing import Any, Tuple, List

import numpy as np


class Node:

    __slots__ = ['prior', 'num_visits', 'total_reward', 'children', 'game_state', 'is_terminal', 'possible_moves', 'is_expanded']

    def __init__(self, prior, game_state, possible_moves):
        self.prior = prior
        self.num_visits = 0
        self.total_reward = 0
        self.children = {}

        self.game_state = game_state
        self.is_terminal = game_state.is_terminal()

        self.possible_moves = possible_moves
        self.is_expanded = False


def puct_score(parent: Node, child: Node):
    c = 4
    prior_score = c * child.prior * math.sqrt(parent.num_visits) / child.num_visits + 1
    if child.num_visits > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = parent.game_state.current_player * child.total_reward / child.num_visits
    else:
        value_score = 0

    return value_score + prior_score


def most_visits(parent: Node, child: Node):
    return child.num_visits


class Reward:

    DRAW = 0.5
    WIN = 1

    def __init__(self, player, result):
        self.player = player
        self.result = result


class MCTS:

    def __init__(self, model, iteration_limit):
        self.iteration_limit = iteration_limit
        self.model = model

    def search(self, root: Node) -> Any:
        for _ in range(self.iteration_limit):
            self._execute_round(root)

        _, child = self._select(root, most_visits)
        return root, child

    def _execute_round(self, node) -> None:
        search_path = [node]
        while not node.is_terminal:
            if node.is_expanded:
                _, node = self._select(node, puct_score)
                search_path.append(node)
            else:
                expanded_node = self._expand(node)
                if expanded_node is not None:
                    node = expanded_node
                    search_path.append(node)
                    break

        reward = self._rollout(node.game_state)
        self._backpropogate(search_path, reward)

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

        if not best_child:
            raise ValueError

        return random.choice(best_child)

    def _expand(self, node: Node):
        try:
            action, probability = next(node.possible_moves)
            new_game_state = node.game_state.take_action(action)
            new_node = Node(probability, new_game_state, self.model.generate_possible_moves(new_game_state))
            node.children[action] = new_node
            return new_node

        except StopIteration:
            node.is_expanded = True
            del node.possible_moves
            return None

    def _rollout(self, game_state) -> Any:
        while not game_state.is_terminal():
            try:
                action = self.model.get_best_action(game_state)
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(game_state))
            game_state = game_state.take_action(action)
        return game_state.get_reward()

    def _backpropogate(self, search_path: List[Node], reward: Any) -> None:
        for node in search_path:
            node.num_visits += 1
            node.total_reward += reward.player
