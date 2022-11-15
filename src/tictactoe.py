from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy.signal import convolve2d
from tabulate import tabulate

from mcts_bak import MCTS, Node, Reward
from mcts import MCTS, Node


class TicTacToeDummyPolicy:

    def _predict(self, game_state):
        prediction = np.zeros_like(game_state.board)
        mask_zero = game_state.board == 0
        probability = 1 / np.count_nonzero(mask_zero)
        prediction[mask_zero] = probability
        return prediction

    def generate_possible_moves(self, game_state):
        predictions = self._predict(game_state)
        shape = predictions.shape
        predictions = predictions.flatten()
        for i in range(predictions.shape[0]):
            if predictions[0] > 0:
                yield np.unravel_index(i, shape), predictions[0]
            predictions = np.delete(predictions, 0)

    def get_best_action(self, game_state):
        moves = np.argwhere(game_state.board == 0)
        idx = np.random.choice(moves.shape[0], 1)[0]
        return moves[idx]


class TicTacToe:
    # Row & Column Kernel
    HORIZONTAL = np.array([[1, 1, 1]])
    VERTICAL = np.transpose(HORIZONTAL)
    # Diagonals
    DIAG1 = np.eye(3, dtype=np.uint8)
    DIAG2 = np.fliplr(DIAG1)

    KERNEL = (HORIZONTAL, VERTICAL, DIAG1, DIAG2)

    def __init__(self, board: np.ndarray = np.zeros((3, 3)), current_player=1):
        self.board = board
        self.current_player = current_player
        self.reward = None

    def take_action(self, action: Tuple[int, int]) -> TicTacToe:
        if self.board[action[0], action[1]] != 0:
            raise IndexError('A player has already placed a token here', self.board)
        new_board = self.board.copy()
        new_board[action[0], action[1]] = 1
        return TicTacToe(new_board * -1, self.current_player * -1)

    def is_terminal(self):
        condition = -3

        for k in TicTacToe.KERNEL:
            if np.any(convolve2d(self.board, k, mode="valid") == condition):
                self.reward = Reward(-self.current_player, Reward.WIN)
                return True

        if np.count_nonzero(self.board == 0) == 0:
            self.reward = Reward(0, Reward.DRAW)
            return True

        return False

    def get_reward(self):
        if self.reward is None:
            raise ValueError("Game State has never terminated.")
        return self.reward

    def get_canonical_board(self):
        return self.board * self.current_player

    def __str__(self) -> str:
        return tabulate(self.get_canonical_board(), tablefmt="fancy_grid")


if __name__ == '__main__':
    game = TicTacToe()
    model = TicTacToeDummyPolicy()

    mcts = MCTS(model, 10000)
    node = Node(0, game, model.generate_possible_moves(game))

    while not node.is_terminal:
        print(node.game_state)
        prev, node = mcts.search(node)

    print(node.game_state)
    print('End')


    # TODO: Clean up Code
    #   - Switch view of player on board and fix mcts algorithm to fit. (It's easier for a machine learning network to
    #   make predictions if the view is always the same
    #   - Write tests to make sure the algorithm is working fine
    #   - Fix code to clean up redundancies
    #   - Make MCTS compatible with machine learning
    #   - Write test backprop