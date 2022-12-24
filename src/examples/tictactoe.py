from __future__ import annotations

from typing import Tuple

import numpy as np
from tabulate import tabulate

from architecture.policy import ResNet, BasicBlock


def create_policy(device):
    return ResNet(BasicBlock, [2, 2, 2], 9, device)


class TicTacToeDummyPolicy:

    def predict(self, s):
        pi = np.zeros_like(s.board)
        mask_zero = s.board == 0
        probability = 1 / np.count_nonzero(mask_zero)
        pi[mask_zero] = probability

        shape = pi.shape
        predictions = pi.flatten()

        return [(np.unravel_index(i, shape), predictions[i])
                for i in range(predictions.shape[0]) if predictions[i] > 0], None

    def get_best_action(self, game_state):
        moves = np.argwhere(game_state.board == 0)
        idx = np.random.choice(moves.shape[0], 1)[0]
        return tuple(moves[idx])


class TicTacToe:
    _ACTIONS = {
        (0, 0): 0b10000000000010000000000010000000,
        (0, 1): 0b01000000000000001000000000000000,
        (0, 2): 0b00100000000000000000100000001000,
        (1, 0): 0b00001000000001000000000000000000,
        (1, 1): 0b00000100000000000100000001000100,
        (1, 2): 0b00000010000000000000010000000000,
        (2, 0): 0b00000000100000100000000000000010,
        (2, 1): 0b00000000010000000010000000000000,
        (2, 2): 0b00000000001000000000001000100000
    }
    _FULL_BOARD = 0xEEEEEEEE
    __slots__ = ['current_player', 'board', 'p1_bits', 'p2_bits']

    def __init__(self, board: np.ndarray = np.zeros((3, 3)),
                 current_player=1,
                 p1_bits: int = 0,
                 p2_bits: int = 0
                 ):
        self.current_player = current_player
        self.board = board
        self.p1_bits = p1_bits
        self.p2_bits = p2_bits

    @classmethod
    def get_emptyboard(cls):
        return np.zeros((3, 3))

    def take_action(self, action: Tuple[int, int]) -> TicTacToe:
        if self.board[action] == 0:
            new_board = self.board.copy()
            new_board[action] = 1
            return TicTacToe(new_board * -1,
                             self.current_player * -1,
                             self.p2_bits,
                             self.p1_bits | TicTacToe._ACTIONS[action])

    def get_valid_moves(self):
        return self.board == 0

    def get_reward(self):
        if self.p2_bits & self.p2_bits << 1 & self.p2_bits >> 1:
            return -self.current_player
        elif not (self.p1_bits | self.p2_bits) ^ TicTacToe._FULL_BOARD:
            return 0

        return None

    def get_canonical_board(self):
        return self.board * self.current_player

    def __str__(self) -> str:
        x = self.get_canonical_board().astype(object)
        x[x == 1] = 'X'
        x[x == 0] = '_'
        x[x == -1] = 'O'
        return tabulate(x, tablefmt="fancy_grid")
