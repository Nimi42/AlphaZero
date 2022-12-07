from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from tabulate import tabulate

from mcts import Node, MCTS, SearchType
from tictactoe.model import ResNet, BasicBlock
from train.self_play import Agent
from train.train_model import run_training
from util.data import IO


def run_mcts():
    game = TicTacToe()
    model = TicTacToeDummyPolicy()
    mcts = MCTS(model, 100, SearchType.MCTS)
    node = Node(0, game)

    while node.game_state.get_reward() is None:
        print(node.game_state)
        prev, node = mcts.search(node)

    print(node.game_state)
    print('End')

    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # profiler.disable()
    # stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumtime')
    # stats.print_stats()


def eval_alphazero():
    io = IO('TicTacToe')
    game = TicTacToe()
    model = ResNet(BasicBlock, [2, 2, 2])

    if not io.load_model(model):
        raise FileNotFoundError('No model has been found')

    model.eval()
    mcts = MCTS(model, 1500, SearchType.AlphaZero)

    with torch.no_grad():
        while game.get_reward() is None:
            print(game)
            root, action = mcts.search(game)
            game = game.take_action(action)

    print(game)
    print('End')


def train_alphazero(steps: int):
    io = IO('TicTacToe')
    game = TicTacToe()
    model = ResNet(BasicBlock, [2, 2, 2])
    if not io.load_model(model):
        print('Training new model')
    agent = Agent(game, model)

    for i in range(steps):
        model.eval()
        with torch.no_grad():
            train_data = agent.create_train_data(num_games=20)
            io.save_and_split_data(train_data)

        train, valid = io.load_dataset()
        model.train()
        run_training(model, train, valid, max_epochs=10)

        io.clear_data()
        io.save_model(model)


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
        return tabulate(self.get_canonical_board(), tablefmt="fancy_grid")


if __name__ == '__main__':
    train_alphazero(100)
    # eval_alphazero()
