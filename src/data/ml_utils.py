import os
import random

import numpy as np
from .convert import UCItoNetwork


def batch_generator(path, batch_size):
    converter = UCItoNetwork()

    while True:
        generators = []
        for file in os.listdir(path):
            pgn_path = os.path.join(path, file)
            game = converter.generate_random_samples_from_game(pgn_path)

            generators.append(game)

        while generators:
            x_train, y_train = [], []
            i = 0
            game = None
            while i != batch_size:
                i += 1

                try:
                    game = random.choice(generators)
                    x, y = next(game)
                    x_train.append(x)
                    y_train.append(y)
                except IndexError:
                    if not generators:
                        break
                    generators.remove(game)
            else:
                x_train = list(zip(*x_train))
                x_train = [np.stack(i) for i in x_train]
                y_train = np.stack(y_train)
                yield x_train, y_train


def get_n_best_moves(move, n, perspective):
    perspective = 0 if perspective else 7
    idx = np.argpartition(-move, n, None)[:n]

    best_moves = []
    if idx.any():
        best_move_idx = np.unravel_index(idx, move.shape)
        for move_idx in zip(*best_move_idx):
            best_moves.append((move[move_idx], move_idx))
    else:
        raise ValueError

    moves_as_string = []
    for _, move_idx in reversed(sorted(best_moves)):
        src_row = str(abs(move_idx[0] - perspective) + 1)
        src_col = chr(abs(move_idx[1] - perspective) + ord('a'))

        dest_row = str(abs(move_idx[2] - perspective) + 1)
        dest_col = chr(abs(move_idx[3] - perspective) + ord('a'))

        moves_as_string.append(src_col + src_row + dest_col + dest_row)

    return moves_as_string