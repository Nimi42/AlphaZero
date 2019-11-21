import os
import numpy as np
import random

from converter import UCItoNetwork


def data_generator(path, batch_size):
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
