from ml.nn import chess_conv
from data.ml_utils import batch_generator


def train(model, gen):
    model.fit_generator(gen, epochs=1, steps_per_epoch=120 * 121 // 64)


if __name__ == '__main__':
    path = "../resources/pgn_data/stockfish_jonny_2014.pgn"

    gen = batch_generator('../resources/pgn_data', 64)
    model = chess_conv()

    train(model, gen)

    model.save('../build/policy_net.h5')
