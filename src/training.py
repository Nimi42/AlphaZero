from nn import chess_conv
from data import data_generator


def train(model, gen):
    model.fit_generator(gen, epochs=50, steps_per_epoch=120 * 121 // 64)


if __name__ == '__main__':
    path = "/home/nemo/Downloads/stockfish_jonny_2014.pgn"

    gen = data_generator('../../resources/pgn_data', 64)
    model = chess_conv()

    train(model, gen)

    model.save('/home/nemo/Downloads/omega_one.h5')
