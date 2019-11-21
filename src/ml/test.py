from keras.models import load_model
from data.convert import UCItoNetwork, print_human_readable
from data.ml_utils import get_n_best_moves

model = load_model('../build/policy_net.h5')
model.summary()

path = '../resources/pgn_data/stockfish_jonny_2014.pgn'

with open(path) as pgn:
    converter = UCItoNetwork()

    for x, y, turn in converter.generate_main_line(path):
        print_human_readable(x[0] + x[1], None)
        print(get_n_best_moves(y, 2, turn))
        print_human_readable(y, None)

        pred = model.predict(
            [item[None] for item in x],
            batch_size=1)
        print(get_n_best_moves(pred[0], 5, turn))
        print_human_readable(pred[0], None)

