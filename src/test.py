import chess
from keras.models import load_model

import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

#indices =  np.argpartition(pred.flatten(), -2)[-5:]
#np.vstack(np.unravel_index(indices, pred.shape)).T

from converter import UCItoNetwork, print_human_readable
import numpy as np

np.set_printoptions(edgeitems=10000000, threshold=100, linewidth=np.inf)
np.core.arrayprint._line_width = 10000000

model = load_model('../build/policy.h5')
model.summary()

path = '../resources/pgn_data/stockfish_jonny_2014.pgn'

with open(path) as pgn:
    converter = UCItoNetwork()

    for x, y, turn in converter.generate_main_line(path):
        print_human_readable(x[0] + x[1], None)
        print(converter.get_n_best_moves(y, 2, turn))
        print_human_readable(y, None)

        pred = model.predict(
            [item[None] for item in x],  # np.stack(history)[None],
            batch_size=1)
        print(converter.get_n_best_moves(pred[0], 5, turn))
        print_human_readable(pred[0], None)

