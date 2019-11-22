import sys
import select
from functools import partial
import chess
from keras.models import load_model
from data.convert import UCItoNetwork

# Use Timer to force the engine to return a result
# Might want to put it inside the engine
from data.ml_utils import get_n_best_moves


class UCI:
    UCI = 'uci'
    SET_OPTION = 'setoption'
    UCI_OK = 'uciok'

    NEW_GAME = 'ucinewgame'

    IS_READY = 'isready'
    READY_OK = 'readyok'

    POSITION = 'position'
    STARTPOS = 'startpos'
    MOVES = 'moves'

    GO = 'go'

    QUIT = 'quit'

    def __init__(self):
        self.board = None
        self.moves = None
        self.terminated = False
        self.protocol = {
            UCI.SET_OPTION: lambda x: x,
            UCI.UCI: partial(self.send_response, UCI.UCI_OK),
            UCI.NEW_GAME: self.new_game,
            UCI.IS_READY: partial(self.send_response, UCI.READY_OK),
            UCI.POSITION: self.setup_position,
            UCI.GO: self.calculate,
            UCI.QUIT: self.shutdown
        }

        self.model = load_model('/home/nemo/Downloads/omega_one.h5', compile=False)
        self.converter = UCItoNetwork()

    def read_input(self, stream=sys.stdin, timeout=None):
        """Where stream is a readable file-descriptor, returns
        True if the given stream has input waiting to be read.
        Otherwise returns False.

        The default timeout of zero indicates that the given
        stream should be polled for input and the function
        returns immediately. Increasing the timeout will cause
        the function to wait that many seconds for the poll to
        succeed. Setting timeout to None will cause the function
        to block until it can return True.
        """
        if stream.readable():
            stream = select.select([stream], [], [], timeout)[0][0]

            print(stream, file=sys.stderr)
            line = stream.readline()
            line = line.strip()

            print(line, file=sys.stderr)
            sys.stderr.flush()

            uci_command, *params = line.split()

            func = self.protocol[uci_command]
            if params:
                func(params)
            else:
                func()

        else:
            raise IOError

    def send_response(self, msg):
        msg = msg + '\n'
        sys.stdout.write(msg)
        sys.stdout.flush()

    def new_game(self):
        """
        Send optional engine parameters.

        :return: None
        """
        self.board = chess.Board()
        self.moves = []

    def next_position(self, fen):
        pass

    def setup_position(self, args=None):
        assert args is not None

        command, *args = args

        if command.lower() == UCI.STARTPOS:
            if args:
                command, *moves = args
                if command.lower() == UCI.MOVES:
                    if self.moves == moves[:-1]:
                        self.board.push(chess.Move.from_uci(moves[-1]))
                    else:
                        self.board = chess.Board()
                        for move in moves:
                            self.board.push(chess.Move.from_uci(move))

                    self.moves = moves
            else:
                self.board = chess.Board()

    def calculate(self, args):
        params_dict = self.unravel_args(args)

        x = self.converter.create_sample(self.board)
        pred = self.model.predict(
            [item[None] for item in x],  # np.stack(history)[None],
            batch_size=1)

        moves = get_n_best_moves(pred[0], 5, self.board.turn)

        for move in moves:
            if self.board.is_legal(chess.Move.from_uci(move)):
                self.send_response('bestmove ' + move)

    def shutdown(self):
        self.terminated = True

    def unravel_args(self, args):
        keys = args[::2]
        values = args[1::2]

        return dict(zip(keys, values))


connection = UCI()
while not connection.terminated:
    connection.read_input()

