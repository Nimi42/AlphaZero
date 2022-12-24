import numpy as np
from tabulate import tabulate

from architecture.policy import ResNet, BasicBlock


def create_policy(device):
    return ResNet(BasicBlock, [2, 2, 2], 7, device)


class Connect4DummyPolicy:

    def predict(self, game_state):
        predictions = np.count_nonzero(game_state.board, axis=0).astype(float)
        mask_zero = predictions < 6
        probability = 1 / np.count_nonzero(mask_zero)
        predictions[mask_zero] = probability
        predictions[~mask_zero] = 0
        for i in range(predictions.shape[0]):
            if predictions[0] > 0:
                yield i, predictions[0]
            predictions = np.delete(predictions, 0)

    def get_best_action(self, game_state):
        moves = np.count_nonzero(game_state.board, axis=0).astype(float)
        moves = np.argwhere(moves < 6).flatten()
        idx = np.random.choice(moves.shape[0], 1)[0]
        return moves[idx]


class Connect4:
    _ROWS = []
    for i in range(6):
        y = []
        for j in range(7):
            y.append(2**((7 * i) + 6 + i - j))
        _ROWS.insert(0, y)
    _ROWS = np.array(_ROWS)

    _COLS = []
    for i in range(7):
        y = []
        for j in range(6):
            y.append(2**((7 * i) + 5 + i - j))
        _COLS.insert(0, y)
    _COLS = np.array(_COLS)

    _DIAG1 = np.array(
        [[0x00020000000, 0x00000200000, 0x00000001000, 0x00000000008, 0,             0,             0],
         [0x01000000000, 0x00010000000, 0x00000100000, 0x00000000800, 0x00000000004, 0,             0],
         [0x80000000000, 0x00800000000, 0x00008000000, 0x00000080000, 0x00000000400, 0x00000000002, 0],
         [0,             0x40000000000, 0x00400000000, 0x00004000000, 0x00000040000, 0x00000000200, 0x1],
         [0,             0,             0x20000000000, 0x00200000000, 0x00002000000, 0x00000020000, 0x100],
         [0,             0,             0,             0x10000000000, 0x00100000000, 0x00001000000, 0x10000]
        ]
    )

    _DIAG2 = np.array(
        [[0,             0,             0,             0x10000000000, 0x00100000000, 0x00001000000, 0x10000],
         [0,             0,             0x20000000000, 0x00200000000, 0x00002000000, 0x00000020000, 0x100],
         [0,             0x40000000000, 0x00400000000, 0x00004000000, 0x00000040000, 0x00000000200, 0x1],
         [0x80000000000, 0x00800000000, 0x00008000000, 0x00000080000, 0x00000000400, 0x00000000002, 0],
         [0x01000000000, 0x00010000000, 0x00000100000, 0x00000000800, 0x00000000004, 0,             0],
         [0x00020000000, 0x00000200000, 0x00000001000, 0x00000000008, 0,             0,             0]
        ]
    )

    _FULL_BOARD = 0x7f7f7f7f7f7f

    def __init__(self, board=np.zeros((6, 7)), current_player=1, p1_bits=(0, 0, 0, 0), p2_bits=(0, 0, 0, 0)):
        self.board = board
        self.current_player = current_player
        self.p1_bits = p1_bits
        self.p2_bits = p2_bits

    @classmethod
    def get_emptyboard(cls):
        return np.zeros(7)

    def take_action(self, action):
        new_board = self.board.copy()
        row = 5 - np.count_nonzero(new_board, axis=0)[action]
        if row < 0:
            raise ValueError('Bord is full')
        new_board[row, action] = 1

        return Connect4(new_board * -1, self.current_player * -1,
                        self.p2_bits,
                        (self.p1_bits[0] | Connect4._ROWS[row, action],
                         self.p1_bits[1] | Connect4._COLS[action, row],
                         self.p1_bits[2] | Connect4._DIAG1[row, action],
                         self.p1_bits[3] | Connect4._DIAG2[row, action]))

    def get_valid_moves(self):
        return self.board[0] == 0

    def get_reward(self):
        rows = self.p2_bits[0]
        cols = self.p2_bits[1]
        diag1 = self.p2_bits[2]
        diag2 = self.p2_bits[3]
        if rows & rows >> 1 & rows >> 2 & rows >> 3\
                or cols & cols >> 1 & cols >> 2 & cols >> 3\
                or diag1 & diag1 >> 1 & diag1 >> 2 & diag1 >> 3\
                or diag2 & diag2 >> 1 & diag2 >> 2 & diag2 >> 3:
            return -self.current_player
        elif not (rows | self.p1_bits[0]) ^ Connect4._FULL_BOARD:
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
