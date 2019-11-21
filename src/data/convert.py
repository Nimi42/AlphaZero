import re
import random
import numpy as np
from colorama import Fore, Style

import chess.pgn


class UCItoNetwork:

    def generate_main_line(self, path):
        with open(path) as pgn:
            game = chess.pgn.read_game(pgn)
            board = game.board()
            line = game.mainline_moves()
            del game

            for move in line:
                x = self.create_sample(board)
                y = self._move_as_tensor(move, board.turn)

                yield x, y, board.turn

                board.push(move)

    def generate_random_samples_from_game(self, path):
        with open(path) as pgn:
            game = chess.pgn.read_game(pgn)
            line = list(game.mainline_moves())
            del game

            indices = list(range(1, len(line) + 1))
            random.shuffle(indices)

            # history = [np.zeros((8, 8, 6, 2)) for _ in range(6)]
            while indices:
                move_idx = indices.pop(0)
                board = chess.Board()

                for move in line[:move_idx]:
                    board.push(move)

                x = self.create_sample(board)
                y = self._move_as_tensor(line[move_idx], board.turn)

                yield x, y

                # history.pop(0)
                # history.append(pos)
                # positions.append(pos) #positions.append(np.stack(history))
                # best_moves.append(convert_move(move, colour))
                # print(board)
                # print(move.uci())

            # return np.stack(positions), np.stack(best_moves)

    def create_sample(self, board):
        colour = board.turn
        white, black = self._position_as_tensor(board, colour)

        p1, p2 = [white, black] if colour else [black, white]

        move_count = board.fullmove_number
        no_progress_count = board.halfmove_clock
        repetition = int(board.can_claim_threefold_repetition())
        p1_castlek = int(board.has_kingside_castling_rights(colour))
        p1_castleq = int(board.has_queenside_castling_rights(colour))
        p2_castlek = int(board.has_kingside_castling_rights(not colour))
        p2_castleq = int(board.has_queenside_castling_rights(not colour))
        scalars = np.array([
            int(colour), move_count, no_progress_count, repetition,
            p1_castlek, p1_castleq, p2_castlek, p2_castleq
        ])

        return p1, p2, scalars

    # noinspection PyMethodMayBeStatic
    def _position_as_tensor(self, board, perspective):
        perspective = 0 if perspective else 63
        white, black = np.zeros((8, 8, 6)), np.zeros((8, 8, 6))
        for k, v in board.piece_map().items():
            idx = np.unravel_index(abs(k - perspective), (8, 8))
            board = white if v.color else black
            board[idx[0], idx[1], v.piece_type - 1] = 1

        return white, black

    # noinspection PyMethodMayBeStatic
    def _move_as_tensor(self, move, perspective):
        perspective = 0 if perspective else 7
        notation = move.uci()

        src_row = abs((int(notation[1]) - 1) - perspective)
        src_col = abs((ord(notation[0]) - ord('a')) - perspective)

        dest_row = abs((int(notation[3]) - 1) - perspective)
        dest_col = abs((ord(notation[2]) - ord('a')) - perspective)

        best_move = np.zeros((8, 8, 8, 8))
        best_move[src_row, src_col, dest_row, dest_col] = 1

        return best_move


def print_human_readable(possible_moves, perspective):
    np.set_printoptions(precision=3, threshold=np.nan, linewidth=np.inf)

    if len(possible_moves.shape) == 4:
        if perspective:
            perspective = not perspective
            possible_moves = np.flip(np.flip(possible_moves, axis=perspective), axis=perspective + 2)
        for i in range(8):
            pos_repr = np.array2string(possible_moves[i], sign=' ', threshold=100, max_line_width=np.inf).split('\n')
            rows = [
                ' '.join(pos_repr[::9]) + '  ' + str(8 - i),
                ' '.join(pos_repr[1::9]),
                ' '.join(pos_repr[2::9]),
                ' '.join(pos_repr[3::9]),
                ' '.join(pos_repr[4::9]),
                ' '.join(pos_repr[5::9]),
                ' '.join(pos_repr[6::9]),
                ''.join(pos_repr[7::9]),
                '']
            reorder = '\n'.join(rows).replace('0.', '_')
            highlight_pieces = re.sub(r'([1-9]+)', Fore.RED + r'\1' + Style.RESET_ALL, reorder)
            pretty_print = highlight_pieces
            print(pretty_print)
    else:
        if perspective:
            perspective = not perspective
            possible_moves = np.flip(possible_moves, axis=perspective)
        for i in range(8):
            pos_repr = np.array2string(possible_moves[i], threshold=100, max_line_width=np.inf).split('\n')

            reorder = '\n\n' + ' '.join(pos_repr).replace('0.', '_ ')
            add_legend = reorder + '  ' + Fore.RED + str(8 - i) + Style.RESET_ALL
            highlight_pieces = add_legend.replace('1.', Fore.RED + '1.' + Style.RESET_ALL)
            pretty_print = highlight_pieces

            print(pretty_print)

    if len(possible_moves.shape) == 4:
        sep = 28
    else:
        sep = 20

    legend = ['  ']
    alpha = 'A'
    for i in range(0, 8):
        legend.append(alpha + ' ' * sep)
        alpha = chr(ord(alpha) + 1)

    print(Fore.RED + ''.join(legend), end='')
    print(Style.RESET_ALL + '\n')
    np.set_printoptions(edgeitems=3, infstr='inf',
                        linewidth=75, nanstr='nan', precision=8,
                        suppress=False, threshold=1000, formatter=None)


if __name__ == '__main__':
    np.set_printoptions(edgeitems=10000000, threshold=100, linewidth=np.inf)
    np.core.arrayprint._line_width = 10000000
    some_pgn = '../resources/pgn_data/stockfish_jonny_2014.pgn'

    converter = UCItoNetwork()
    # test = converter.generate_random_samples_from_game(some_pgn)
    # next(test)

    with open(some_pgn) as pgn_file:
        chess_game = chess.pgn.read_game(pgn_file)
        board_state = chess_game.board()

        ml = list(chess_game.mainline())
        print(ml)

        # tail = line
        head, *tail = ml
        print(board_state.fullmove_number)
        board_state.push_uci(head.uci())
        colour = 1
        for mo in tail:
            print(mo.uci())
            p1, p2 = converter._position_as_tensor(board_state, colour)
            print_human_readable(p1 + p2, None)
            best_move = converter._move_as_tensor(mo, colour)

            print_human_readable(best_move, None)

            print(board_state.fullmove_number)
            board_state.push_uci(mo.uci())

            print(board_state)