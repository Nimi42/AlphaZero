from typing import List, Tuple

import numpy as np

from mcts import MCTS, SearchType


class Agent:

    def __init__(self, s, model):
        self.s = s
        self.model = model

    def create_train_data(self, num_games=100):
        train_data = []

        for i in range(num_games):
            samples = self.play_game()
            train_data.append(samples)

        return train_data

    def play_game(self) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        mcts = MCTS(self.model, 2000, SearchType.AlphaZero)
        train_data = []

        s = self.s
        while s.get_reward() is None:
            root, action = mcts.search(s, train=True)
            s = s.take_action(action)
            print(s)
            probabilities = root.game_state.get_emptyboard()

            for action, child in root.children.items():
                prob = child.N_sa / root.N_sa
                probabilities[action] = prob

            probabilities /= np.sum(probabilities)
            train_data.append((root.game_state, probabilities))
            del root

        reward = s.get_reward()
        train_data = [(i, game.board[None], probs, np.array([reward * game.current_player]))
                      for i, (game, probs) in enumerate(train_data)]

        return train_data


if __name__ == '__main__':
    from util.data import IO
    from tictactoe.game import TicTacToe
    from tictactoe.model import ResNet, BasicBlock

    io = IO('TicTacToe')
    game = TicTacToe()
    model = ResNet(BasicBlock, [2, 2, 2])
    if not io.load_model(model):
        print('Training new model')
    agent = Agent(game, model)
    train_data = agent.create_train_data(num_games=1)

    print(train_data[0][0])
    print('######################')
    print(train_data[0][1])