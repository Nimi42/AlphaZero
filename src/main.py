from multiprocessing import set_start_method

import torch
import torch.nn as nn

from architecture.mcts import MCTS, SearchType, Node
from architecture.model import ResNet, BasicBlock
from examples.tictactoe import TicTacToe, TicTacToeDummyPolicy, create_policy
from training.self_play import generate_data
from training.train_model import run_training
from util.data import IO


def run_mcts():
    game = TicTacToe()
    model = TicTacToeDummyPolicy()
    mcts = MCTS(model, 100, SearchType.MCTS)
    node = Node(0, game)

    while node.game_state.get_reward() is None:
        print(node.game_state)
        prev, node = mcts.search(node)

    print(node.game_state)
    print('End')


def eval_alphazero():
    io = IO('TicTacToe')
    game = TicTacToe()
    model = ResNet(BasicBlock, [2, 2, 2])

    if not io.load_model(model):
        raise FileNotFoundError('No model has been found')

    model.eval()
    mcts = MCTS(model, 1500, SearchType.AlphaZero)

    with torch.no_grad():
        while game.get_reward() is None:
            print(game)
            root, action = mcts.search(game)
            game = game.take_action(action)

    print(game)
    print('End')


def train_alphazero(game, name, model_cls, steps: int):
    io = IO(name)
    model: nn.Module = model_cls('cuda')
    if not io.load_model(model):
        print('Training new model')

    for i in range(steps):
        params = model.state_dict()
        train_data = generate_data(game, model_cls, params, 16)

        io.save_and_split_data(train_data)

        train, valid = io.load_dataset()
        model.train()
        run_training(model, train, valid, batch_size=16, max_epochs=5)

        io.save_model(model)
        io.clear_data()


if __name__ == '__main__':
    set_start_method('spawn')

    game = TicTacToe()
    name = 'TicTacToe'
    model_cls = create_policy
    train_alphazero(game, name, model_cls, 100)
    # eval_alphazero()
