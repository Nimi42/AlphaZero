import torch
import torch.nn as nn
from copy import deepcopy

from architecture.mcts import MCTS, SearchType, Node, ResultNode
from architecture.policy import ResNet, BasicBlock
from examples.tictactoe import TicTacToe, TicTacToeDummyPolicy, create_policy
from examples.connect4 import Connect4, create_policy
from training.self_play import generate_data
from training.train_model import run_training
from util.data import IO
import util
import examples
import training
import architecture
import pathlib
import ray
from pprint import pprint


def run_mcts():
    game = TicTacToe()
    model = TicTacToeDummyPolicy()
    mcts = MCTS(model, 2000, SearchType.MCTS)

    while game.get_reward() is None:
        node: ResultNode = mcts.search(game)
        game = game.take_action(node.action)
        print(game)

# def eval_alphazero():
#     io = IO('TicTacToe')
#     game = TicTacToe()
#     model = ResNet(BasicBlock, [2, 2, 2])
#
#     if not io.load_model(model):
#         raise FileNotFoundError('No model has been found')
#
#     model.eval()
#     mcts = MCTS(model, 1500, SearchType.AlphaZero)
#
#     with torch.no_grad():
#         while game.get_reward() is None:
#             print(game)
#             root, action = mcts.search(game)
#             game = game.take_action(action)
#
#     print(game)
#     print('End')


def train_alphazero(game, name, model_cls, steps: int):
    io = IO(name)
    model: nn.Module = model_cls('cuda')
    if not io.load_model(model):
        print('Training new model')

    for i in range(steps):
        params = model.state_dict()
        train_data = generate_data(game, model_cls, params, 40)

        io.save_and_split_data(train_data)

        train, valid = io.load_dataset()
        model.train()
        run_training(model, train, valid, batch_size=16, max_epochs=15)

        io.save_model(model)
        io.clear_data()


if __name__ == '__main__':
    runtime_env = {'conda': str(pathlib.Path.cwd().parent / 'configs' / 'env.yml'),
                   'py_modules': [examples, util, training, architecture]}

    ray.init(address='auto', runtime_env=runtime_env)
    pprint(ray.nodes())

    game = Connect4()
    name = 'Connect4'
    model_cls = create_policy
    train_alphazero(game, name, model_cls, 200)
    # eval_alphazero()

    # run_mcts()