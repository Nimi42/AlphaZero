import itertools
import time
from pprint import pprint

import numpy as np
import ray, torch
import torch.nn as nn
from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy, remove_placement_group
from ray.util.queue import Queue
from copy import deepcopy

from architecture.mcts import MCTS, SearchType
from util.data import print_one_line


def generate_data(s, model_cls, params, num_games):
    bundles = []
    tasks = []

    # Create a placement group.
    gpu_decorator = ray.remote(num_cpus=4, num_gpus=1)
    cpu_decorator = ray.remote(num_cpus=4)
    for node in ray.nodes():
        if node['Alive']:
            if 'GPU' in node['Resources']:
                tasks.append(gpu_decorator(Worker))
                bundles.append({'CPU': 4, 'GPU': 1})
            else:
                bundles.append({'CPU': 4})
                tasks.append(cpu_decorator(Worker))
    print(bundles)
    pg = placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    task_queue = Queue(maxsize=num_games)
    for i in range(num_games):
        task_queue.put(i)

    for i, t in enumerate(tasks):
        tasks[i] = t\
            .options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg))\
            .remote(s, model_cls, params, task_queue)

    start = time.time()
    result = ray.get([t.run.remote() for t in tasks])
    print(time.time() - start)
    remove_placement_group(pg)
    return itertools.chain(*result)


class Worker:
    def __init__(self, s, model_cls, params, task_queue: Queue):
        self.s = deepcopy(s)
        self.task_queue = task_queue

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model: nn.Module = model_cls(device)
        self.model.load_state_dict(params)
        self.model.to(device)

    def run(self):
        self.model.eval()
        with torch.no_grad():
            mcts = MCTS(self.model, 2000, SearchType.AlphaZero)
            result = []

            while not self.task_queue.empty():
                item = self.task_queue.get(block=True)

                train_data = []
                print_moves = []

                s = self.s
                while s.get_reward() is None:
                    root = mcts.search(s, train=True)

                    s = s.take_action(root.action)
                    print_moves.append(str(s))
                    probabilities = root.game_state.get_emptyboard()

                    for child in root.children:
                        prob = child.N_sa / root.N_sa
                        probabilities[child.action] = prob

                    probabilities /= np.sum(probabilities)
                    train_data.append((root.game_state, probabilities))
                    del root

                reward = s.get_reward()
                train_data = [(game.board[None], probs, np.array([reward * game.current_player]))
                              for game, probs in train_data]
                print_one_line(print_moves)

                result.append(train_data)
            return result
