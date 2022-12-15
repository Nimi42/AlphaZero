import time
from multiprocessing import Process, Queue

import numpy as np
import torch
import torch.nn as nn

from architecture.mcts import MCTS, SearchType
from util.data import print_one_line


def generate_data(s, model_cls, params, num_games):
    task_queue = Queue()
    for i in range(num_games):
        task_queue.put(i)

    return_queue = Queue()
    workers = []

    device = 'cpu'
    for i in range(torch.cuda.device_count() + 1):
        task_queue.put(None)

        model: nn.Module = model_cls(device)
        model.load_state_dict(params)
        model.to(device)

        worker = Worker(s, model, task_queue, return_queue)
        worker.start()
        workers.append(worker)

        device = 'cuda:' + str(i)

    start = time.time()

    res = []
    for i in range(num_games):
        res.append(return_queue.get())

    for worker in workers:
        worker.join()

    print(time.time() - start)
    return res


class Worker(Process):

    def __init__(self, s, model, task_queue: Queue, return_queue: Queue):
        super().__init__(daemon=True)
        self.s = s
        self.model = model

        self.task_queue = task_queue
        self.return_queue = return_queue

    def run(self) -> None:
        if self.model.device == 'cpu':
            torch.set_num_threads(1)

        self.model.eval()
        with torch.no_grad():
            mcts = MCTS(self.model, 1000, SearchType.AlphaZero)

            while True:
                item = self.task_queue.get(block=False)
                if item is None:
                    print(self.model.device)
                    break

                train_data = []
                print_moves = []

                s = self.s
                while s.get_reward() is None:
                    root, action = mcts.search(s, train=True)

                    s = s.take_action(action)
                    print_moves.append(str(s))
                    probabilities = root.game_state.get_emptyboard()

                    for action, child in root.children.items():
                        prob = child.N_sa / root.N_sa
                        probabilities[action] = prob

                    probabilities /= np.sum(probabilities)
                    train_data.append((root.game_state, probabilities))
                    del root

                reward = s.get_reward()
                train_data = [(game.board[None], probs, np.array([reward * game.current_player]))
                              for game, probs in train_data]
                print_one_line(print_moves)

                self.return_queue.put(train_data)
