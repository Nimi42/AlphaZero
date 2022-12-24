import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset

# Name of the Project
_PROJECT_NAME: str = 'AlphaZero'

# Paths to the directory where training/validation data is saved
_DATA_DIR: Path = Path(__file__)
while _DATA_DIR.stem != _PROJECT_NAME:
    try:
        _DATA_DIR = _DATA_DIR.parents[0]
    except IndexError:
        _DATA_DIR = Path(__file__)
        break
        #raise NotADirectoryError('Project root - ' + _PROJECT_NAME + ' - does not exist')
_DATA_DIR /= 'data'

# Numpy data saved as .npy with following keys
_BOARD: str = 'board'
_PREDICTION: str = 'prediction'
_EVALUATION: str = 'evaluation'


class Dataset(TorchDataset):

    def __init__(self, dir_name: Path):
        self.files = list(dir_name.iterdir())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = str(self.files[index])
        sample = np.load(filename)

        return torch.from_numpy(sample[_BOARD]).float(), \
               torch.from_numpy(sample[_PREDICTION]).float(), \
               torch.from_numpy(sample[_EVALUATION]).float()


class IO:

    def __init__(self, game_name):
        print('Data is located at ' + str(_DATA_DIR / game_name))

        self.TRAIN_DIR: Path = _DATA_DIR / game_name / 'train'
        self.VALID_DIR: Path = _DATA_DIR / game_name / 'valid'
        self.MODEL_NAME: Path = _DATA_DIR / game_name / 'best_model'

    def save_and_split_data(self,
                            samples: List[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
                            split=0.8
                            ) -> None:
        samples = [(i, move_nr, s, P_sa, v) for i, game in enumerate(samples) for move_nr, (s, P_sa, v) in
                   enumerate(game)]
        random.shuffle(samples)
        train_split = int(len(samples) * split)

        self._save_samples(self.TRAIN_DIR, samples[:train_split])
        self._save_samples(self.VALID_DIR, samples[train_split:])

    def _save_samples(self,
                      directory: Path, samples: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]
                      ) -> None:
        for game_nr, move_nr, s, p_sa, value in samples:
            file = 'game_' + f'{game_nr:004}' + '_move_' + f'{move_nr:004}'
            directory.mkdir(parents=True, exist_ok=True)

            np.savez(directory / file, **{_BOARD: s, _PREDICTION: p_sa, _EVALUATION: value})

    def clear_data(self):
        for file in self.TRAIN_DIR.iterdir():
            file.unlink()
        for file in self.VALID_DIR.iterdir():
            file.unlink()

    def save_model(self,
                   model: nn.Module
                   ) -> None:
        torch.save(model.state_dict(), self.MODEL_NAME)

    def load_model(self,
                   model: nn.Module
                   ) -> bool:
        try:
            model_state = torch.load(self.MODEL_NAME)
            model.load_state_dict(model_state)
        except FileNotFoundError:
            return False
        return True

    def load_dataset(self) -> Tuple[Dataset, Dataset]:
        return Dataset(self.TRAIN_DIR), \
               Dataset(self.VALID_DIR)


def print_one_line(strings):
    split_strings = [x.split('\n') for x in strings]
    result = ''

    for x in zip(*split_strings):
        for y in x:
            result += y + ' ' * 5
        result += '\n'
    result += '-----------------------\n'
    print(result, flush=True)
