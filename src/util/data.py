import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset
import numpy as np

import torchvision.transforms.functional as TF
import random
from typing import Sequence


class RotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class IO:
    # Name of the Project
    PROJECT_NAME: Path = 'AlphaZero'

    # Paths to the directory where training/validation data is saved
    DATA_DIR: Path = Path(__file__)
    while DATA_DIR.stem != PROJECT_NAME:
        try:
            DATA_DIR = DATA_DIR.parents[0]
        except IndexError:
            raise NotADirectoryError('Project root - ' + PROJECT_NAME + ' - does not exist')
    DATA_DIR /= 'data'
    TRAIN_DIR: Path = None
    VALID_DIR: Path = None

    # Path to the parameters of the best model
    MODEL_NAME: Path = None

    def __init__(self, game_name):
        print('Data is located at ' + str(IO.DATA_DIR / game_name))

        IO.TRAIN_DIR = IO.DATA_DIR / game_name / 'train'
        IO.VALID_DIR = IO.DATA_DIR / game_name / 'valid'
        IO.MODEL_NAME = IO.DATA_DIR / game_name / 'best_model'

    def save_and_split_data(self,
                            samples: List[List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]],
                            split=0.8
                            ) -> None:
        samples = [(i, move_nr, s, P_sa, v) for i, game in enumerate(samples) for move_nr, s, P_sa, v in game]
        random.shuffle(samples)
        train_split = int(len(samples) * split)

        self._save_samples(IO.TRAIN_DIR, samples[:train_split])
        self._save_samples(IO.VALID_DIR, samples[train_split:])

    def _save_samples(self,
                      directory: Path, samples: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]
                      ) -> None:
        for game_nr, move_nr, s, p_sa, value in samples:
            file = 'game_' + f'{game_nr:004}' + '_move_' + f'{move_nr:004}'
            directory.mkdir(parents=True, exist_ok=True)

            np.savez(directory / file, **{IO._BOARD: s, IO._PREDICTION: p_sa, IO._EVALUATION: value})

    def clear_data(self):
        for file in IO.TRAIN_DIR.iterdir():
            file.unlink()
        for file in IO.VALID_DIR.iterdir():
            file.unlink()

    def save_model(self,
                   model: nn.Module
                   ) -> None:
        torch.save(model.state_dict(), IO.MODEL_NAME)

    def load_model(self,
                   model: nn.Module
                   ) -> bool:
        try:
            model_state = torch.load(IO.MODEL_NAME)
            model.load_state_dict(model_state)
        except FileNotFoundError:
            return False
        return True

    def load_dataset(self) -> Tuple['IO.Dataset', 'IO.Dataset']:
        return self.Dataset(IO.TRAIN_DIR),\
               self.Dataset(IO.VALID_DIR)

    # Numpy data saved as .npy with following keys
    _BOARD = 'board'
    _PREDICTION = 'prediction'
    _EVALUATION = 'evaluation'

    class Dataset(TorchDataset):

        def __init__(self, dir_name: Path):
            self.files = list(dir_name.iterdir())

        def __len__(self):
            return len(self.files)

        def __getitem__(self, index):
            filename = str(self.files[index])
            sample = np.load(filename)

            return torch.from_numpy(sample[IO._BOARD]).float(), \
                torch.from_numpy(sample[IO._PREDICTION]).float(), \
                torch.from_numpy(sample[IO._EVALUATION]).float()


# if __name__ == '__main__':
    # # d = Dataset(TRAIN_DIR)
    # # print(d[0])
    # p = Path(TRAIN_DIR)
    # for f in p.iterdir():
    #     x = np.load(f)
    #     print(x['game'])
    #     print(x['predictions'])