import pickle
import pathlib
from typing import Optional, Union

def save_experiment(experiment: dict, filepath: Union[str, pathlib.PosixPath]) -> None:

    with open(filepath, 'wb') as f:
        pickle.dump(experiment, f)