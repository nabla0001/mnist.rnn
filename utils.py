import torch
import pickle
import pathlib
from typing import Optional, Union

def save_experiment(experiment: dict, filepath: Union[str, pathlib.PosixPath]) -> None:

    with open(filepath, 'wb') as f:
        pickle.dump(experiment, f)

def evaluate_loss(model,
                  data_loader,
                  criterion,
                  device,
                  num_batches: Optional[int] = None) -> float:

    if model.training:
        model.eval()

    if num_batches is None:
        num_batches = len(data_loader)

    loss = torch.empty(num_batches)

    with torch.no_grad():
        for i in range(num_batches):
            input, target_output, _ = next(iter(data_loader))

            input = input.to(device)
            target_output = target_output.to(device)

            output, _ = model(input)

            loss[i] = criterion(output, target_output).mean()

        return loss.mean().item()