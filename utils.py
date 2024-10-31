import torch
from torch.utils.data import DataLoader

import pickle
import pathlib
from tqdm import tqdm
from typing import Optional, Union

def save_experiment(experiment: dict, filepath: Union[str, pathlib.PosixPath]) -> None:

    with open(filepath, 'wb') as f:
        pickle.dump(experiment, f)

def evaluate_loss(model: torch.nn.Module,
                  data_loader: DataLoader,
                  criterion,
                  device: torch.device,
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


def complete_mnist(model: torch.nn.Module,
                   data_loader: DataLoader,
                   device: torch.device,
                   n_pixels: int) -> tuple[torch.Tensor]:
    """Given incomplete MNIST pixel sequences generate remaining pixels with RNN."""

    if model.training:
        model.eval()

    n_steps = 784 - n_pixels

    generated_pixels = []
    given_pixels = []
    ground_truth = []
    labels = []

    with torch.no_grad():
        for i, (input, _, label) in tqdm(enumerate(data_loader), total=len(data_loader), desc='Completing MNIST'):

            ground_truth.append(input.squeeze(2))

            input = input[:, :n_pixels, :]
            input = input.to(device)

            given_pixels.append(input)

            generated = model.sample(input, n_steps=n_steps)  # N x n_steps x 1
            generated = generated.squeeze()

            generated_pixels.append(generated)

            label = label.argmax(axis=1)
            label = label.unsqueeze(1)
            labels.append(label)

    given_pixels = torch.cat(given_pixels, dim=0)
    generated_pixels = torch.cat(generated_pixels, dim=0)
    labels = torch.cat(labels, dim=0)
    ground_truth = torch.cat(ground_truth, dim=0)

    return given_pixels, generated_pixels, labels, ground_truth