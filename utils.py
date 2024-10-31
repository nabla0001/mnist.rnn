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


def complete_mnist(model, data_loader, device, n_pixels):
    """Given incomplete MNIST pixel sequences generate remaining pixels with RNN."""

    if model.training:
        model.eval()

    n_steps = 784 - n_pixels

    generated_pixels = []
    given_pixels = []

    with torch.no_grad():
        for i, (input, _, _) in enumerate(data_loader):
            input = input[:, :n_pixels, :]
            input = input.to(device)

            given_pixels.append(input)

            generated = model.sample(input, n_steps=n_steps)  # N x n_steps x 1
            generated = generated.squeeze()

            generated_pixels.append(generated)

    given_pixels = torch.cat(given_pixels, dim=0)
    generated_pixels = torch.cat(generated_pixels, dim=0)

    return given_pixels, generated_pixels