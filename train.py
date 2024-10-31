import torch
from torchinfo import summary
from model import RNN
from data import mnist_data_loaders
from utils import save_experiment

from datetime import datetime
import argparse
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trains RNN to generate MNIST images.')
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--exp-dir', type=str, default='experiments')
    parser.add_argument('--data-dir', type=str, default='data')
    args = parser.parse_args()
    print(args)

    # experiment & checkpoint tracking
    exp_path = Path(args.exp_dir) / args.exp_name
    exp_path.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_path = exp_path / (timestamp + '.pkl')
    checkpoint_path = exp_path / (timestamp + '.ckpt')

    print(f'experiment: {args.exp_name}')
    print(f'exp dir: {exp_path}')
    print(f'results path: {experiment_path}')
    print(f'checkpoint path: {checkpoint_path}')

    # device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'device: {device}')

    # hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 20
    momentum = 0.9
    weight_decay = 1e-4

    # data
    train_loader, val_loader, test_loader = mnist_data_loaders(batch_size, data_dir=args.data_dir)

    # model
    model = RNN(args.hidden_size)

    print(summary(model, (batch_size, 784, 1), verbose=2))
    model.to(device)

    experiment = {
        'name': args.exp_name,
        'train_loss': [],
        'val_loss': [],
        'test_loss': None,
        'batch': [],
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'momentum': momentum,
            'weight_decay': weight_decay,
        },
        'args': args
    }

    # loss & optimiser
    criterion = torch.nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train
    n_batches = len(train_loader)
    total_batch_count = 0
    stop_training = False

    for epoch in range(num_epochs):
        for i, (input, target_output, _) in enumerate(train_loader):
            total_batch_count += 1

            input = input.to(device)
            target_output = target_output.to(device)

            output, _ = model(input)

            loss = criterion(output, target_output).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch:04d}/{num_epochs:04d}]\tBatch [{total_batch_count:06d}]\tLoss: {loss.item():.4f}')

        # evaluate train/val error
        # model.eval()
        # TODO
        # model.train()

        # experiment['train_error'].append(train_error)
        # experiment['val_error'].append(val_error)
        # experiment['batch'].append(total_batch_count)


    # evaluate model
    # TODO

    # save results
    save_experiment(experiment, experiment_path)

    # Save the model checkpoint
    torch.save(model.state_dict(), checkpoint_path)