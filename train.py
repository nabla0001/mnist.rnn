import torch
from torchinfo import summary
from model import RNN
from data import mnist_data_loaders
from utils import save_experiment, evaluate_loss

from datetime import datetime
import argparse
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trains RNN to generate MNIST images.')
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--checkpoint', type=str, default=None)
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

    # resume training if specified
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(Path(args.checkpoint), weights_only=True))
        model.train()

        print(f'Resuming training from checkpoint {args.checkpoint}')

    print(summary(model, (batch_size, 784, 1), verbose=2))
    model.to(device)

    experiment = {
        'name': args.exp_name,
        'loss': [],
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

            experiment['loss'].append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch:04d}/{num_epochs:04d}]\tBatch [{total_batch_count:06d}]\tLoss: {loss.item():.4f}')

        # evaluate train/val error
        model.eval()

        val_loss = evaluate_loss(model, val_loader, criterion, device, 10)
        train_loss = evaluate_loss(model, train_loader, criterion, device, 10)

        experiment['val_loss'].append(val_loss)
        experiment['train_loss'].append(train_loss)
        experiment['batch'].append(total_batch_count)

        print()
        print(f'Epoch [{epoch:04d}/{num_epochs:04d}]\tBatch [{total_batch_count:06d}]\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}')
        print()

        model.train()

        # save model checkpoint
        torch.save(model.state_dict(), checkpoint_path)

    # evaluate model
    # TODO

    # save results
    save_experiment(experiment, experiment_path)
