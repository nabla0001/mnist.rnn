import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset


class MNISTPixelData(Dataset):
    def __init__(self, data: Dataset):
        """Custom dataset for MNIST prediction task

        Each sample consists of
            input: flattened image 1x783 (ignore last pixel since no target exists)
            target output: flattened image shifted by 1 pixel
            digit class: 1-hot encoded class
        """
        super().__init__()

        self.data = []

        for image, label in data:
            input_pixels = image.reshape(784, -1)
            target_pixels = input_pixels[1:]  # 1x783, input shifted by 1 pixel
            input_pixels = input_pixels[:-1]  # 1x783, ignore last pixel since no target exists

            # 1-hot encode digit class
            label = torch.tensor(label, dtype=torch.long)
            label = torch.nn.functional.one_hot(label, num_classes=10)

            self.data.append((input_pixels, target_pixels, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def mnist_data_loaders(batch_size: int = 128,
                       data_dir: str = 'data',
                       binarise=True) -> tuple[DataLoader, DataLoader, DataLoader]:

    if binarise:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda image: (image > 0.1).to(torch.float32))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    train = MNIST(root=data_dir, train=True, download=True, transform=transform)
    test = MNIST(root=data_dir, train=False, download=True, transform=transform)

    # convert to sequence dataset
    train = MNISTPixelData(train)
    test = MNISTPixelData(test)

    # 55k/5k train/val split
    n_train = 55000
    idx = torch.randperm(len(train))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    val = torch.utils.data.Subset(train, val_idx)
    train = torch.utils.data.Subset(train, train_idx)

    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader