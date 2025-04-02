import torch
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class CircleData(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


def get_fmnist_loaders(data_dir, batch_size, shuffle=True):
    """Helper function that deserializes FashionMNIST data
    and returns the relevant data loaders.

    params:
        data_dir:    string - root directory where the data will be saved
        b_sz:        integer - the batch size
        shuffle:     boolean - whether to shuffle the training set or not
    """
    train_loader = DataLoader(
        FashionMNIST(data_dir, train=True, transform=ToTensor(), download=True),
        shuffle=shuffle,
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        FashionMNIST(data_dir, train=False, transform=ToTensor(), download=True),
        shuffle=False,
        batch_size=batch_size,
    )

    return train_loader, test_loader


def get_mnist_loaders(data_dir, b_sz, shuffle=True):
    """Helper function that deserializes MNIST data
    and returns the relevant data loaders.

    params:
        data_dir:    string - root directory where the data will be saved
        b_sz:        integer - the batch size
        shuffle:     boolean - whether to shuffle the training set or not
    """
    train_loader = DataLoader(
        MNIST(data_dir, train=True, transform=ToTensor(), download=True),
        shuffle=shuffle,
        batch_size=b_sz,
    )
    test_loader = DataLoader(
        MNIST(data_dir, train=False, transform=ToTensor(), download=True),
        shuffle=False,
        batch_size=b_sz,
    )

    return train_loader, test_loader


def get_kmnist_loaders(data_dir, b_sz, shuffle=True):
    """Helper function that deserializes KMNIST data
    and returns the relevant data loaders.

    params:
        data_dir:    string - root directory where the data will be saved
        b_sz:        integer - the batch size
        shuffle:     boolean - whether to shuffle the training set or not
    """
    train_loader = DataLoader(
        KMNIST(data_dir, transform=ToTensor(), download=True),
        shuffle=shuffle,
        batch_size=b_sz,
    )
    test_loader = DataLoader(
        KMNIST(data_dir, train=False, transform=ToTensor(), download=True),
        shuffle=False,
        batch_size=b_sz,
    )

    return train_loader, test_loader


def get_circle_loaders(data_dir, b_sz, shuffle=True):
    train_loader = DataLoader(
        CircleData(data_dir + "circle_train.ptc"), batch_size=b_sz, shuffle=True
    )
    test_loader = DataLoader(
        CircleData(data_dir + "circle_test.ptc"), batch_size=b_sz, shuffle=False
    )

    return train_loader, test_loader


def embed_data(model, data_loader):
    model.eval()
    all_z = []
    all_q_mu = []
    all_q_t = []
    all_labels = []
    with torch.no_grad():
        for data, labels in tqdm(data_loader):
            z, q_mu, q_t = model.encode(data.view(-1, data.shape[-1] * data.shape[-2]))
            all_z.append(z)
            all_q_mu.append(q_mu)
            all_q_t.append(q_t)
            all_labels.append(labels)

    all_z = torch.cat(all_z)
    all_q_mu = torch.cat(all_q_mu)
    all_q_t = torch.cat(all_q_t)
    all_labels = torch.cat(all_labels)
    return all_z, all_q_mu, all_q_t, all_labels
