import os
import sys
import numpy as np
from math import floor
import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def splitset(key, dataset, parts):
    """Partition data into "parts" partitions"""
    n = dataset.shape[0]
    local_n = floor(n / parts)
    print(key)
    print("Original shape : ", dataset.shape)
    arr = np.array(np.split(dataset, parts))
    print("Splitted shape : ", arr.shape)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n: (i + 1) * local_n])
    return np.array(result)


def create_cifar10_partitions(nr_of_datasets):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=50000, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=10000, shuffle=True, pin_memory=True)
    data = {}
    for x, y in train_loader:
        data["x_train"], data["y_train"] = x.numpy(), y.numpy()
    for x, y in val_loader:
        data["x_test"], data["y_test"] = x.numpy(), y.numpy()
    print("CREATING {} PARTITIONS INSIDE {}/data/clients".format(nr_of_datasets, os.getcwd()))
    for key, val in data.items():
        data[key] = splitset(key, val, nr_of_datasets)
    if not os.path.exists(os.getcwd() + '/data/clients'):
        os.mkdir(os.getcwd() + '/data/clients')
    for i in range(nr_of_datasets):
        if not os.path.exists('data/clients/{}'.format(str(i + 1))):
            os.mkdir('data/clients/{}'.format(str(i + 1)))

        np.savez('data/clients/{}'.format(str(i + 1)) + '/data.npz',
                 x_train=data['x_train'][i],
                 y_train=data['y_train'][i],
                 x_test=data['x_test'][i],
                 y_test=data['y_test'][i])
    print("DONE")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        nr_of_datasets = 10
    else:
        nr_of_datasets = int(sys.argv[1])
    create_cifar10_partitions(nr_of_datasets)
