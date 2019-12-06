import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import os
from torch.utils.data import Dataset, TensorDataset
NUM_WORKERS = 4


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return len(self.tensors[0])


def load_new_test_data(version_string="v6"):
    data_path = os.path.join(os.path.dirname(__file__), '.')
    filename = 'cifar10.1_' + version_string
    label_filename = filename + '_labels.npy'
    imagedata_filename = filename + '_data.npy'
    label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
    imagedata_filepath = os.path.abspath(
        os.path.join(data_path, imagedata_filename))
    print('Loading labels from file {}'.format(label_filepath))
    labels = np.load(label_filepath)
    print('Loading image data from file {}'.format(imagedata_filepath))
    imagedata = np.load(imagedata_filepath)

    return imagedata, torch.Tensor(labels).long()


def get_cifar(num_classes=100, dataset_dir="./data", batch_size=128):
    imagedata, labels = load_new_test_data()
    """
      :param num_classes: 10 for cifar10, 100 for cifar100
      :param dataset_dir: location of datasets,
       default is a directory named "data"
      :param batch_size: batchsize, default to 128
      :return:
    """
    if num_classes == 10:
        # CIFAR10
        print("=> loading CIFAR10...")
        dataset = torchvision.datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        # CIFAR100
        print("=> loading CIFAR100...")
        dataset = torchvision.datasets.CIFAR100
        normalize = transforms.Normalize(
            mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = dataset(root=dataset_dir, train=True,
                       download=True,
                       transform=train_transform)

    testset = CustomTensorDataset((imagedata, labels), transform=test_transform)
    # testset = dataset(root=dataset_dir, train=False,
    #                   download=True,
    #                   transform=test_transform)

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size,
                                               num_workers=NUM_WORKERS,
                                               pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size,
                                              num_workers=NUM_WORKERS,
                                              pin_memory=True, shuffle=False)
    return train_loader, test_loader
