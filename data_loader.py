import torch
import torchvision
import torchvision.transforms as transforms


NUM_WORKERS = 4


def get_cifar(num_classes=100, dataset_dir="./data", batch_size=128):
    """
    :param num_classes: 10 for cifar10, 100 for cifar100
    :param dataset_dir: location of datasets, default is a directory named "data"
    :param batch_size: batchsize, default to 128
    :return:
    """
    if num_classes == 10:
        # CIFAR10
        print("=> loading CIFAR10...")
        dataset = torchvision.datasets.CIFAR10
        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    else:
        # CIFAR100
        print("=> loading CIFAR100...")
        dataset = torchvision.datasets.CIFAR100
        normalize = transforms.Normalize(
            mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
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

    testset = dataset(root=dataset_dir, train=False,
                      download=True,
                      transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              num_workers=NUM_WORKERS,
                                              pin_memory=True, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             num_workers=NUM_WORKERS,
                                             pin_memory=True, shuffle=False)
    return trainloader, testloader
