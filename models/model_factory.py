import models


def is_resnet(name):
    """
    Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
    :param name:
    :return:
    """
    name = name.lower()
    return name.startswith('resnet')


model_dict = {
    "WRN10_4": models.WRN10_4,
    "WRN16_1": models.WRN16_1,
    "WRN16_2": models.WRN16_2,
    "WRN16_4": models.WRN16_4,
    "WRN16_8": models.WRN16_8,
    "WRN28_2": models.WRN28_2,
    "WRN22_4": models.WRN22_4,
    "WRN22_8": models.WRN22_8,
    "WRN28_1": models.WRN28_1,
    "WRN10_1": models.WRN10_1,
    "WRN40_1": models.WRN40_1,
    "WRN40_4": models.WRN40_4,
    "resnet18": models.resnet18,
    "resnet20": models.resnet20_cifar,
    "resnet32": models.resnet32_cifar,
    "resnet44": models.resnet44_cifar,
    "resnet56": models.resnet56_cifar,
    "resnet110": models.resnet110_cifar,
    "resnet1202": models.resnet1202_cifar,
    "resnet164": models.resnet164_cifar,
    "resnet1001": models.resnet1001_cifar,
    "preact_resnet110": models.preact_resnet110_cifar,
    "preact_resnet164": models.preact_resnet164_cifar,
    "preact_resnet1001": models.preact_resnet1001_cifar,
}


def create_cnn_model(name, dataset="cifar100", use_cuda=False):
    """
    Create a student for training, given student name and dataset
    :param name: name of the student. e.g., resnet110, resnet32, plane2, plane10, ...
    :param dataset: the dataset which is used to determine last layer's output size. Options are cifar10 and cifar100.
    :return: a pytorch student for neural network
    """
    num_classes = 100 if dataset == 'cifar100' else 10
    resnet_model = model_dict[name](num_classes=num_classes)
    model = resnet_model

    # copy to cuda if activated
    if use_cuda:
        model = model.cuda()

    return model
