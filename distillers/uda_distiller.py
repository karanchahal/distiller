import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as torch_func
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from trainer import KDTrainer


class CIFAR10Policy():
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.
        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2,
                 magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            new_img = Image.new("RGBA", rot.size, (128,) * 4)
            composite = Image.composite(rot, new_img, rot)
            return composite.convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude *
                                         random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude *
                                         random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude *
                                         img.size[0] *
                                         random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude *
                                         img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class UDADataset(Dataset):

    def __init__(self, dataset, normalize, transform=None):
        self.dataset = dataset
        self.transform = transform

        self.aug_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, target = self.dataset[idx]
        aug_x = self.aug_tf(x)
        if self.transform:
            x = self.transform(x)

        return x, aug_x, target


class UDATrainer(KDTrainer):
    def __init__(self, s_net, t_net, config):
        super(UDATrainer, self).__init__(s_net, t_net, config)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net
        self.kd_fun = nn.KLDivLoss(size_average=False)

    def train_single_epoch(self, t_bar):
        self.net.train()
        total_correct = 0.0
        total_loss = 0.0
        len_train_set = len(self.train_loader.dataset)
        for batch_idx, (x, x_aug, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            x_aug = x_aug.to(self.device)
            self.optimizer.zero_grad()

            # this function is implemented by the subclass
            y_hat, loss = self.calculate_loss(x, x_aug, y)

            # Metric tracking boilerplate
            pred = y_hat.data.max(1, keepdim=True)[1]
            total_correct += pred.eq(y.data.view_as(pred)).sum()
            total_loss += loss
            curr_acc = 100.0 * (total_correct / float(len_train_set))
            curr_loss = (total_loss / float(batch_idx))
            t_bar.update(self.batch_size)
            t_bar.set_postfix_str(f"Acc {curr_acc:.3f}% Loss {curr_loss:.3f}")
        total_acc = float(total_correct / len_train_set)
        return total_acc

    def uda_loss(self, n_out, aug_out):
        batch_size = n_out.shape[0]
        n_out = torch_func.log_softmax(n_out, dim=1)
        aug_out = torch_func.softmax(aug_out, dim=1)
        return self.kd_fun(n_out, aug_out) / batch_size

    def calculate_loss(self, data, aug_data, target):
        #print(data.size())
        out_s = self.s_net(data)
        out_t = self.t_net(data)
        #print(aug_data.size())
        out_aug_s = self.s_net(aug_data)
        out_aug_t = self.t_net(aug_data)

        s_cifar = out_s[target != -1]
        t_cifar = out_t[target != -1]
        tar_cifar = target[target != -1]
        sa_cifar = out_aug_s[target != -1]
        ta_cifar = out_aug_t[target != -1]
        
        s_stl = out_s[target == -1]
        t_stl = out_t[target == -1]
        sa_stl = out_aug_s[target == -1]
        ta_stl = out_aug_t[target == -1]
        
        min_size = min(s_stl.size(0), s_cifar.size(0))
        
        s_cifar = s_cifar[:min_size]
        t_cifar = t_cifar[:min_size]
        tar_cifar = tar_cifar[:min_size]
        sa_cifar = sa_cifar[:min_size]
        ta_cifar = ta_cifar[:min_size]
        s_stl = s_stl[:min_size]
        t_stl = t_stl[:min_size]
        sa_stl = sa_stl[:min_size]
        ta_stl = ta_stl[:min_size]

        loss = self.kd_loss(s_cifar, t_cifar, tar_cifar) / 4
        loss += self.kd_loss(sa_cifar, ta_cifar, tar_cifar) / 4
       
        loss += self.uda_loss(s_stl, t_stl) / 4
        loss += self.uda_loss(sa_stl, ta_stl) / 4
        loss.backward()
        self.optimizer.step()
        return out_s, loss


def override_loader(train_loader):
    # This is some hackery to get the original dataset from the train loader
    # and override it
    # We have all the information we need to do this
    batch_size = train_loader.batch_size
    num_workers = train_loader.num_workers

    dataset = train_loader.dataset
    # Assume that the normalization step is the last transform
    normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    #transform = None
    #normalize = transform.transforms[-1]
    #print(type(normalize))
    #if not isinstance(normalize, transforms.Normalize):
    #    normalize = None
    #dataset.transform = None
    # replace the dataset with the custom UDA dataset and refresh the loader
    trainset = UDADataset(dataset=dataset, transform=train_transform,
                          normalize=normalize)
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               pin_memory=True, shuffle=True)
    return train_loader


def run_uda_distillation(s_net, t_net, **params):

    # Grab a new training loader
    params["train_loader"] = override_loader(params["train_loader"])
    # Student training
    s_trainer = UDATrainer(s_net, t_net, config=params)
    best_s_acc = s_trainer.train()

    return best_s_acc
