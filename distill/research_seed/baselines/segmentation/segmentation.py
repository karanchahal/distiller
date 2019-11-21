"""
This file defines the core research contribution   
"""
import os
import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from argparse import ArgumentParser
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from collections import OrderedDict
import torchvision
from coco_utils import get_coco
import transforms as T
import utils


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	else:
		return False

def load_model_chk(model, path):
    chkp = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in chkp['state_dict'].items():
        name = k[6:] # remove `model.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

def get_transform(train):
    base_size = 520
    crop_size = 480

    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transforms = []
    transforms.append(T.RandomResize(min_size, max_size))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomCrop(crop_size))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)

def get_dataset(name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "voc": ('./datasets01/VOC/060817/', torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": ('./datasets01/SBDD/072318/', sbd, 21),
        "coco": ('./datasets01/COCO/022719/', get_coco, 21)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform, download=True)
    return ds, num_classes

class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)
    
    def stats(self):
        acc_global, acc, iu = self.compute()
        global_correct = acc_global.item() * 100
        average_row_correct = ['{:.1f}'.format(i) for i in (acc * 100).tolist()]
        IoU = ['{:.1f}'.format(i) for i in (iu * 100).tolist()]
        mean_iou = iu.mean().item() * 100
        return global_correct, average_row_correct, IoU, mean_iou

    def __str__(self):
        acc_global, acc, iu = self.compute()

        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)

class KD_Segmentation(pl.LightningModule):

    def __init__(self, student, teacher, hparams):
        super(KD_Segmentation, self).__init__()
        # not the best model...
        self.hparams = hparams

        self.student = student
        self.teacher = teacher
        
        self.teacher.eval()
        self.student.train()

        self.confmat = ConfusionMatrix(self.hparams.num_classes)


    def criterion(self, inputs, target):
        losses = {}
        for name, x in inputs.items():
            losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

        if len(losses) == 1:
            return losses['out']

        return losses['out'] + 0.5 * losses['aux']

    def loss_fn_kd(self, outputs, labels, teacher_outputs):
        """
        Credits: https://github.com/peterliht/knowledge-distillation-pytorch/blob/e4c40132fed5a45e39a6ef7a77b15e5d389186f8/model/net.py#L100

        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """
        
        alpha = self.hparams.alpha
        T = self.hparams.temperature
        losses = {}
        for (name, x), (_, t_x) in zip(outputs.items(), teacher_outputs.items()):
            losses[name] = nn.KLDivLoss()(F.log_softmax(x/T, dim=1),
                                F.softmax(t_x/T, dim=1)) * (alpha * T * T) + \
                                nn.functional.cross_entropy(x, labels, ignore_index=255)*(1. - alpha)

        if len(losses) == 1:
            return losses['out']
        
        return losses['out'] + 0.5 * losses['aux']

    def forward(self, x, mode):
        if mode == 'student':
            return self.student(x)
        elif mode == 'teacher':
            return self.teacher(x)
        else:
            raise ValueError("mode should be teacher or student")

    def training_step(self, batch, batch_idx):

        x, y = batch

        y_teacher = self.forward(x, 'teacher')
        y_student = self.forward(x, 'student')
        
        loss = self.loss_fn_kd(y_student, y, y_teacher)

        return {
            'loss': loss,
            'log' : {
                'train_loss' : loss.item()
            } 
        }


    def validation_step(self, batch, batch_idx):
        self.student.eval()
        
        x, y = batch

        y_output = self.forward(x, 'student')

        output = y_output['out']

        self.confmat.update(y.flatten(), output.argmax(1).flatten())
        
        loss = self.criterion(y_output, y)

        return {
            'val_loss': loss,
            'log' : {
                'val_loss' : loss.item()
            } 
        }

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        print(self.confmat)

        global_correct, average_row_correct, IoU, mean_iou = self.confmat.stats()

        self.confmat = ConfusionMatrix(self.hparams.num_classes) # reset

        # back to training
        self.student.train()

        return {
            'val_loss': avg_loss,
            'log': {
                "val_avg_loss": avg_loss.item(),
                "iou": mean_iou,
                "global_correct": global_correct,
            }
        }

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers

        params_to_optimize = [
            {"params": [p for p in self.student.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in self.student.classifier.parameters() if p.requires_grad]},
        ]

        if self.hparams.aux_loss:
            params = [p for p in self.student.aux_classifier.parameters() if p.requires_grad]
            params_to_optimize.append({"params": params, "lr": self.hparams.learning_rate * 10})

        self.optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=self.hparams.learning_rate, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: (1 - x / (183 * self.hparams.epochs)) ** 0.9)

        return [self.optimizer], [self.scheduler]

    @pl.data_loader
    def train_dataloader(self):

        if self.hparams.dataset == 'voc':
            transform_test = get_transform(train=True)
        else:
            raise ValueError('Dataset not supported !')

        dataset_train, _ = get_dataset(self.hparams.dataset, "train", transform_test)

        if self.hparams.gpus > 1:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        else:
            dist_sampler = None

        self.data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=self.hparams.batch_size,
            sampler=dist_sampler, num_workers=self.hparams.num_workers,
            collate_fn=utils.collate_fn, drop_last=True)
        print("Length: ", len(self.data_loader_train))

        return self.data_loader_train

    @pl.data_loader
    def val_dataloader(self):

        if self.hparams.dataset == 'voc':
            transform_test = get_transform(train=False)
        else:
            raise ValueError('Dataset not supported !')

        dataset_test, _ = get_dataset(self.hparams.dataset, "val", transform_test)

        if self.hparams.gpus > 1:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        else:
            dist_sampler = None

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            sampler=dist_sampler, num_workers=self.hparams.num_workers,
            collate_fn=utils.collate_fn)

        return data_loader_test

    @pl.data_loader
    def test_dataloader(self):
        
        if self.hparams.dataset == 'voc':
            transform_test = get_transform(train=False)
        else:
            raise ValueError('Dataset not supported !')

        dataset_test, _ = get_dataset(self.hparams.dataset, "val", transform_test)

        if self.hparams.gpus > 1:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        else:
            dist_sampler = None

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            sampler=dist_sampler, num_workers=self.hparams.num_workers,
            collate_fn=utils.collate_fn)

        return data_loader_test


    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--dataset', default='voc', type=str, help='dataset. can be either cifar10 or cifar100')
        parser.add_argument('--batch-size', default=8, type=int, help='batch_size')
        parser.add_argument('--learning-rate', default=0.01, type=float, help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
        parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 1e-4)')
        parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
        parser.add_argument('--optim', default='adam', type=str, help='Optimizer')
        parser.add_argument('--num-workers', default=8, type=int,  help='Num workers for data loader')
        parser.add_argument('--num-classes', default=21, type=int,  help='Num workers for data loader')
        parser.add_argument('--student-model', default='fcn_resnet50', type=str, help='student name')
        parser.add_argument('--teacher-model', default='fcn_resnet101', type=str, help='teacher name')
        parser.add_argument('--path-to-teacher', default='', type=str, required=True, help='teacher chkp path')
        parser.add_argument('--temperature', default=5, type=float, help='Temperature for knowledge distillation')
        parser.add_argument('--alpha', default=0.7, type=float, help='Alpha for knowledge distillation')
        parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            help="Use pre-trained models from the modelzoo",
            action="store_true",
        )
        return parser

