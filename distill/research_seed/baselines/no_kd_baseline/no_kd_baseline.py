"""
This file defines the core research contribution   
"""
import os
import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser
from research_seed.baselines.no_kd_baseline.model.model_factory import create_cnn_model, is_resnet
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	else:
		return False

class NO_KD_Cifar(pl.LightningModule):

    def __init__(self, hparams):
        super(NO_KD_Cifar, self).__init__()
        # not the best model...
        self.hparams = hparams
        self.model = create_cnn_model(hparams.model, dataset=hparams.dataset, use_cuda=hparams.cuda)
        self.criterion = nn.CrossEntropyLoss()
        self.device = 'cuda:0' if hparams.cuda == True else 'cpu'

        self.train_step = 0
        self.train_num_correct = 0

        self.val_step = 0
        self.val_num_correct = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        pred = y_hat.data.max(1, keepdim=True)[1]

        self.train_step += x.size(0)
        self.train_num_correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        return {
            'loss': loss,
            'log' : {
                'train_loss' : loss.item(), 
                'train_accuracy': float(self.train_num_correct*100/self.train_step)
            } 
        }


    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.forward(x)
        val_loss = self.criterion(y_hat, y)

        pred = y_hat.data.max(1, keepdim=True)[1]

        self.val_step += x.size(0)
        self.val_num_correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        return {
            'val_loss': val_loss
        }

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        log_metrics = {
            'val_avg_loss': avg_loss.item(),
            'val_accuracy': float(self.val_num_correct*100/self.val_step)
        }

        self.scheduler.step(np.around(avg_loss.item(),2))

        # reset logging stuff
        self.train_step = 0
        self.train_num_correct = 0
        self.val_step = 0
        self.val_num_correct = 0

        return {'val_loss': avg_loss, 'log': log_metrics}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)
        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=self.hparams.dataset_dir, train=True,
												 download=True, transform=transform_train)
        #dist_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        return DataLoader(trainset, batch_size=self.hparams.batch_size, num_workers=4)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        valset = torchvision.datasets.CIFAR10(root=self.hparams.dataset_dir, train=False,
												download=True, transform=transform_test)
        #dist_sampler = torch.utils.data.distributed.DistributedSampler(valset)
        return DataLoader(valset, batch_size=self.hparams.batch_size, num_workers=4)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(root=self.hparams.dataset_dir, train=False,
												download=True, transform=transform_test)
        #dist_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        return DataLoader(testset, batch_size=self.hparams.batch_size,num_workers=4)


    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--dataset', default='cifar10', type=str, help='dataset. can be either cifar10 or cifar100')
        parser.add_argument('--batch-size', default=128, type=int, help='batch_size')
        parser.add_argument('--learning-rate', default=0.001, type=float, help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
        parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 1e-4)')
        parser.add_argument('--model', default='resnet110', type=str, help='teacher student name')
        parser.add_argument('--cuda', default=False, type=str2bool, help='whether or not use cuda(train on GPU)')
        parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')

        return parser

