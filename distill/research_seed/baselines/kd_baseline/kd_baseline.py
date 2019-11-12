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
from research_seed.baselines.model.model_factory import create_cnn_model, is_resnet
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	else:
		return False

def load_model_chk(model, path):
    chkp = torch.load(path)
    model.load_state_dict(chkp['state_dict'])
    return model
    
class KD_Cifar(pl.LightningModule):

    def __init__(self, hparams):
        super(KD_Cifar, self).__init__()
        # not the best model...
        self.hparams = hparams

        self.student = create_cnn_model(hparams.student_model, dataset=hparams.dataset, use_cuda=hparams.cuda)
        self.teacher = create_cnn_model(hparams.teacher_model, dataset=hparams.dataset, use_cuda=hparams.cuda)
        
        # self.student = load_student_chk(student, hparams.path_to_student)
        self.teacher = load_model_chk(self.teacher, hparams.path_to_teacher)

        self.teacher.eval()
        self.student.train()

        self.criterion = nn.CrossEntropyLoss()

        self.device = 'cuda:0' if hparams.cuda == True else 'cpu'

        self.train_step = 0
        self.train_num_correct = 0

        self.val_step = 0
        self.val_num_correct = 0


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
        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                                F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
                F.cross_entropy(outputs, labels) * (1. - alpha)

        return KD_loss

    def forward(self, x, mode):
        if mode == 'student':
            return self.student(x)
        elif mode == 'teacher':
            return self.teacher(x)
        else:
            raise ValueError("mode should be teacher or student")

    def training_step(self, batch, batch_idx):

        x, y = batch

        x = x.to(self.device)
        y = y.to(self.device)

        y_teacher = self.forward(x, 'teacher')
        y_student = self.forward(x, 'student')
        
        loss = self.loss_fn_kd(y_student, y, y_teacher)

        pred = y_student.data.max(1, keepdim=True)[1]

        self.train_step += x.size(0)
        self.train_num_correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        return {
            'loss': loss,
            'log' : {
                'train_loss' : loss.item(), 
                'train_accuracy': float(self.train_num_correct*100/self.train_step)
            } 
        }

    def training_end(self, outputs):
        self.student.eval()
        return outputs

    def validation_step(self, batch, batch_idx):

        x, y = batch

        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.forward(x, 'student')
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

        # back to training
        self.student.train()

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
        parser.add_argument('--student-model', default='resnet110', type=str, help='teacher student name')
        parser.add_argument('--teacher-model', default='resnet110', type=str, help='teacher student name')
        parser.add_argument('--path-to-teacher', default='', type=str, help='teacher chkp path')
        parser.add_argument('--cuda', default=False, type=str2bool, help='whether or not use cuda(train on GPU)')
        parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
        parser.add_argument('--temperature', default=20, type=float, help='Temperature for knowledge distillation')
        parser.add_argument('--alpha', default=0.9, type=float, help='Alpha for knowledge distillation')

        return parser

