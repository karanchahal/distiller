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
from argparse import ArgumentParser, Action
from research_seed.baselines.model.model_factory import create_cnn_model, is_resnet
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from collections import OrderedDict
import losses
import pairs
from enum import Enum
from embedding import LinearEmbedding
import argparse
from metrics import recall, pdist
from losses import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, AttentionTransfer

class Train_Mode(Enum):
    TEACHER = 1
    STUDENT = 2

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

def addEmbedding(base, hparams):
    embed = LinearEmbedding(base, 
        output_size=hparams.output_size, 
        embedding_size=hparams.embedding_size, 
        normalize=hparams.l2normalize == 'true')

    return embed


def findNumCorrect(embed, labels, K=[1]):
    D = pdist(embed, squared=True)
    knn_inds = D.topk(1 + max(K), dim=1, largest=False, sorted=True)[1][:, 1:]

    """
    Check if, knn_inds contain index of query image.
    """
    assert ((knn_inds == torch.arange(0, len(labels), device=knn_inds.device).unsqueeze(1)).sum().item() == 0)

    selected_labels = labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)
    correct_labels = labels.unsqueeze(1) == selected_labels

    correct_k = (correct_labels[:, :1].sum(dim=1) > 0).float().mean().item()

    return correct_k
    # recall_k = []

    # for k in K:
    #     correct_k = (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
    #     recall_k.append(correct_k)
    # return recall_k

class RKD_Cifar(pl.LightningModule):

    def __init__(self, student_base, teacher_base=None, hparams=None, mode=Train_Mode.TEACHER):
        super(RKD_Cifar, self).__init__()
        # not the best model...
        self.hparams = hparams
        self.mode = mode

        if self.mode == Train_Mode.TEACHER:
            for m in student_base.modules():
                m.requires_grad = False
            self.student = addEmbedding(student_base, hparams)
            self.student.train()
        elif self.mode == Train_Mode.STUDENT:
            self.teacher = teacher_base
            self.student = student_base
            self.student.train()
            self.teacher.eval()


        if self.mode == Train_Mode.TEACHER:
            self.criterionFM = losses.L2Triplet(sampler=self.hparams.sample(), margin=self.hparams.margin)
        elif self.mode == Train_Mode.STUDENT:
            self.dist_criterion = RkdDistance()
            self.angle_criterion = RKdAngle()
            self.dark_criterion = HardDarkRank(alpha=self.hparams.dark_alpha, beta=self.hparams.dark_beta)
            self.triplet_criterion = L2Triplet(sampler=self.hparams.triplet_sample(), margin=self.hparams.triplet_margin)
            

        self.train_step = 0
        self.train_num_correct = 0

        self.val_step = 0
        self.val_num_correct = 0

        self.embeddings_all, self.labels_all = [], []
        self.K = hparams.recall


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
        loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                                F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
                F.cross_entropy(outputs, labels) * (1. - alpha)
        
        return loss

    def forward(self, x, mode):
        if mode == 'student':
            return self.student(x)
        elif mode == 'teacher':
            return self.teacher(x)
        else:
            raise ValueError("mode should be teacher or student")

    def training_step(self, batch, batch_idx):

        x, y = batch
        
        if self.mode == Train_Mode.TEACHER:
            
            embedding = self.student(x)
            loss = self.criterionFM(embedding, y)

            loss_metrics = {
                'train_loss' : loss.item(),
            }

        elif self.mode == Train_Mode.STUDENT:
            
            t_e = self.teacher(x)
            s_e = self.student(x)

            triplet_loss = self.hparams.triplet_ratio * self.triplet_criterion(s_e, y)
            dist_loss = self.hparams.dist_ratio * self.dist_criterion(s_e, t_e)
            angle_loss = self.hparams.angle_ratio * self.angle_criterion(s_e, t_e)
            dark_loss = self.hparams.dark_ratio * self.dark_criterion(s_e, t_e)

            loss = triplet_loss + dist_loss + angle_loss + dark_loss

            acc = findNumCorrect(s_e, y)

            loss_metrics = {
                'train_loss' : loss.item(),
                'triplet_loss': triplet_loss.item(),
                'angle_loss' : angle_loss.item(),
                'dark_loss' : dark_loss.item(),
                'accuracy' : acc,
            }

        return {
            'loss': loss,
            'log' : loss_metrics
        }


    def validation_step(self, batch, batch_idx):
        self.student.eval()
        x, y = batch

        if self.mode == Train_Mode.TEACHER:
            embedding = self.student(x)
            val_loss = self.criterionFM(embedding, y)
            self.embeddings_all.append(embedding.data)
            self.labels_all.append(y.data)

            return {
                'val_loss': val_loss,
            }
        elif self.mode == Train_Mode.STUDENT:
            
            t_e = self.teacher(x)
            s_e = self.student(x)

            triplet_loss = self.hparams.triplet_ratio * self.triplet_criterion(s_e, y)
            dist_loss = self.hparams.dist_ratio * self.dist_criterion(s_e, t_e)
            angle_loss = self.hparams.angle_ratio * self.angle_criterion(s_e, t_e)
            dark_loss = self.hparams.dark_ratio * self.dark_criterion(s_e, t_e)

            loss = triplet_loss + dist_loss + angle_loss + dark_loss

            acc = findNumCorrect(s_e, y)

            return {
                'val_loss' : loss,
                'val_triplet_loss': triplet_loss,
                'val_angle_loss' : angle_loss,
                'val_dark_loss' : dark_loss,
                'accuracy': acc,
            }

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        if self.mode == Train_Mode.TEACHER:
            self.embeddings_all = torch.cat(self.embeddings_all).cpu()
            self.labels_all = torch.cat(self.labels_all).cpu()
            rec = recall(self.embeddings_all, self.labels_all, K=self.K)

            log_metrics = {
                    "recall" : rec[0],
                    "val_loss": avg_loss.item(),
            }
        elif self.mode == Train_Mode.STUDENT:
            avg_triplet_loss = torch.stack([x['val_triplet_loss'] for x in outputs]).mean()
            avg_angle_loss = torch.stack([x['val_angle_loss'] for x in outputs]).mean()
            avg_dark_loss = torch.stack([x['val_dark_loss'] for x in outputs]).mean()
            log_metrics = {
                    "val_triplet_loss" : avg_triplet_loss.item(),
                    "val_angle_loss": avg_angle_loss.item(),
                    "val_dark_loss": avg_dark_loss.item(),
            }
        
        self.embeddings_all, self.labels_all = [], []

        self.train_step = 0
        self.train_num_correct = 0

        self.val_step = 0
        self.val_num_correct = 0

        return { 'val_loss': avg_loss, 'log': log_metrics}
        

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers

        if self.hparams.optim == 'adam':
            optimizer = torch.optim.Adam(self.student.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optim == 'sgd':
            optimizer = torch.optim.SGD(self.student.parameters(), nesterov=True, momentum=self.hparams.momentum, 
            weight_decay=self.hparams.weight_decay, lr=self.hparams.learning_rate)
        else:
            raise ValueError('No such optimizer, please use adam or sgd')
 
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)
        return optimizer
        
    @pl.data_loader
    def train_dataloader(self):

        if self.hparams.dataset == 'cifar10' or self.hparams.dataset == 'cifar100':
            transform_train = transforms.Compose([
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            raise ValueError('Dataset not supported !')

        trainset = torchvision.datasets.CIFAR10(root=self.hparams.dataset_dir, train=True,
												 download=True, transform=transform_train)
        if self.hparams.gpus > 1:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        else:
            dist_sampler = None

        return DataLoader(trainset, batch_size=self.hparams.batch_size, 
                            num_workers=self.hparams.num_workers, sampler=dist_sampler)

    @pl.data_loader
    def val_dataloader(self):
        
        if self.hparams.dataset == 'cifar10' or self.hparams.dataset == 'cifar100':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            raise ValueError('Dataset not supported !')

        valset = torchvision.datasets.CIFAR10(root=self.hparams.dataset_dir, train=False,
												download=True, transform=transform_test)
        if self.hparams.gpus > 1:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(valset)
        else:
            dist_sampler = None

        return DataLoader(valset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, sampler=dist_sampler)

    @pl.data_loader
    def test_dataloader(self):
        
        if self.hparams.dataset == 'cifar10' or self.hparams.dataset == 'cifar100':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            raise ValueError('Dataset not supported !')

        testset = torchvision.datasets.CIFAR10(root=self.hparams.dataset_dir, train=False,
												download=True, transform=transform_test)
        if self.hparams.gpus > 1:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        else:
            dist_sampler = None

        return DataLoader(testset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, sampler=dist_sampler)


    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--dataset', default='cifar10', type=str, help='dataset. can be either cifar10 or cifar100')
        parser.add_argument('--batch-size', default=128, type=int, help='batch_size')
        parser.add_argument('--learning-rate', default=0.001, type=float, help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
        parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 1e-4)')
        parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
        parser.add_argument('--optim', default='adam', type=str, help='Optimizer')
        parser.add_argument('--num-workers', default=4, type=float,  help='Num workers for data loader')
        parser.add_argument('--student-model', default='resnet8', type=str, help='teacher student name')
        parser.add_argument('--teacher-model', default='resnet110', type=str, help='teacher student name')
        parser.add_argument('--path-to-teacher', default='', type=str, help='teacher chkp path')
        parser.add_argument('--temperature', default=10, type=float, help='Temperature for knowledge distillation')
        parser.add_argument('--alpha', default=0.7, type=float, help='Alpha for knowledge distillation')

        parser.add_argument('--triplet_ratio', default=1, type=float)
        parser.add_argument('--dist_ratio', default=1, type=float)
        parser.add_argument('--angle_ratio', default=1, type=float)

        parser.add_argument('--dark_ratio', default=0, type=float)
        parser.add_argument('--dark_alpha', default=2, type=float)
        parser.add_argument('--dark_beta', default=3, type=float)

        parser.add_argument('--at_ratio', default=0, type=float)

        parser.add_argument('--triplet_sample',
                            choices=dict(random=pairs.RandomNegative,
                                        hard=pairs.HardNegative,
                                        all=pairs.AllPairs,
                                        semihard=pairs.SemiHardNegative,
                                        distance=pairs.DistanceWeighted),
                            default=pairs.DistanceWeighted,
                            action=LookupChoices)
        
        parser.add_argument('--sample',
                    choices=dict(random=pairs.RandomNegative,
                                 hard=pairs.HardNegative,
                                 all=pairs.AllPairs,
                                 semihard=pairs.SemiHardNegative,
                                 distance=pairs.DistanceWeighted),
                    default=pairs.AllPairs,
                    action=LookupChoices)

        parser.add_argument('--triplet_margin', type=float, default=0.2)
        parser.add_argument('--l2normalize', choices=['true', 'false'], default='true')
        parser.add_argument('--embedding_size', default=128, type=int)

        parser.add_argument('--teacher_l2normalize', choices=['true', 'false'], default='true')
        parser.add_argument('--teacher_embedding_size', default=128, type=int)

                
        parser.add_argument('--lr', default=1e-5, type=float)
        parser.add_argument('--lr_decay_epochs', type=int,
                            default=[25, 30, 35], nargs='+')
        parser.add_argument('--lr_decay_gamma', default=0.5, type=float)
        parser.add_argument('--data', default='data')
        parser.add_argument('--batch', default=64, type=int)
        parser.add_argument('--iter_per_epoch', default=100, type=int)
        parser.add_argument('--output-size', type=int, default=4096)
        parser.add_argument('--recall', default=[1], type=int, nargs='+')
        parser.add_argument('--save_dir', default=None)
        parser.add_argument('--load', default=None)
        parser.add_argument('--margin', type=float, default=0.2)


        return parser