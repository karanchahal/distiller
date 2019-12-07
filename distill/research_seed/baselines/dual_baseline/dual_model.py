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

class Dual_Cifar(pl.LightningModule):

    def __init__(self, student, teacher, hparams):
        super(Dual_Cifar, self).__init__()
        # not the best model...
        self.hparams = hparams
        self.student = student
        self.teacher = teacher

        self.criterion = nn.CrossEntropyLoss()

        self.init_acc_step_stats()
        
        # It will be the teacher's turn starting out
        self.student_turn = False
        self.teacher_turn = True

        self.teacher_epochs = 0
        self.student_epochs = 0

        self.prev_val_teacher_acc = 0
        self.current_val_teacher_acc = 0

        self.prev_val_student_acc = 0
        self.current_val_student_acc = 0

    def init_acc_step_stats(self):
        # Teacher stats
        self.train_teacher_step = 0
        self.train_teacher_num_correct = 0

        self.val_step = 0
        self.val_num_correct = 0

        # Student stats
        self.train_student_step = 0
        self.train_student_num_correct = 0


    def forward(self, x, mode):
        if mode == 'teacher':
            return self.teacher(x)
        elif mode == 'student':
            return self.student(x)

    def cross_entropy(self, out, target):
        return self.criterion(out, target)

    def kd_loss(self, outputs, labels, teacher_outputs):
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
    
    def teachers_turn(self):
        self.teacher_turn = True
        self.student_turn = False
        self.student.train()
        self.teacher.train()
        self.teacher_epochs += 1
        self.student_epochs = 0
    
    def students_turn(self):
        self.student.train()
        self.teacher.eval()
        self.teacher_turn = False
        self.student_turn = True
        self.teacher.eval()
        self.student_epochs += 1
        self.teacher_epochs = 0

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        # override default optimizer setup
        if self.teacher_turn == True:
            self.optimizer2.step()
            self.optimizer2.zero_grad()
        else:
            self.optimizer1.step()
            self.optimizer1.zero_grad() 

    def setup_mode(self):
        if self.teacher_turn == True:

            if self.teacher_epochs >  5:
                # if teacher has been training for more than 5 epochs without gaining 2 percent, give to student training
                # forces more training time to student
                self.prev_val_teacher_acc = self.current_val_teacher_acc
                self.students_turn()
                return

            if self.current_val_teacher_acc > self.prev_val_teacher_acc + 2:
                self.prev_val_teacher_acc = self.current_val_teacher_acc
                self.students_turn()
                return
            else:
                self.teachers_turn()
                return
            return
        elif self.student_turn == True:
            if self.current_val_student_acc >= self.current_val_teacher_acc:
                self.prev_val_student_acc = self.current_val_student_acc
                self.teachers_turn()
                return
            else:
                if self.current_val_student_acc + 10 < self.prev_val_student_acc:
                    # student is degrading badly like dipping more than 10 percent, maybe 
                    # - train the teacher
                    # - weight cross entropy loss more ?
                    self.prev_val_student_acc = self.current_val_student_acc
                    self.teachers_turn()
                    return

                self.students_turn()
                return
        else:
            raise ValueError("Both student and teacher can't be false to train")


    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if self.teacher_turn == True:
            y_hat = self.forward(x, 'teacher')
            loss = self.cross_entropy(y_hat, y)
            pred = y_hat.data.max(1, keepdim=True)[1]
            self.train_teacher_step += x.size(0)
            self.train_teacher_num_correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            self.current_train_teacher_acc = float(self.train_teacher_num_correct*100/self.train_teacher_step)
            loss_metrics = {
                'teacher_train_loss': loss.item(),
                'teacher_train_accuracy': self.current_train_teacher_acc
            }
        elif self.student_turn == True:
            y_hat = self.forward(x, 'student')
            y_t = self.forward(x, 'teacher')
            loss = self.kd_loss(y_hat, y, y_t)
            pred = y_hat.data.max(1, keepdim=True)[1]
            self.train_student_step += x.size(0)
            self.train_student_num_correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            self.current_train_student_acc = float(self.train_student_num_correct*100/self.train_student_step)
            loss_metrics = {
                'student_train_loss': loss.item(),
                'student_train_accuracy': self.current_train_student_acc
            }

        return {
            'loss': loss,
            'log' : loss_metrics,
        }


    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.teacher_turn == True:
            self.teacher.eval()
            y_hat = self.forward(x,'teacher')
            val_loss = self.cross_entropy(y_hat, y)

            pred = y_hat.data.max(1, keepdim=True)[1]

            self.val_step += x.size(0)
            self.val_num_correct += pred.eq(y.data.view_as(pred)).cpu().sum()

            return {
                'val_loss': val_loss
            }
        elif self.student_turn == True:
            self.student.eval()
            y_hat = self.forward(x,'student')
            val_loss = self.cross_entropy(y_hat, y)

            pred = y_hat.data.max(1, keepdim=True)[1]

            self.val_step += x.size(0)
            self.val_num_correct += pred.eq(y.data.view_as(pred)).cpu().sum()

            return {
                'val_loss': val_loss
            }

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_accuracy = float(self.val_num_correct*100/self.val_step)

        if self.teacher_turn == True:
            self.current_val_teacher_acc = val_accuracy
            self.scheduler2.step(np.around(avg_loss.item(),2))
            self.teacher_epochs += 1
            log_metrics = {
                'val_avg_teacher_loss': avg_loss.item(),
                'val_teacher_accuracy': self.current_val_teacher_acc,
                'val_student_accuracy': self.current_val_student_acc,
                'student_turn' : 0,
                'teacher_turn' : 1,
            } 
        elif self.student_turn == True:
            self.current_val_student_acc = val_accuracy
            self.scheduler1.step(np.around(avg_loss.item(),2))
            self.student_epochs += 1
            log_metrics = {
                'val_avg_student_loss': avg_loss.item(),
                'val_student_accuracy': self.current_val_student_acc,
                'val_teacher_accuracy': self.current_val_teacher_acc,
                'student_turn' : 1,
                'teacher_turn' : 0,

            }
        
        # set mode of training; teacher or student
        self.setup_mode()

        # reset logging stuff
        self.init_acc_step_stats()

        return {'val_loss': avg_loss, 'log': log_metrics}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        if self.hparams.optim == 'adam':
            self.optimizer1 = torch.optim.Adam(self.student.parameters(), lr=self.hparams.learning_rate)
            self.optimizer2 = torch.optim.Adam(self.teacher.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optim == 'sgd':
            self.optimizer1 = torch.optim.SGD(self.student.parameters(), nesterov=True, momentum=self.hparams.momentum, 
            weight_decay=self.hparams.weight_decay, lr=self.hparams.learning_rate)
            self.optimizer2 = torch.optim.SGD(self.teacher.parameters(), nesterov=True, momentum=self.hparams.momentum, 
            weight_decay=self.hparams.weight_decay, lr=self.hparams.learning_rate)
        else:
            raise ValueError('No such optimizer, please use adam or sgd')

        self.scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer1, 'min',patience=5,factor=0.5,verbose=True)
        self.scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer2, 'min',patience=5,factor=0.5,verbose=True)
        return self.optimizer1

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        transform_train = transforms.Compose([
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=self.hparams.dataset_dir, train=True,
												 download=True, transform=transform_train)
        if self.hparams.gpus > 1:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        else:
            dist_sampler = None

        return DataLoader(trainset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, sampler=dist_sampler)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        valset = torchvision.datasets.CIFAR10(root=self.hparams.dataset_dir, train=False,
												download=True, transform=transform_test)
        if self.hparams.gpus > 1:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(valset)
        else:
            dist_sampler = None
        return DataLoader(valset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, sampler=dist_sampler)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(root=self.hparams.dataset_dir, train=False,
												download=True, transform=transform_test)
        
        if self.hparams.gpus > 1:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        else:
            dist_sampler = None
            
        return DataLoader(testset, batch_size=self.hparams.batch_size,num_workers=self.hparams.num_workers, sampler=dist_sampler)


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
        parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
        parser.add_argument('--optim', default='adam', type=str, help='Optimizer')
        parser.add_argument('--num-workers', default=4, type=float,  help='Num workers for data loader')
        parser.add_argument('--student-model', default='resnet8', type=str, help='teacher student name')
        parser.add_argument('--teacher-model', default='resnet110', type=str, help='teacher student name')
        parser.add_argument('--path-to-teacher', default='', type=str, help='teacher chkp path')
        parser.add_argument('--path-to-student', default='', type=str, help='student chkp path')
        parser.add_argument('--temperature', default=5, type=float, help='Temperature for knowledge distillation')
        parser.add_argument('--alpha', default=0.7, type=float, help='Alpha for knowledge distillation')
        return parser

