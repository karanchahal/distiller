import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import namedtuple


def load_checkpoint(model, checkpoint_path):
    """
    Loads weights from checkpoint
    :param model: a pytorch nn student
    :param str checkpoint_path: address/path of a file
    :return: pytorch nn student with weights loaded from checkpoint
    """
    model_ckp = torch.load(checkpoint_path)
    model.load_state_dict(model_ckp["model_state_dict"])
    return model


class ClassDict:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class LightningTrainer(pl.LightningModule):

    def __init__(self, net, params):
        self.hparams = ClassDict(**params)
        super(LightningTrainer, self).__init__()

        self.net = net
        self.device = self.hparams.device
        self.name = self.hparams.name

        optim_cls, optim_args = self.hparams.optim
        sched_cls, sched_args = self.hparams.sched
        self.optimizer = optim_cls(net.parameters(), **optim_args)
        self.scheduler = sched_cls(self.optimizer, **sched_args)
        self.loss_fun = nn.CrossEntropyLoss()
        self.train_loader = self.hparams.train_loader
        self.test_loader = self.hparams.test_loader
        # tqdm bar
        self.t_bar = None

        self.train_step = 0
        self.train_num_correct = 0
        self.val_step = 0
        self.val_num_correct = 0

    def forward(self, x):
        return self.net(x)

    def calculate_loss(self, data, target):
        raise NotImplementedError("calculate_loss not implemented!")

    def training_step(self, batch, batch_idx):

        x, y = batch
        loss, y_pred = self.calculate_loss(x, y)
        pred = y_pred.data.max(1, keepdim=True)[1]
        self.train_step += x.size(0)
        self.train_num_correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        return {
            'loss': loss,
            'log': {
                'train_loss': loss.item(),
                'train_accuracy': float(self.train_num_correct * 100 / self.train_step),
            }
        }

    def validation_step(self, batch, batch_idx):
        self.net.eval()
        x, y = batch

        y_hat = self.forward(x)
        val_loss = self.loss_fun(y_hat, y)

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
            'val_accuracy': float(self.val_num_correct * 100 / self.val_step)
        }

        if self.scheduler:
            self.scheduler.step()

        # reset logging stuff
        self.train_step = 0
        self.train_num_correct = 0
        self.val_step = 0
        self.val_num_correct = 0

        # back to training
        self.net.train()

        return {'val_loss': avg_loss, 'log': log_metrics}

    def configure_optimizers(self):
        return self. optimizer

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.test_loader

    @pl.data_loader
    def test_dataloader(self):
        return self.test_loader


class BaseTrainer(LightningTrainer):

    def calculate_loss(self, data, target):
        # Standard Learning Loss ( Classification Loss)
        output = self.net(data)
        loss = self.loss_fun(output, target)
        return loss, output


class KDTrainer(LightningTrainer):
    def __init__(self, s_net, t_net, train_config):
        super(KDTrainer, self).__init__(s_net, train_config)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net
        # set the teacher net into evaluation mode
        self.t_net.eval()
        self.t_net.train(mode=False)

    def calculate_loss(self, data, target):
        lambda_ = self.hparams.lambda_student
        T = self.hparams.T_student
        output = self.s_net(data)
        # Standard Learning Loss ( Classification Loss)
        loss = self.loss_fun(output, target)

        # Knowledge Distillation Loss
        teacher_outputs = self.t_net(data)
        student_max = F.log_softmax(output / T, dim=1)
        teacher_max = F.softmax(teacher_outputs / T, dim=1)
        loss_KD = nn.KLDivLoss()(student_max, teacher_max)
        loss = (1 - lambda_) * loss + lambda_ * T * T * loss_KD
        return loss, output
