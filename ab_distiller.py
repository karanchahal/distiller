'''Train CIFAR10 with PyTorch.'''

import time
import copy
import math
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

from trainer import BaseTrainer


def criterion_alternative_L2(source, target, margin):
    # Proposed alternative loss function
    loss = ((source + margin)**2 * ((source > -margin) & (target <= 0)).float() +
            (source - margin)**2 * ((source <= margin) & (target > 0)).float())
    return torch.abs(loss).sum()


DISTILL_EPOCHS = 3


class Active_Soft_WRN_norelu(nn.Module):
    def __init__(self, t_net, s_net):
        super(Active_Soft_WRN_norelu, self).__init__()

        # Connection layers
        if t_net.nChannels == s_net.nChannels:
            C1 = []
            C2 = []
            C3 = []
        else:
            C1 = [nn.Conv2d(int(s_net.nChannels / 4), int(t_net.nChannels / 4), kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(int(t_net.nChannels / 4))]
            C2 = [nn.Conv2d(int(s_net.nChannels / 2), int(t_net.nChannels / 2), kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(int(t_net.nChannels / 2))]
            C3 = [nn.Conv2d(s_net.nChannels, t_net.nChannels, kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(t_net.nChannels)]

        # Weight initialize
        for m in C1 + C2 + C3:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.Connect1 = nn.Sequential(*C1)
        self.Connect2 = nn.Sequential(*C2)
        self.Connect3 = nn.Sequential(*C3)
        self.Connectors = nn.ModuleList(
            [self.Connect1, self.Connect2, self.Connect3])

        self.t_net = t_net
        self.s_net = s_net

    def forward(self, x):
        # For teacher network
        self.res0_t = self.t_net.conv1(x)

        self.res1_t = self.t_net.block1(self.res0_t)
        self.res2_t = self.t_net.block2(self.res1_t)
        self.res3_t = self.t_net.bn1(self.t_net.block3(self.res2_t))

        out = self.t_net.relu(self.res3_t)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.t_net.nChannels)
        self.out_t = self.t_net.fc(out)

        # For student network
        self.res0 = self.s_net.conv1(x)

        self.res1 = self.s_net.block1(self.res0)
        self.res2 = self.s_net.block2(self.res1)
        self.res3 = self.s_net.block3(self.res2)

        out = self.s_net.relu(self.s_net.bn1(self.res3))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.s_net.nChannels)
        self.out_s = self.s_net.fc(out)

        # Features before ReLU
        self.res0_t = self.t_net.block1.layer[0].bn1(self.res0_t)
        self.res1_t = self.t_net.block2.layer[0].bn1(self.res1_t)
        self.res2_t = self.t_net.block3.layer[0].bn1(self.res2_t)

        self.res0 = self.s_net.block1.layer[0].bn1(self.res0)
        self.res1 = self.s_net.block2.layer[0].bn1(self.res1)
        self.res2 = self.s_net.block3.layer[0].bn1(self.res2)
        self.res3 = self.s_net.bn1(self.res3)

        return self.out_s


def init_distillation(s_net, ta_net, epoch, train_loader):
    epoch_start_time = time.time()
    print('\nDistillation epoch: %d' % epoch)
    ta_net.train()
    ta_net.s_net.train()
    ta_net.t_net.train()
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    optimizer = optim.SGD([{'params': s_net.parameters()},
                           {'params': ta_net.Connectors.parameters()}], lr=0.1, nesterov=True, momentum=0.9, weight_decay=5e-4)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        batch_size = inputs.shape[0]
        ta_net(inputs)

        # Activation transfer loss
        loss_AT1 = ((ta_net.Connect1(ta_net.res1) > 0) ^ (
            ta_net.res1_t.detach() > 0)).sum().float() / ta_net.res1_t.nelement()
        loss_AT2 = ((ta_net.Connect2(ta_net.res2) > 0) ^ (
            ta_net.res2_t.detach() > 0)).sum().float() / ta_net.res2_t.nelement()
        loss_AT3 = ((ta_net.Connect3(ta_net.res3) > 0) ^ (
            ta_net.res3_t.detach() > 0)).sum().float() / ta_net.res3_t.nelement()

        # Alternative loss
        margin = 1.0
        loss_alter = criterion_alternative_L2(ta_net.Connect3(
            ta_net.res3), ta_net.res3_t.detach(), margin) / batch_size
        loss_alter += criterion_alternative_L2(ta_net.Connect2(
            ta_net.res2), ta_net.res2_t.detach(), margin) / batch_size / 2
        loss_alter += criterion_alternative_L2(ta_net.Connect1(
            ta_net.res1), ta_net.res1_t.detach(), margin) / batch_size / 4

        loss = loss_alter / 1000 * 3

        loss.backward()
        optimizer.step()

        train_loss1 += loss_AT1.item()
        train_loss2 += loss_AT2.item()
        train_loss3 += loss_AT3.item()

        b_idx = batch_idx

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('layer1_activation similarity %.1f%%' %
          (100 * (1 - train_loss1 / (b_idx + 1))))
    print('layer2_activation similarity %.1f%%' %
          (100 * (1 - train_loss2 / (b_idx + 1))))
    print('layer3_activation similarity %.1f%%' %
          (100 * (1 - train_loss3 / (b_idx + 1))))


class ABTrainer(BaseTrainer):
    def __init__(self, s_net, t_net, train_config):
        super(ABTrainer, self).__init__(s_net, train_config)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net
        # set the teacher net into evaluation mode
        self.t_net.eval()
        self.t_net.train(mode=False)

        self.optimizer = optim.SGD(s_net.parameters(),
                                   nesterov=True,
                                   lr=train_config["learning_rate"],
                                   momentum=train_config["momentum"],
                                   weight_decay=train_config["weight_decay"])
        self.loss_fun = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

    def calculate_loss(self, data, target):
        batch_size = data.shape[0]

        self.optimizer.zero_grad()
        inputs, target = Variable(data), Variable(target)
        out_t = self.t_net(inputs)
        out_s = self.s_net(inputs)

        loss = - (F.softmax(out_t / self.config["T_student"], 1).detach() * (F.log_softmax(
            out_s / self.config["T_student"], 1) - F.log_softmax(out_t / self.config["T_student"], 1).detach())).sum() / batch_size

        loss.backward()
        self.optimizer.step()
        return loss


def run_ab_distillation(s_net, t_net, **params):
    d_net = Active_Soft_WRN_norelu(t_net, s_net).to(params["device"])

    # Distillation (Initialization)
    for epoch in range(1, int(DISTILL_EPOCHS) + 1):
        init_distillation(s_net, d_net, epoch, params["train_loader"])
    # Student training
    print("---------- Training AB Student -------")
    student_name = params["s_name"]
    s_train_config = copy.deepcopy(params)
    s_train_config["name"] = student_name
    s_trainer = ABTrainer(s_net, t_net=t_net, train_config=s_train_config)
    best_s_acc = s_trainer.train()
    return best_s_acc
