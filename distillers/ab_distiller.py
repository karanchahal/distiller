# Original Repo:
# https://github.com/bhheo/AB_distillation
# @inproceedings{heo2019knowledge,
#  title={Knowledge transfer via distillation of activation boundaries
#  formed by hidden neurons},
#  author={Heo, Byeongho and Lee, Minsik and Yun, Sangdoo and Choi, Jin Young},
#  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
#  volume={33},
#  pages={3779--3787},
#  year={2019}
# }


import copy
import math
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch

from trainer import BaseTrainer, KDTrainer


def alt_L2(source, target, margin):
    # Proposed alternative loss function
    loss = ((source + margin)**2 * ((source > -margin) & (target <= 0)).float() +
            (source - margin)**2 * ((source <= margin) & (target > 0)).float())
    return torch.abs(loss).sum()


DISTILL_EPOCHS = 10


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


class DistillTrainer(BaseTrainer):
    def __init__(self, s_net, d_net, config):
        super(DistillTrainer, self).__init__(s_net, config)
        # the student net is the base net
        self.s_net = self.net
        self.d_net = d_net
        d_net.train()
        d_net.s_net.train()
        d_net.t_net.train()
        self.optimizer = optim.SGD([{'params': s_net.parameters()},
                                    {'params': d_net.Connectors.parameters()}],
                                   lr=0.1, nesterov=True, momentum=0.9,
                                   weight_decay=5e-4)

    def calculate_loss(self, data, target):

        batch_size = data.shape[0]
        self.d_net(data)

        # Alternative loss
        margin = 1.0
        loss_alter = alt_L2(self.d_net.Connect3(
            self.d_net.res3), self.d_net.res3_t.detach(), margin) / batch_size
        loss_alter += alt_L2(self.d_net.Connect2(self.d_net.res2),
                             self.d_net.res2_t.detach(), margin) / batch_size / 2
        loss_alter += alt_L2(self.d_net.Connect1(self.d_net.res1),
                             self.d_net.res1_t.detach(), margin) / batch_size / 4

        loss = loss_alter / 1000 * 3

        loss.backward()
        self.optimizer.step()
        return loss


def run_ab_distillation(s_net, t_net, **params):
    d_net = Active_Soft_WRN_norelu(t_net, s_net).to(params["device"])

    # Distillation (Initialization)
    print("---------- Initialize AB Student Distillation-------")
    d_config = copy.deepcopy(params)
    d_config["test_name"] = "ab_distillation"
    d_config["epochs"] = DISTILL_EPOCHS
    d_trainer = DistillTrainer(s_net, d_net=d_net, config=d_config)
    d_trainer.train()
    s_net = d_trainer.s_net

    # Student training
    print("---------- Training AB Student -------")
    sname = params["s_name"]
    s_config = copy.deepcopy(params)
    s_config["name"] = sname
    s_trainer = KDTrainer(s_net, t_net=t_net, config=s_config)
    best_s_acc = s_trainer.train()
    return best_s_acc
