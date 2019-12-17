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
import torch.nn.functional as F
import torch.nn as nn
import torch
import util

from trainer import BaseTrainer, KDTrainer


def alt_L2(source, target, margin):
    # Proposed alternative loss function
    loss = ((source + margin)**2 * ((source > -margin) & (target <= 0)).float() +
            (source - margin)**2 * ((source <= margin) & (target > 0)).float())
    return torch.abs(loss).sum()


DISTILL_EPOCHS = 10
# SUPPORTED = ["WRN10_4", "WRN16_1", "WRN16_2", "WRN16_4",
#              "WRN16_8", "WRN28_2", "WRN22_4", "WRN22_8",
#              "WRN28_1", "WRN10_1", "WRN40_1", "WRN40_4",
#              ]

SUPPORTED = ["resnet8", "resnet14", "resnet20", "resnet26",
             "resnet32", "resnet44", "resnet56", "resnet10",
             "resnet18", "resnet34", "resnet50", "resnet101",
             "resnet152", ]


def get_feat_layers(net):
    layers = []
    for layer in list(net.children()):
        if not isinstance(layer, nn.Linear):
            layers.append(layer)
    return layers


class AB_distill_Resnet(nn.Module):

    def __init__(self, t_net, s_net):
        super(AB_distill_Resnet, self).__init__()

        # another hack to support dataparallel models...
        if isinstance(t_net, nn.DataParallel):
            t_net = t_net.module
        if isinstance(s_net, nn.DataParallel):
            s_net = s_net.module

        self.expansion = 2

        self.n_channels_s = s_net.get_channel_num()
        self.n_channels_t = t_net.get_channel_num()
        bns_s = []
        bns_t = []
        for idx, channel in enumerate(self.n_channels_s):
            bns_s.append(nn.BatchNorm2d(channel))
        for idx, channel in enumerate(self.n_channels_t):
            bns_t.append(nn.BatchNorm2d(channel))
        self.bns_s = nn.ModuleList(bns_s)
        self.bns_t = nn.ModuleList(bns_t)
        fc_channel_s = self.n_channels_s[-1]
        fc_channel_t = self.n_channels_t[-1]
        # connection layers
        if fc_channel_t == fc_channel_s:
            C1 = []
            C2 = []
            C3 = []
        else:
            C1 = [nn.Conv2d(int(fc_channel_s / 4), int(fc_channel_t / 4), kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(int(fc_channel_t / 4))]
            C2 = [nn.Conv2d(int(fc_channel_s / 2), int(fc_channel_t / 2), kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(int(fc_channel_t / 2))]
            C3 = [nn.Conv2d(fc_channel_s, fc_channel_t, kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(fc_channel_t)]

        for m in C1 + C2 + C3:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        connect1 = nn.Sequential(*C1)
        connect2 = nn.Sequential(*C2)
        connect3 = nn.Sequential(*C3)
        self.connectors = nn.ModuleList([connect1, connect2, connect3])

        self.t_net = t_net
        self.s_net = s_net

        self.stage1 = True
        self.criterion_CE = nn.CrossEntropyLoss(size_average=False)

    def forward(self, x):

        # Teacher network
        self.res0_t = F.relu(self.t_net.bn1(self.t_net.conv1(x)))

        self.res1_t = self.t_net.layer1(self.res0_t)
        self.res2_t = self.t_net.layer2(self.res1_t)
        self.res3_t = self.t_net.layer3(self.res2_t)

        out = F.relu(self.res3_t)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        self.out_t = self.t_net.linear(out)

        # Student network
        self.res0_s = F.relu(self.s_net.bn1(self.s_net.conv1(x)))
        self.res1_s = self.s_net.layer1(self.res0_s)
        self.res2_s = self.s_net.layer2(self.res1_s)
        self.res3_s = self.s_net.layer3(self.res2_s)
        # self.res1_s = self.s_net.model[3][:-1](self.s_net.model[0:3](x))
        # self.res2_s = self.s_net.model[5][:-
        #                                   1](self.s_net.model[4:5](F.relu(self.res1_s)))
        # self.res3_s = self.s_net.model[11][:-
        #                                    1](self.s_net.model[6:11](F.relu(self.res2_s)))

        out = F.relu(self.res3_s)
        out = F.avg_pool2d(out, 4)
        out = out.view(-1, self.n_channels_s[-1])
        self.out_s = self.s_net.linear(out)

        # Return all losses
        return self.out_s


class Active_Soft_WRN_norelu(nn.Module):
    def __init__(self, t_net, s_net):

        super(Active_Soft_WRN_norelu, self).__init__()

        # another hack to support dataparallel models...
        if isinstance(t_net, nn.DataParallel):
            t_net = t_net.module
        if isinstance(s_net, nn.DataParallel):
            s_net = s_net.module
        self.n_channels_s = s_net.get_channel_num()
        self.n_channels_t = t_net.get_channel_num()
        fc_channel_s = self.n_channels_s[-1]
        fc_channel_t = self.n_channels_t[-1]
        bns_s = []
        bns_t = []
        for idx, channel in enumerate(self.n_channels_s):
            bns_s.append(nn.BatchNorm2d(channel))
        for idx, channel in enumerate(self.n_channels_t):
            bns_t.append(nn.BatchNorm2d(channel))
        self.bns_s = nn.ModuleList(bns_s)
        self.bns_t = nn.ModuleList(bns_t)
        # connection layers
        if fc_channel_t == fc_channel_s:
            C1 = []
            C2 = []
            C3 = []
        else:
            C1 = [nn.Conv2d(int(fc_channel_s / 4), int(fc_channel_t / 4), kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(int(fc_channel_t / 4))]
            C2 = [nn.Conv2d(int(fc_channel_s / 2), int(fc_channel_t / 2), kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(int(fc_channel_t / 2))]
            C3 = [nn.Conv2d(fc_channel_s, fc_channel_t, kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(fc_channel_t)]

        # Weight initialize
        for m in C1 + C2 + C3:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.connect1 = nn.Sequential(*C1)
        self.connect2 = nn.Sequential(*C2)
        self.connect3 = nn.Sequential(*C3)

        self.t_net = t_net
        self.s_net = s_net
        self.res0_t = None
        self.res1_t = None
        self.res2_t = None

        self.res0 = None
        self.res1 = None
        self.res2 = None
        self.res3 = None

    def forward(self, x):
        # For teacher network
        res0_t = self.t_net.conv1(x)

        res1_t = self.t_net.layer1(res0_t)
        res2_t = self.t_net.layer2(res1_t)
        res3_t = self.t_net.layer3(res2_t)
        res3_t = self.bns_t[3](res3_t)

        out = self.t_net.relu(res3_t)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.n_channels_t[-1])

        # For student network
        res0 = self.s_net.conv1(x)

        res1 = self.s_net.layer1(res0)
        res2 = self.s_net.layer2(res1)
        res3 = self.s_net.layer3(res2)
        res3 = self.bns_s[3](res3)

        out = self.s_net.relu(res3)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.n_channels_s[-1])
        out_s = self.s_net.linear(out)

        # Features before ReLU
        self.res0_t = self.t_net.layer1.layer[0].bn1(res0_t)
        self.res1_t = self.t_net.layer2.layer[0].bn1(res1_t)
        self.res2_t = self.t_net.layer3.layer[0].bn1(res2_t)

        self.res0 = self.s_net.layer1.layer[0].bn1(res0)
        self.res1 = self.s_net.layer2.layer[0].bn1(res1)
        self.res2 = self.s_net.layer3.layer[0].bn1(res2)
        self.res3 = self.bns_s[3](self.res3)

        return out_s


class DistillTrainer(BaseTrainer):
    def __init__(self, s_net, d_net, config):
        super(DistillTrainer, self).__init__(s_net, config)
        # the student net is the base net
        self.s_net = self.net
        self.d_net = d_net
        optim_params = [{"params": self.s_net.parameters()},
                        {"params": self.d_net.connectors.parameters()}]

        # Retrieve preconfigured optimizers and schedulers for all runs
        self.optimizer = self.optim_cls(optim_params, **self.optim_args)
        self.scheduler = self.sched_cls(self.optimizer, **self.sched_args)

    def calculate_loss(self, data, target):

        batch_size = data.shape[0]
        output = self.d_net(data)

        # Alternative loss
        margin = 1.0
        loss_alter = alt_L2(self.d_net.connectors[2](
            self.d_net.res3_s), self.d_net.res3_t.detach(), margin) / batch_size
        loss_alter += alt_L2(self.d_net.connectors[1](self.d_net.res2_s),
                             self.d_net.res2_t.detach(), margin) / batch_size / 2
        loss_alter += alt_L2(self.d_net.connectors[0](self.d_net.res1_s),
                             self.d_net.res1_t.detach(), margin) / batch_size / 4

        loss = loss_alter / 1000 * 3

        loss.backward()
        self.optimizer.step()
        return output, loss


def run_ab_distillation(s_net, t_net, **params):

    # check if this technique supports these kinds of models
    models = [params["student_name"], params["teacher_name"]]
    if not util.check_support(models, SUPPORTED):
        return 0.0

    d_net = AB_distill_Resnet(t_net, s_net).to(params["device"])

    # Distillation (Initialization)
    print("---------- Initialize AB Student Distillation-------")
    d_config = copy.deepcopy(params)
    d_config["test_name"] = "ab_distillation"
    d_config["epochs"] = DISTILL_EPOCHS
    d_trainer = DistillTrainer(s_net, d_net=d_net, config=d_config)
    d_trainer.train()
    s_net = d_trainer.d_net.s_net
    t_net = d_trainer.d_net.t_net
    # set the teacher net into evaluation mode again
    t_net.eval()

    # Student training
    print("---------- Training AB Student -------")
    s_config = copy.deepcopy(params)
    s_trainer = KDTrainer(s_net, t_net=t_net, config=s_config)
    best_s_acc = s_trainer.train()
    return best_s_acc
