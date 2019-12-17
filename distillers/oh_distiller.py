# Original Repo:
# https://github.com/clovaai/overhaul-distillation
# @inproceedings{heo2019overhaul,
#  title={A Comprehensive Overhaul of Feature Distillation},
#  author={Heo, Byeongho and Kim, Jeesoo and Yun, Sangdoo and Park, Hyojin
#  and Kwak, Nojun and Choi, Jin Young},
#  booktitle = {International Conference on Computer Vision (ICCV)},
#  year={2019}
# }

import math
import torch
import torch.nn as nn
from scipy.stats import norm
from trainer import BaseTrainer
import util


SUPPORTED = ["resnet8", "resnet14", "resnet20", "resnet26",
             "resnet32", "resnet44", "resnet56", "resnet10",
             "resnet18", "resnet34", "resnet50", "resnet101",
             "resnet152", ]


def distillation_loss(source, target, margin):
    loss = ((source - margin)**2 * ((source > margin) & (target <= margin)).float() +
            (source - target)**2 * ((source > target) & (target > margin) & (target <= 0)).float() +
            (source - target)**2 * (target > 0).float())
    return torch.abs(loss).sum()


def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) /
                          math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)


class Distiller(nn.Module):
    def __init__(self, s_net, t_net):
        super(Distiller, self).__init__()

        if isinstance(t_net, nn.DataParallel):
            t_channels = t_net.module.get_channel_num()
            teacher_bns = t_net.module.get_bn_before_relu()
        else:
            teacher_bns = t_net.get_bn_before_relu()
            t_channels = t_net.get_channel_num()

        if isinstance(s_net, nn.DataParallel):
            s_channels = s_net.module.get_channel_num()
        else:
            s_channels = s_net.get_channel_num()

        self.connectors = nn.ModuleList(
            [build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (
                i + 1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.s_net = s_net

    def forward(self, x, t_feats=None):

        s_feats, s_pool, s_out = self.s_net(x, is_feat=True, use_relu=False)

        if t_feats:
            s_feats_num = len(s_feats)
            loss_distill = 0
            for i in range(s_feats_num):
                s_feats[i] = self.connectors[i](s_feats[i])
                loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i + 1))) \
                    / 2 ** (s_feats_num - i - 1)
            return s_out, loss_distill
        return s_out


class OHTrainer(BaseTrainer):
    def __init__(self, d_net, t_net, config):
        # the student net is the base net
        super(OHTrainer, self).__init__(d_net, config)
        # decouple the teacher from the student
        self.t_net = t_net
        optim_params = [{"params": self.net.parameters()}]

        # Retrieve preconfigured optimizers and schedulers for all runs
        self.optimizer = self.optim_cls(optim_params, **self.optim_args)
        self.scheduler = self.sched_cls(self.optimizer, **self.sched_args)

    def calculate_loss(self, data, target):
        t_feats, t_pool, t_out = self.t_net(data, is_feat=True, use_relu=False)

        s_out, loss_distill = self.net(data, t_feats)
        loss_CE = self.loss_fun(s_out, target)

        loss = loss_CE + loss_distill.sum() / self.batch_size / 1000

        loss.backward()
        self.optimizer.step()
        return t_out, loss


def run_oh_distillation(s_net, t_net, **params):

    # check if this technique supports these kinds of models
    models = [params["student_name"], params["teacher_name"]]
    if not util.check_support(models, SUPPORTED):
        return 0.0

    # Student training
    # Define loss and the optimizer
    print("---------- Training OKD Student -------")
    d_net = Distiller(s_net, t_net).to(params["device"])
    s_trainer = OHTrainer(d_net, t_net, config=params)
    best_s_acc = s_trainer.train()

    return best_s_acc
