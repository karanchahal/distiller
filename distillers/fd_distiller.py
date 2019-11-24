import copy
import math
import torch
import torch.nn as nn
from trainer import BaseTrainer


def distillation_loss(source, target):
    loss = (source - target)**2 * (target > 0).float()
    return torch.abs(loss).sum()


def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0,
                   bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


def get_net_info(net):
    device = next(net.parameters()).device
    layers = list(net.children())
    feat_layers = layers[:-1]
    linear = layers[-1]
    channels = []
    input_size = [[3, 28, 28]]
    x = [torch.rand(2, *in_size) for in_size in input_size]
    x = torch.Tensor(*x).to(device)
    for layer in feat_layers:
        x = layer(x)
        channels.append(x.shape[1])
    return feat_layers, linear, channels


class Distiller(nn.Module):
    def __init__(self, s_net, t_net):
        super(Distiller, self).__init__()

        self.t_feat_layers, self.t_linear, t_channels = get_net_info(t_net)
        self.s_feat_layers, self.s_linear, s_channels = get_net_info(s_net)

        channel_tuples = zip(t_channels, s_channels)
        self.Connectors = nn.ModuleList(
            [build_feature_connector(t, s) for t, s in channel_tuples])

        self.t_net = t_net
        self.s_net = s_net

    def get_features(self, feat_layers, x):
        feats = []
        out = x
        for layer in feat_layers:
            out = layer(out)
            feats.append(out)
        return feats

    def forward(self, x, is_loss=False):
        t_feats = self.get_features(self.t_feat_layers, x)
        s_feats = self.get_features(self.s_feat_layers, x)

        loss_distill = 0
        for idx, s_feat in enumerate(s_feats):
            s_feat = self.Connectors[idx](s_feat)
            kd_loss = distillation_loss(s_feat, t_feats[idx].detach())
            loss_distill += kd_loss / 2 ** (len(t_feats) - idx - 1)

        s_out = self.s_net(x)
        if is_loss:
            return s_out, loss_distill
        return s_out


class FDTrainer(BaseTrainer):
    def __init__(self, s_net, train_config):
        super(FDTrainer, self).__init__(s_net, train_config)
        # the student net is the base net
        self.s_net = self.net

    def calculate_loss(self, data, target):
        outputs, loss_distill = self.s_net(data, is_loss=True)
        loss_CE = self.loss_fun(outputs, target)
        loss = loss_CE + loss_distill.sum() / self.batch_size / 1000

        loss.backward()
        self.optimizer.step()
        return loss


def run_fd_distillation(s_net, t_net, **params):

    # Student training
    # Define loss and the optimizer
    print("---------- Training FD Student -------")
    student_name = params["s_name"]
    s_train_config = copy.deepcopy(params)
    s_train_config["name"] = student_name
    s_net = Distiller(s_net, t_net).cuda()
    s_trainer = FDTrainer(s_net, train_config=s_train_config)
    best_s_acc = s_trainer.train()

    return best_s_acc
