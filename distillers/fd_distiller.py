import copy
import math
import torch
import torch.nn as nn
from trainer import BaseTrainer


def distillation_loss(source, target):
    loss = (source - target)**2 * (target > 0)
    return torch.abs(loss).sum()


def build_feature_connector(s_channel, t_channel):
    C = [
        nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0,
                  bias=False), nn.BatchNorm2d(t_channel)
    ]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


def build_connectors(s_channels, t_channels):
    channel_tuples = zip(s_channels, t_channels)
    return [build_feature_connector(s, t) for s, t in channel_tuples]


def get_convs(feat_layers):
    conv_layers = []
    for layer in feat_layers:
        if isinstance(layer, nn.Conv2d):
            conv_layers.append(layer)
    return conv_layers


def get_net_info(net):
    device = next(net.parameters()).device
    layers = list(net.children())
    # just get the conv layers
    feat_layers = get_convs(layers[:-1])
    linear = layers[-1]
    channels = []
    input_size = [[3, 28, 28]]
    x = [torch.rand(2, *in_size) for in_size in input_size]
    x = torch.Tensor(*x).to(device)
    for layer in feat_layers:
        x = layer(x)
        channels.append(x.shape[1])
    return feat_layers, linear, channels


def get_features(feat_layers, x):
    feats = []
    out = x
    for layer in feat_layers:
        out = layer(out)
        feats.append(out)
    return feats


class Distiller(nn.Module):
    def __init__(self, s_net, t_net):
        super(Distiller, self).__init__()

        self.s_feat_layers, self.s_linear, s_channels = get_net_info(s_net)
        self.t_feat_layers, self.t_linear, t_channels = get_net_info(t_net)
        connectors = build_connectors(s_channels, t_channels)
        self.connectors = nn.ModuleList(connectors)

        self.s_net = s_net
        self.t_net = t_net
        # freeze the layers of the teacher
        for param in self.t_net.parameters():
            param.requires_grad = False
        # set the teacher net into evaluation mode
        self.t_net.eval()
        self.t_net.train(mode=False)

    def forward(self, x, is_loss=False):
        t_feats = get_features(self.t_feat_layers, x)
        s_feats = get_features(self.s_feat_layers, x)

        loss_distill = 0
        for idx, s_feat in enumerate(s_feats):
            s_feat = self.connectors[idx](s_feat)
            kd_loss = distillation_loss(s_feat, t_feats[idx])
            loss_distill += kd_loss / 2 ** (len(t_feats) - idx - 1)

        s_out = self.s_net(x)
        if is_loss:
            return s_out, loss_distill
        return s_out


class FDTrainer(BaseTrainer):
    def __init__(self, s_net, train_config):
        super(FDTrainer, self).__init__(s_net, train_config)
        # the student net is the base net
        self.s_net = self.net.s_net
        self.d_net = self.net

    def calculate_loss(self, data, target):
        outputs, loss_distill = self.d_net(data, is_loss=True)
        loss_CE = self.loss_fun(outputs, target)
        loss = loss_CE + loss_distill.sum() / self.batch_size / 1000

        loss.backward()
        self.optimizer.step()
        return loss


def run_fd_distillation(s_net, t_net, **params):

    # Student training
    print("---------- Training FD Student -------")
    student_name = params["s_name"]
    s_train_config = copy.deepcopy(params)
    s_train_config["name"] = student_name
    s_net = Distiller(s_net, t_net).to(params["device"])
    s_trainer = FDTrainer(s_net, train_config=s_train_config)
    best_s_acc = s_trainer.train()

    return best_s_acc
