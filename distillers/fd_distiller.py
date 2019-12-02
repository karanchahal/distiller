import torch
import torch.nn as nn
from trainer import BaseTrainer


def build_feature_connector(s_channel, t_channel):
    c_in = s_channel[0]
    h_in = s_channel[1]
    w_in = s_channel[2]
    c_out = t_channel[0]
    h_out = t_channel[1]
    w_out = t_channel[2]
    conv = nn.Conv2d(c_in, c_out, kernel_size=1,
                     stride=1, padding=0, bias=False)
    connector = [
        conv,
        nn.BatchNorm2d(c_out),
    ]
    return nn.Sequential(*connector)


def build_connectors(s_channels, t_channels):
    # channel_tuples = zip(t_channels, s_channels)
    channel_tuples = []
    for idx, s_channel in enumerate(s_channels):
        channel_tuples.append((s_channel, t_channels[idx]))
    return [build_feature_connector(s, t) for s, t in channel_tuples]


def get_layer_types(feat_layers, types):
    conv_layers = []
    for layer in feat_layers:
        if not isinstance(layer, nn.Linear):
            conv_layers.append(layer)
    return conv_layers


def get_net_info(net):
    device = next(net.parameters()).device
    if isinstance(net, nn.DataParallel):
        net = net.module
    layers = list(net.children())
    # just get the conv layers
    types = [nn.Conv2d]
    feat_layers = get_layer_types(layers, types)
    linear = layers[-1]
    channels = []
    input_size = [[3, 32, 32]]
    x = [torch.rand(2, *in_size) for in_size in input_size]
    x = torch.Tensor(*x).to(device)
    for layer in feat_layers:
        x = layer(x)
        channels.append(x.shape[1:])
    return feat_layers, linear, channels


def get_features(feat_layers, x):
    feats = []
    out = x
    for layer in feat_layers:
        out = layer(out)
        feats.append(out)
    return feats


def distillation_loss(source, target, margin):
    loss = ((source - target)**2 * (target > 0).float())
    return torch.abs(loss).sum()


class Distiller(nn.Module):
    def __init__(self, s_net, t_net):
        super(Distiller, self).__init__()

        self.s_feat_layers, self.s_linear, s_channels = get_net_info(s_net)
        self.t_feat_layers, self.t_linear, t_channels = get_net_info(t_net)
        connectors = build_connectors(s_channels, t_channels)
        self.connectors = nn.ModuleList(connectors)

        self.s_net = s_net
        self.t_net = t_net

    def forward(self, x, is_loss=False):
        t_feats = get_features(self.t_feat_layers, x)
        s_feats = get_features(self.s_feat_layers, x)
        loss_distill = 0.0
        for idx, s_feat in enumerate(s_feats):
            t_feat = t_feats[idx]
            s_feat = self.connectors[idx](s_feat)
            loss = nn.KLDivLoss()(s_feat, t_feat)
            loss_distill += loss
        x = nn.functional.avg_pool2d(s_feats[-1], 4)
        x = x.view(x.size(0), -1)
        s_out = self.s_linear(x)
        if is_loss:
            return s_out, loss_distill / float(len(s_feats))
        return s_out


class FDTrainer(BaseTrainer):
    def __init__(self, s_net, config):
        super(FDTrainer, self).__init__(s_net, config)
        # the student net is the base net
        self.s_net = self.net.s_net
        self.d_net = self.net

    def calculate_loss(self, data, target):
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        output, loss_distill = self.d_net(data, is_loss=True)
        loss_CE = self.loss_fun(output, target)
        loss = lambda_ * loss_distill + loss_CE

        loss.backward()
        self.optimizer.step()
        return output, loss


def run_fd_distillation(s_net, t_net, **params):

    # Student training
    print("---------- Training FD Student -------")
    s_net = Distiller(s_net, t_net).to(params["device"])
    s_trainer = FDTrainer(s_net, config=params)
    best_s_acc = s_trainer.train()

    return best_s_acc
