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

    connector = []
    if h_in < h_out or w_in < w_out:
        scale = int(h_out / h_in)
        upsampler = nn.Upsample(scale_factor=scale)
        connector.append(upsampler)
    stride = int(h_in / h_out)
    conv = nn.Conv2d(c_in, c_out, kernel_size=1,
                     stride=stride, padding=0, bias=False)
    connector.append(conv)
    connector.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*connector)


def build_connectors(s_channels, t_channels):
    # channel_tuples = zip(t_channels, s_channels)
    channel_tuples = []
    len_s_channels = len(s_channels)
    len_t_channels = len(t_channels)
    for idx in range(len_t_channels):
        t_channel = t_channels[idx]
        s_idx = idx
        if idx > len_s_channels - 1:
            s_idx = len_s_channels - 1
        channel_tuples.append((s_channels[s_idx], t_channel))
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


def compute_last_layer(linear, last_channel):
    # assume that h_in and w_in are equal...
    c_in = last_channel[0]
    h_in = last_channel[1]
    w_in = last_channel[2]
    flat_size = c_in * h_in * w_in
    pooling = int((flat_size / linear.in_features)**(0.5))
    module = nn.Sequential(
        nn.ReLU(),
        nn.AvgPool2d(pooling),
        nn.Flatten(),
        linear)
    return module


class Distiller(nn.Module):
    def __init__(self, s_net, t_net, batch_size=128, device="cuda"):
        super(Distiller, self).__init__()

        self.s_feat_layers, self.s_linear, s_channels = get_net_info(s_net)
        self.t_feat_layers, self.t_linear, t_channels = get_net_info(t_net)
        connectors = build_connectors(s_channels, t_channels)
        self.connectors = nn.ModuleList(connectors)

        # infer the necessary pooling based on the last feature size
        self.last = compute_last_layer(self.s_linear, s_channels[-1])
        self.s_net = s_net
        self.t_net = t_net
        self.batch_size = batch_size
        self.y = torch.ones([batch_size], device=device)

    def compute_feature_loss(self, s_feats, t_feats):
        loss_distill = 0.0
        for idx in range(len(s_feats)):
            t_feat = t_feats[idx]
            s_idx = idx
            if s_idx > len(s_feats) - 1:
                s_idx = len(s_feats) - 1
            s_feat = s_feats[s_idx]
            connector = self.connectors[idx]
            s_feat = connector(s_feat)
            s_feat = s_feat.reshape((self.batch_size, -1))
            t_feat = t_feat.reshape((self.batch_size, -1))
            loss = nn.CosineEmbeddingLoss()(s_feat, t_feat, self.y)
            loss_distill += loss
        return loss_distill

    def forward(self, x, is_loss=False):
        s_feats = get_features(self.s_feat_layers, x)
        t_feats = get_features(self.t_feat_layers, x)
        s_out = self.last(s_feats[-1])
        loss_distill = self.compute_feature_loss(s_feats, t_feats)
        if is_loss:
            return s_out, loss_distill
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
        loss = loss_CE * (1 + loss_distill)
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
