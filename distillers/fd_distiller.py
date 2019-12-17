import torch
import torch.nn as nn
import torch.nn.functional as torch_func
from trainer import Trainer


def get_layer_types(feat_layers):
    conv_layers = []
    for layer in feat_layers:
        if not isinstance(layer, nn.Linear):
            conv_layers.append(layer)
    return conv_layers


def get_net_info(net, as_module=False):
    device = next(net.parameters()).device
    if isinstance(net, nn.DataParallel):
        net = net.module
    layers = list(net.children())
    feat_layers = get_layer_types(layers)
    linear = layers[-1]
    channels = []
    input_size = [[3, 32, 32]]
    x = [torch.rand(2, *in_size) for in_size in input_size]
    x = torch.Tensor(*x).to(device)
    for layer in feat_layers:
        x = layer(x)
        channels.append(x.shape)
    if as_module:
        return nn.ModuleList(feat_layers), linear, channels
    return feat_layers, linear, channels


def set_last_layers(linear, last_channel, as_module=False):
    # assume that h_in and w_in are equal...
    c_in = last_channel[1]
    h_in = last_channel[2]
    w_in = last_channel[3]
    flat_size = c_in * h_in * w_in
    pooling = int((flat_size / linear.in_features)**(0.5))
    modules = [nn.ReLU(), nn.AvgPool2d((pooling)), nn.Flatten(), linear]
    if as_module:
        return nn.ModuleList(modules)
    return modules


def build_transformers(s_channels, t_channels):
    transfomers = []
    for idx, s_channel in enumerate(s_channels):
        t_channel = t_channels[idx]
        transformer = nn.Conv2d(s_channel[1], t_channel[1], kernel_size=1)
        transfomers.append(transformer)
    return nn.ModuleList(transfomers)


def get_layers(x, layers, linear, use_relu=True):
    layer_feats = []
    layer_feats_relu = []
    out = x
    for layer in layers:
        out = layer(out)
        layer_feats.append(out)
        if not isinstance(layer, nn.Conv2d):
            out = torch_func.relu(out)
            layer_feats_relu.append(out)
    out = torch_func.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = linear(out)
    return layer_feats, layer_feats_relu, out


def compute_feature_loss(s_feats, t_feats, batch_size):
    feature_loss = 0.0
    for idx, s_feat in enumerate(s_feats):
        t_feat = t_feats[idx]
        s_feat = torch_func.adaptive_max_pool2d(s_feat, (1, 1))
        t_feat = torch_func.adaptive_max_pool2d(t_feat, (1, 1))
        feature_loss += torch_func.pairwise_distance(s_feat, t_feat).max()
    return feature_loss


class Distiller(nn.Module):
    def __init__(self, s_net):
        super(Distiller, self).__init__()

        self.s_feat_layers, self.s_linear, s_channels = get_net_info(
            s_net, as_module=True)

    def forward(self, x, t_feats=None):
        s_feats, s_feats_relu, s_out = get_layers(
            x, self.s_feat_layers, self.s_linear)
        if t_feats:
            batch_size = s_out.shape[0]
            feature_loss = 0.0
            feature_loss += compute_feature_loss(s_feats, t_feats, batch_size)
            return s_out, feature_loss
        return s_out


class FDTrainer(Trainer):
    def __init__(self, s_net, t_net, config):
        super(FDTrainer, self).__init__(s_net, config)
        optim_params = [{"params": s_net.parameters()}]

        # Retrieve preconfigured optimizers and schedulers for all runs
        self.optimizer = self.optim_cls(optim_params, **self.optim_args)
        self.scheduler = self.sched_cls(self.optimizer, **self.sched_args)

        self.t_feat_layers, self.t_linear, t_channels = get_net_info(
            t_net, as_module=False)
        self.t_last = set_last_layers(
            self.t_linear, t_channels[-1], as_module=False)

    def calculate_loss(self, data, target):
        t_feats, t_feats_relu, t_out = get_layers(
            data, self.t_feat_layers, self.t_linear)
        s_out, feature_loss = self.net(data, t_feats)
        loss = 0.0
        loss += self.loss_fun(s_out, target)
        # loss += self.kd_loss(s_out, t_out, target)
        loss += feature_loss
        loss.backward()
        self.optimizer.step()
        return s_out, loss


def run_fd_distillation(s_net, t_net, **params):

    # Student training
    print("---------- Training FD Student -------")
    s_net = Distiller(s_net).to(params["device"])
    total_params = sum(p.numel() for p in s_net.parameters())
    print(f"FD distiller total parameters: {total_params}")
    s_trainer = FDTrainer(s_net, t_net, config=params)
    best_s_acc = s_trainer.train()

    return best_s_acc
